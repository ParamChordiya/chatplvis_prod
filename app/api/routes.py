"""ChatPLVis Flask Blueprint — index and chatbot routes."""
from __future__ import annotations

import logging
import traceback

import json

from flask import Blueprint, Response, jsonify, render_template, request, stream_with_context
from sklearn.decomposition import PCA

from app.api.schemas import ChatRequest, PlotState
from app.core.config import Settings
from app.core.exceptions import (
    DataError,
    EmbeddingError,
    LLMError,
    RateLimitedError,
)
from app.services.data_service import DataService
from app.services.embedding_service import EmbeddingService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ordered column list shown in the dropdown (matches original app.py order)
# ---------------------------------------------------------------------------
_COLUMN_OPTIONS: list[str] = [
    'Counts_1st_normalized',
    'Counts_2nd_normalized',
    'Counts_3rd_normalized',
    'Total_Counts_normalized',
    'Rank_normalized',
    'Counts_1st',
    'Counts_2nd',
    'Counts_3rd',
    'Total_Counts',
    'Rank',
]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_blueprint(
    data_svc: DataService,
    embed_svc: EmbeddingService,
    rag_svc: RAGService,
    settings: Settings,
) -> Blueprint:
    """
    Build and return the 'main' blueprint with all routes closed over the
    injected service instances.  No global state is kept in this module.

    Args:
        data_svc:  DataService for proteome data loading and plot building.
        embed_svc: EmbeddingService for FAISS index construction (3D PCA).
        rag_svc:   RAGService for chatbot answer generation.
        settings:  Frozen Settings instance carrying whitelists and limits.

    Returns:
        A Flask Blueprint named 'main'.
    """
    bp = Blueprint('main', __name__)

    # -----------------------------------------------------------------------
    # Internal validation helper (stateless, pure)
    # -----------------------------------------------------------------------

    def _validated(value: str, whitelist: frozenset, default: str) -> str:
        """Return *value* if it is in *whitelist*, else log and return *default*."""
        if value in whitelist:
            return value
        logger.warning(
            "Rejected invalid form value %r; falling back to %r",
            value,
            default,
        )
        return default

    # -----------------------------------------------------------------------
    # GET / POST  /
    # -----------------------------------------------------------------------

    @bp.route('/', methods=['GET', 'POST'])
    def index():  # type: ignore[return]
        """Render the main proteome visualisation page."""
        try:
            # ---- 1. Parse form fields with safe defaults ----
            raw_col  = request.form.get('sel_col',     settings.default_col)
            raw_comp = request.form.get('sel_comp',    settings.default_comp)
            raw_plot = request.form.get('plot_type',   settings.default_plot)
            raw_info = request.form.get('info_source', settings.default_info)

            # ---- 2. Validate against whitelists (silent fallback) ----
            sel_col     = _validated(raw_col,  settings.valid_columns,      settings.default_col)
            sel_comp    = _validated(raw_comp, settings.valid_comparisons,   settings.default_comp)
            plot_type   = _validated(raw_plot, settings.valid_plot_types,    settings.default_plot)
            info_source = _validated(raw_info, settings.valid_info_sources,  settings.default_info)

            # ---- 3. Build plot state ----
            state = PlotState(
                sel_col=sel_col,
                sel_comp=sel_comp,
                plot_type=plot_type,
                info_source=info_source,
            )

            # ---- 4. Delegate data loading and node construction ----
            nodes, _tb_df = data_svc.build_plot_data(state)

            # ---- 5. 3D PCA override (EmbeddingError is recoverable) ----
            if plot_type.startswith('3D'):
                try:
                    _faiss_index, embeddings = embed_svc.get_or_build_index(info_source)
                    coords = PCA(n_components=3).fit_transform(embeddings)
                    logger.info(
                        "PCA computed for 3D plot (%d points)", len(coords)
                    )
                    for i, node in enumerate(nodes):
                        if i < len(coords):
                            node.x = float(coords[i][0])
                            node.y = float(coords[i][1])
                            node.z = float(coords[i][2])
                except EmbeddingError:
                    logger.warning(
                        "EmbeddingError during 3D PCA — falling back to 2D",
                        exc_info=True,
                    )
                    # nodes retain their UMAP x/y; z stays None → 2D rendering

            # ---- 6. Serialise nodes ----
            nodes_dicts = [n.to_dict() for n in nodes]

            # ---- 7. Build dropdown option lists ----
            column_options = list(_COLUMN_OPTIONS)  # copy to avoid mutation
            comparison_options = sorted(
                settings.valid_comparisons,
                key=lambda x: (x != 'All proteomes', x),
            )
            plot_options = ['2D UMAP Based', '3D PCA Based']
            info_options = ['Function [CC]', 'Abstracts']

            # ---- 8. Render ----
            return render_template(
                'index.html',
                nodes=nodes_dicts,
                edges=[],
                sel_col=sel_col,
                sel_comp=sel_comp,
                column_options=column_options,
                comparison_options=comparison_options,
                plot_type=plot_type,
                plot_options=plot_options,
                info_source=info_source,
                info_options=info_options,
            )

        # ---- 9. DataError → 503 ----
        except DataError:
            logger.error("DataError in index()", exc_info=True)
            return jsonify({'error': 'Data unavailable. Please try again later.'}), 503

        # ---- 10. EmbeddingError that bubbled past the 3D block → 503 ----
        except EmbeddingError:
            logger.error("EmbeddingError in index()", exc_info=True)
            return jsonify({'error': 'Embedding service unavailable. Please try again later.'}), 503

        # ---- 11. Catch-all → 500 ----
        except Exception:
            logger.error("Unhandled exception in index()\n%s", traceback.format_exc())
            return jsonify({'error': 'Internal server error. Please check server logs.'}), 500

    # -----------------------------------------------------------------------
    # POST  /chatbot
    # -----------------------------------------------------------------------

    @bp.route('/chatbot', methods=['POST'])
    def chatbot():  # type: ignore[return]
        """Receive a chat request and return an HTML-formatted RAG response."""

        # ---- 1. Parse body ----
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'error': 'Request body must be valid JSON.'}), 400

        # ---- 2. Validate message ----
        message = data.get('message', '')
        if not isinstance(message, str):
            return jsonify({'error': "'message' must be a string."}), 400
        if len(message) > settings.max_message_length:
            return jsonify({
                'error': (
                    f"'message' exceeds maximum length of "
                    f"{settings.max_message_length} characters."
                )
            }), 400

        # ---- 3. Validate node_ids ----
        node_ids = data.get('node_ids', [])
        if not isinstance(node_ids, list):
            return jsonify({'error': "'node_ids' must be a list."}), 400
        if len(node_ids) > settings.max_node_ids:
            return jsonify({
                'error': f"Too many node IDs (max {settings.max_node_ids})."
            }), 400

        node_ids_int: list[int] = []
        for raw_id in node_ids:
            try:
                val = int(raw_id)
            except (TypeError, ValueError):
                return jsonify({'error': f"Invalid node_id value: {raw_id!r}"}), 400
            if not (0 <= val <= 999_999):
                return jsonify({'error': f"node_id out of range: {val}"}), 400
            node_ids_int.append(val)

        # ---- 4. Empty selection fast-path ----
        if not node_ids_int:
            return jsonify({'message': 'No node was selected.'})

        # ---- 5. Validate include_similar ----
        include_similar = data.get('include_similar', True)
        if not isinstance(include_similar, bool):
            return jsonify({'error': "'include_similar' must be a boolean."}), 400

        # ---- 6. Validate info_source ----
        raw_info = data.get('info_source', settings.default_info)
        if not isinstance(raw_info, str):
            raw_info = settings.default_info
        info_source = _validated(raw_info, settings.valid_info_sources, settings.default_info)

        # ---- 7. Build validated request object ----
        req = ChatRequest(
            node_ids=tuple(node_ids_int),
            message=message,
            include_similar=include_similar,
            info_source=info_source,
        )

        # ---- 8. Log ----
        logger.info(
            "Chatbot: node_ids=%s info=%r msg_len=%d",
            node_ids_int,
            info_source,
            len(message),
        )

        # ---- 9–13. Call RAG service with structured error handling ----
        try:
            response_html = rag_svc.answer(req)

        # ---- 10. Rate-limited → 429 ----
        except RateLimitedError:
            logger.warning("RAG service rate-limited", exc_info=True)
            return jsonify({
                'error': (
                    'The AI service is currently rate-limited. '
                    'Please wait a moment and try again.'
                )
            }), 429

        # ---- 11. Other LLM errors → 502 ----
        except LLMError:
            logger.error("LLMError in chatbot()", exc_info=True)
            return jsonify({'error': 'Language model service error.'}), 502

        # ---- 12. Data / embedding errors → 503 ----
        except (DataError, EmbeddingError):
            logger.error("Data or EmbeddingError in chatbot()", exc_info=True)
            return jsonify({'error': 'Data service unavailable. Please try again later.'}), 503

        # ---- 13. Catch-all → 500 ----
        except Exception:
            logger.error(
                "Unhandled exception in chatbot()\n%s", traceback.format_exc()
            )
            return jsonify({'error': 'Chatbot encountered an internal error.'}), 500

        # ---- 14. Return response ----
        return jsonify({'message': response_html})

    # -----------------------------------------------------------------------
    # POST  /chatbot/stream  — SSE streaming response
    # -----------------------------------------------------------------------

    @bp.route('/chatbot/stream', methods=['POST'])
    def chatbot_stream():
        """
        Same validation as /chatbot, then streams raw LLM tokens via SSE.
        The browser accumulates chunks and renders markdown client-side on completion.

        SSE event format:
          data: {"t": "<text_chunk>"}\n\n   — text token
          data: {"e": "<message>"}\n\n       — error
          data: [DONE]\n\n                   — stream finished
        """
        # ---- Parse + validate (mirrors /chatbot) ----
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'error': 'Request body must be valid JSON.'}), 400

        message = data.get('message', '')
        if not isinstance(message, str) or len(message) > settings.max_message_length:
            return jsonify({'error': 'Invalid or too-long message.'}), 400

        node_ids_raw = data.get('node_ids', [])
        if not isinstance(node_ids_raw, list):
            return jsonify({'error': "'node_ids' must be a list."}), 400

        node_ids_int: list[int] = []
        for raw_id in node_ids_raw:
            try:
                val = int(raw_id)
            except (TypeError, ValueError):
                return jsonify({'error': f"Invalid node_id: {raw_id!r}"}), 400
            if not (0 <= val <= 999_999):
                return jsonify({'error': f"node_id out of range: {val}"}), 400
            node_ids_int.append(val)

        if not node_ids_int:
            # Fast-path: no selection
            def _empty():
                yield f"data: {json.dumps({'t': 'No proteins selected.'})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(_empty()), mimetype='text/event-stream',
                            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

        include_similar = data.get('include_similar', True)
        if not isinstance(include_similar, bool):
            include_similar = True

        raw_info = data.get('info_source', settings.default_info)
        if not isinstance(raw_info, str):
            raw_info = settings.default_info
        info_source = _validated(raw_info, settings.valid_info_sources, settings.default_info)

        from app.api.schemas import ChatRequest
        req = ChatRequest(
            node_ids=tuple(node_ids_int),
            message=message,
            include_similar=include_similar,
            info_source=info_source,
        )

        logger.info(
            "Stream: node_ids=%s info=%r msg_len=%d",
            node_ids_int[:5], info_source, len(message),
        )

        def generate():
            try:
                for chunk in rag_svc.answer_stream(req):
                    yield f"data: {json.dumps({'t': chunk})}\n\n"
            except RateLimitedError as exc:
                yield f"data: {json.dumps({'e': str(exc)})}\n\n"
            except LLMError as exc:
                yield f"data: {json.dumps({'e': str(exc)})}\n\n"
            except Exception:
                logger.error("Unhandled error in chatbot_stream\n%s", traceback.format_exc())
                yield f"data: {json.dumps({'e': 'Internal server error.'})}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
        )

    return bp
