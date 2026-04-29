"""ChatPLVis application factory."""
from __future__ import annotations

import logging

from flask import Flask


def create_app(settings=None) -> Flask:
    """
    Flask application factory.

    Wire all services with dependency injection, attach them to
    app.extensions, and register the main blueprint.

    Args:
        settings: Pre-built Settings instance.  When None the module-level
                  default from app.core.config is used.

    Returns:
        A fully-configured Flask application instance.
    """
    # Lazy imports — keep service deps (sentence_transformers, faiss, etc.)
    # out of the module-level scope so submodules can be imported independently.
    from app.core.config import Settings, settings as default_settings
    from app.utils.cache import RAGCache
    from app.services.data_service import DataService
    from app.services.embedding_service import EmbeddingService
    from app.services.llm_service import LLMService
    from app.services.rag_service import RAGService

    if settings is None:
        settings = default_settings

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )
    logger = logging.getLogger(__name__)
    logger.info(
        "Creating ChatPLVis app (debug=%s, port=%d)",
        settings.debug,
        settings.port,
    )

    # ------------------------------------------------------------------
    # Flask app
    # ------------------------------------------------------------------
    app = Flask(
        __name__,
        template_folder='../templates',   # relative to app/ package
        static_folder='../static',
    )

    # ------------------------------------------------------------------
    # Wire services (dependency-injection order matters)
    # ------------------------------------------------------------------
    data_svc = DataService(settings)
    embed_svc = EmbeddingService(settings)
    llm_svc = LLMService(settings)
    cache = RAGCache(
        maxsize=settings.response_cache_maxsize,
        ttl_seconds=settings.response_cache_ttl,
    )
    rag_svc = RAGService(data_svc, embed_svc, llm_svc, settings, cache)

    # ------------------------------------------------------------------
    # Store on app.extensions for testing / introspection
    # ------------------------------------------------------------------
    app.extensions['data_service'] = data_svc
    app.extensions['embed_service'] = embed_svc
    app.extensions['llm_service'] = llm_svc
    app.extensions['rag_service'] = rag_svc
    app.extensions['settings'] = settings

    # ------------------------------------------------------------------
    # Register blueprint
    # ------------------------------------------------------------------
    from app.api.routes import create_blueprint
    bp = create_blueprint(data_svc, embed_svc, rag_svc, settings)
    app.register_blueprint(bp)

    # ------------------------------------------------------------------
    # Background pre-warm: load model + FAISS index so first chat is fast
    # ------------------------------------------------------------------
    import threading

    def _prewarm() -> None:
        try:
            logger.info("Pre-warming: loading model and FAISS index…")
            embed_svc.get_or_build_index(settings.default_info)
            logger.info("Pre-warm complete.")
        except Exception:
            logger.debug("Pre-warm failed (non-fatal).", exc_info=True)

    threading.Thread(target=_prewarm, daemon=True).start()

    logger.info("ChatPLVis app created successfully.")
    return app
