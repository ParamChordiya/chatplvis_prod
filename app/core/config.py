from __future__ import annotations

from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class Settings:
    # Paths
    csv_path: str = field(default_factory=lambda: os.getenv('CSV_PATH', 'uploads/mycobacterium_proteome_df.csv'))
    counts_path: str = field(default_factory=lambda: os.getenv('COUNTS_PATH', 'data/counts_all_stages_MAGECK_with_ES.xlsx'))

    # Server
    port: int = field(default_factory=lambda: int(os.getenv('PORT', '5000')))
    debug: bool = field(default_factory=lambda: os.getenv('FLASK_DEBUG', 'false').lower() == 'true')

    # ML model
    embedding_model: str = field(default_factory=lambda: os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2'))

    # OpenAI (used when OPENAI_API_KEY is set)
    openai_model: str = 'gpt-4o-mini'

    # Ollama fallback (used when OPENAI_API_KEY is absent and Ollama is reachable)
    ollama_base_url: str = field(default_factory=lambda: os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'))
    ollama_model: str = field(default_factory=lambda: os.getenv('OLLAMA_MODEL', 'llama3.2'))

    # RAG retrieval
    faiss_top_k: int = 12
    bm25_top_k: int = 12
    rrf_k: int = 60
    context_size: int = 6  # final top-K after RRF merge

    # Caches
    response_cache_maxsize: int = 128
    response_cache_ttl: int = 3600
    query_cache_maxsize: int = 512
    df_cache_maxsize: int = 4

    # Validation whitelists
    valid_columns: frozenset = field(default_factory=lambda: frozenset({
        'Total_Counts', 'Counts_1st', 'Counts_2nd', 'Counts_3rd', 'Rank',
        'Total_Counts_normalized', 'Counts_1st_normalized',
        'Counts_2nd_normalized', 'Counts_3rd_normalized', 'Rank_normalized',
    }))
    valid_comparisons: frozenset = field(default_factory=lambda: frozenset({
        'All proteomes', 'Mycobacterium tuberculosis',
        'vs smegmatis', 'vs marinum', 'vs leprae',
        'vs kansasii', 'vs intracellulare', 'vs fortuitum', 'vs bovis',
    }))
    valid_plot_types: frozenset = field(default_factory=lambda: frozenset({
        '2D UMAP Based', '3D PCA Based',
    }))
    valid_info_sources: frozenset = field(default_factory=lambda: frozenset({
        'Function [CC]', 'Abstracts',
    }))

    # Defaults for form fields
    default_col: str = 'Total_Counts'
    default_comp: str = 'All proteomes'
    default_plot: str = '2D UMAP Based'
    default_info: str = 'Function [CC]'

    # RAG context limits
    max_selected_context: int = 20   # max selected proteins included with full detail
    max_context_chars: int = 16_000  # hard cap on total context characters sent to LLM

    # Request limits
    max_node_ids: int = 100_000
    max_message_length: int = 1000


# Module-level default instance — import this everywhere
settings = Settings()
