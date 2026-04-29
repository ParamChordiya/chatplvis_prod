"""
Entry point — run with:  python run.py

LLM backend (auto-detected, first match wins):
  1. OpenAI  — set OPENAI_API_KEY in .env
  2. Ollama  — install Ollama (https://ollama.com), run `ollama pull llama3.2`,
               leave OPENAI_API_KEY unset. Customize via:
               OLLAMA_BASE_URL  (default: http://localhost:11434)
               OLLAMA_MODEL     (default: llama3.2)

Other environment variables:
  CSV_PATH      Path to proteome CSV  (default: uploads/mycobacterium_proteome_df.csv)
  COUNTS_PATH   Path to counts Excel file
  PORT          Port to listen on     (default: 5000)
  FLASK_DEBUG   Set to 'true' for debug mode
"""
import os
# All of these must be set before any PyTorch / HuggingFace import.
# On macOS, conda's OpenMP + PyTorch's internal thread pool segfaults at the
# first forward pass. Capping every threading layer to 1 prevents this.
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

from dotenv import load_dotenv
load_dotenv()

from app import create_app
from app.core.config import settings

app = create_app(settings)

if __name__ == '__main__':
    app.run(debug=settings.debug, port=settings.port)
