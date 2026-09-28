import os

def load_dotenv(path: str = None):
    if path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, ".env")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

load_dotenv()

# Configuration settings for Agent0Mem
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_EMBED_URL = os.environ.get("DASHSCOPE_EMBED_URL", "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding")

DATA_DIR = os.environ.get("AGENT0MEM_DATA_DIR", "/var/lib/agent0mem/data" if os.path.exists("/var/lib/agent0mem") else os.path.join(os.getcwd(), "data"))
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "memories.sqlite3")

# Engine tuning hyperparameters
DEFAULT_TOP_K = 100
BM25_WEIGHT = 0.50
DENSE_WEIGHT = 0.50
RECENCY_BOOST_MAX = 0.20
ENABLE_LLM_EXTRACTION = True
