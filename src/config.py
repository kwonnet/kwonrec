import os
from dotenv import load_dotenv

load_dotenv()

# os.environ['TF_USE_LEGACY_KERAS'] = '1'

CLICKHOUSE_URL = os.getenv("CLICKHOUSE_URL")
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", 8443))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD")
CLICKHOUSE_DB = os.getenv("CLICKHOUSE_DB")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

TRAINING_INTERVAL_MINUTES = int(os.getenv("TRAINING_INTERVAL_MINUTES", 200))

EXPLORATION_RATE = float(os.getenv("EXPLORATION_RATE", 0.1))

# Hyperparameters

BATCH_SIZE = 128

EMBEDDING_DIM = 128

NUM_EPOCHS = 10 

LEARNING_RATE = 0.005 # LEARNING_RATE is higher for Adagrad (e.g., 0.1 or 0.5) than for Adam (e.g., 0.001).

RATING_WEIGHT = 1.2

RETRIEVAL_WEIGHT = 1.8

# checkpoints 🚀🏆

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/app/checkpoints")

MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "/app/checkpoints/model.weights.h5")

# for serving

SERVING_DIR = os.getenv("SERVING_DIR", "/app/serving_model")

MODEL_SERVING_PATH = os.getenv("MODEL_SERVING_PATH", "/app/serving_model/model")

MAPPINGS_PATH = os.getenv("MAPPINGS_PATH", "/app/serving_model/mappings.pkl")

SCANN_INDEX_PATH = os.getenv("SCANN_INDEX_PATH", "/app/serving_model/scann")

#  Logs

LOG_DIR = os.getenv("LOG_DIR", "/app/logs")

# Ensure checkpoint directories exist
os.makedirs(os.path.dirname(CHECKPOINT_DIR), exist_ok=True)
# Ensure serving_model directory exists
os.makedirs(os.path.dirname(SERVING_DIR), exist_ok=True)
# Ensure logs directory exists
os.makedirs(os.path.dirname(LOG_DIR), exist_ok=True)