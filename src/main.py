from fastapi import FastAPI
from .serving import get_recommendations
from apscheduler.schedulers.background import BackgroundScheduler
import logging
from .config import TRAINING_INTERVAL_MINUTES
from .trainer.train import train_full_pipeline
import uvicorn

app = FastAPI()

# Proper logging so you actually see what went wrong
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rec2")

# ─── SAFE BACKGROUND TASK ─────────────────────
def safe_train_incremental():
    try:
        train_full_pipeline()
        logger.info("Incremental training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)

def schedule_training():
    scheduler = BackgroundScheduler()
    scheduler.add_job(safe_train_incremental, 'interval', minutes=TRAINING_INTERVAL_MINUTES)
    scheduler.start()


@app.on_event("startup")
async def startup_event():
    schedule_training()

@app.get("/recommend/{user_id}")
def recs(user_id: str, limit: int = 10):
    try:
        result = get_recommendations(user_id, limit)
        return {"user_id": user_id, "recommendations": result}
    except Exception as e:
        logger.error(f"Recommendation failed for {user_id}: {e}")
        raise  # will return e

@app.post("/train")
def trigger_train():
    safe_train_incremental()
    return {"status": "training queued"}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "hybrid-recommender"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)