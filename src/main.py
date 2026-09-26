from fastapi import FastAPI
from .serving import build_cooccurrence, filter_wrong_phrases, get_classifier, get_content_keywords, get_recommendations, has_non_stopword, kw_model, phrase_coherence_score, restore_phrases_casing
from apscheduler.schedulers.background import BackgroundScheduler
import logging
from .config import TRAINING_INTERVAL_MINUTES
from .trainer.train import train_full_pipeline
import uvicorn
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Proper logging so you actually see what went wrong
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Kwonrec")

class ClassificationRequest(BaseModel):
    text: str
    labels: List[str]

class ExtractKeywordRequest(BaseModel):
    text: str

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

@app.post("/classify")
async def classify_text(request: ClassificationRequest):
    try:
        classifier = get_classifier()
        if classifier is None:
            raise
        logger.info("Trying to get content topic...")
        result = classifier(
            request.text, 
            request.labels, 
            hypothesis_template="This text is about {}.",
            multi_label=False
        )
        return {
            "label": result['labels'][0],
            "score": result['scores'][0],
            # "all_results": dict(zip(result['labels'], result['scores']))
        }
    except Exception as e:
        logger.error("Unable to get content topic...")
        raise

@app.post("/keywords")
async def extract_keywords(req: ExtractKeywordRequest):
    try:
        logger.info("Trying to get content keywords...")

        keywords = get_content_keywords(req.text)

        logger.info(f"Total keywords extracted {len(keywords)}")

        return { "keywords": keywords }

    except Exception as e:
        logger.error("Unable to get content keywords...")
        raise




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