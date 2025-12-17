FROM python:3.10-slim

WORKDIR /app

# ----------------------------------------------------------------------
# 1. SET THE ENVIRONMENT VARIABLE
# This ENV instruction makes the variable available to all subsequent 
# RUN, CMD, and ENTRYPOINT commands, including the 'pip install' step.
# ----------------------------------------------------------------------
ENV TF_USE_LEGACY_KERAS="1"

COPY requirements.txt .

# 2. RUN pip install (The variable is available here for any build steps)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# FastAPI

EXPOSE 8000 

# TensorBoard

EXPOSE 6006

# Start both services:
# - TensorBoard in background (writes to /app/logs/fit)
# - Uvicorn in foreground (keeps container alive)
CMD ["bash", "-c", \
     "tensorboard --logdir /app/logs/fit --host 0.0.0.0 --port 6006 & \
      uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload"]
