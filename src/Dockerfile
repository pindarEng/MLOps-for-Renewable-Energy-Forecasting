FROM python:3.12

WORKDIR /app

COPY src/requirements.txt .

COPY src/serve.py .

COPY dataset/processed/final_dataset.csv ./dataset/processed/

COPY dataset/fastapi-test/kafka_testing_dataset.csv ./dataset/fastapi-test/

COPY src/temp/latest_batches.csv ./src/temp/
COPY src/temp/drift_init.html ./src/temp/

COPY src/serve_model_svr_lag /app/model/svr_lag
COPY src/serve_model_xgb_lag /app/model/xgb_lag
COPY src/serve_model_randomForest_lag /app/model/randomForest_lag

COPY src/serve_model_svr_nolag /app/model/svr_nolag
COPY src/serve_model_xgb_nolag /app/model/xgb_nolag
COPY src/serve_model_randomForest_nolag /app/model/randomForest_nolag


RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
