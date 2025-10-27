from kafka import KafkaConsumer
import json
import pandas as pd
import time
from datetime import datetime
import mlflow.artifacts
import os
import boto3
from io import BytesIO
import numpy as np
import mlflow
from prometheus_client import start_http_server, Gauge, Counter, Histogram
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Prometheus metrics 
PREDICTION_MSE = Gauge('ml_prediction_mse', 'Mean Squared Error of the latest prediction batch')
PREDICTION_RMSE = Gauge('ml_prediction_rmse', 'Root Mean Squared Error of the latest prediction batch')
PREDICTION_R2 = Gauge('ml_prediction_r2', 'R2 Score of the latest prediction batch')
PREDICITON_MAE = Gauge('ml_prediction_mae', 'Mean Absolute Error of the latest prediction batch')

MESSAGES_PROCESSED_TOTAL = Counter('kafka_messages_processed_total', 'Total number of Kafka messages processed')
BATCHES_PROCESSED_TOTAL = Counter('ml_batches_processed_total', 'Total number of batches processed for prediction')
PREDICTION_ERRORS_TOTAL = Counter('ml_prediction_errors_total', 'Total count of errors during prediction')

BATCH_PROCESSING_TIME_SECONDS = Histogram('ml_batch_processing_time_seconds', 'Histogram of batch processing time')

PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", 8000))


KAFKA_TOPIC = 'wind-data-testing-12'
BATCH_SIZE = 168
CSV_KEY_PREFIX = "batches/"
BUCKET_NAME = "datasets-mlops"



def get_minio_client():
    return boto3.client(
        's3',
        endpoint_url="http://localhost:9000",
        aws_access_key_id="IFSYL3fy8jSmMfK2To6G",
        aws_secret_access_key="WzhnEeyrtGmjtxE5QtSzK7ka0fZhgAgnqoykxm5g")

def load_model():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = "models:/model_xgb_model_scaler_batches_nolag/latest"
    
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info("MLflow model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}", exc_info=True)
        return None

def save_batch_to_minio(df, s3, batch_number):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    local_file = f"new_data_batch_{timestamp}.csv"
    minio_key = f"{CSV_KEY_PREFIX}batch_{timestamp}_final-{batch_number}.csv"
    
    df.to_csv(local_file, index=False)
    try:
        s3.upload_file(local_file, BUCKET_NAME, minio_key)
        logging.info(f"Batch saved to MinIO: {minio_key}")
    finally:
        os.remove(local_file)


# def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
#     df['lag_1'] = df['DE_wind_generation_actual'].shift(1)
#     df['lag_2'] = df['DE_wind_generation_actual'].shift(2)
#     df['lag_3'] = df['DE_wind_generation_actual'].shift(3)

#     df = df.dropna(subset=['lag_1', 'lag_2', 'lag_3'])
#     df = df.fillna(df.mean(numeric_only=True))
#     return df


def log_and_update_metrics(predictions, y_true):
    if len(y_true) == 0 or len(predictions) != len(y_true):
        logging.warning("Prediction and actual data length mismatch.")
        return

    mse = mean_squared_error(y_true, predictions)
    mae = mean_absolute_error(y_true,predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, predictions)
    
    PREDICTION_MSE.set(mse)
    PREDICTION_RMSE.set(rmse)
    PREDICTION_R2.set(r2)
    PREDICITON_MAE.set(mae)

    logging.info(f"Batch Metrics -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")


def consume(model, s3):
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    buffer = []
    batch_number = 0

    for message in consumer:
        MESSAGES_PROCESSED_TOTAL.inc()
        buffer.append(message.value)

        if len(buffer) >= BATCH_SIZE:
            BATCHES_PROCESSED_TOTAL.inc()
            with BATCH_PROCESSING_TIME_SECONDS.time():
                try:
                    df = pd.DataFrame(buffer)
                    # df = preprocess_data(df)
                    if df.empty:
                        logging.warning("DataFrame is empty after preprocessing. Skipping batch.")
                        buffer.clear()
                        continue

                    features = ['year','month','day','hour','DE_wind_capacity', 'DE_wind_speed', 'DE_temperature', 'DE_air_density']
                    # features = ['DE_wind_capacity', 'DE_wind_speed', 'DE_temperature', 'DE_air_density']
                    X = df[features]
                    y_true = df['DE_wind_generation_actual']

                    predictions = model.predict(X)
                    log_and_update_metrics(predictions, y_true)

                    save_batch_to_minio(df, s3, batch_number)
                    batch_number += 1

                    df['predicitons'] = predictions
                    print(df)

                    CSV_KEY_PREFIX_PREDICTIONS = "predictions/"
                    timestamp_pred = datetime.now().strftime("%Y-%m-%d_%H%M")
                    local_file_predictions = f"prediction_batch_{timestamp_pred}.csv"
                    minio_key = f"{CSV_KEY_PREFIX_PREDICTIONS}prediction_batch_{timestamp_pred}_final_{batch_number}.csv"

                    df.to_csv(local_file_predictions, index=False)
                    try:
                        s3.upload_file(local_file_predictions, BUCKET_NAME, minio_key)
                        logging.info(f"Batch with predictions saved to MinIO: {minio_key}")
                    finally:
                        os.remove(local_file_predictions)

                except Exception as e:
                    PREDICTION_ERRORS_TOTAL.inc()
                    logging.error(f"Error processing batch: {e}", exc_info=True)

                finally:
                    buffer.clear()

if __name__ == "__main__":
    logging.info("Starting Prometheus service for Consumer...")
    start_http_server(PROMETHEUS_PORT)

    model = load_model()
    if model:
        s3_client = get_minio_client()
        consume(model, s3_client)
    else:
        logging.error("Model failed to load. Exiting.")