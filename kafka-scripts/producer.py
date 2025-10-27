import pandas as pd
from time import sleep
from kafka import KafkaProducer
import json
import os
import boto3
from io import BytesIO
from prometheus_client import start_http_server, Counter

MESSAGES_SENT_TOTAL = Counter('kafka_messages_sent_total', 'Total number of Kafka messages sent')
PRODUCER_ERRORS_TOTAL = Counter('kafka_producer_errors_total', 'Total errors encountered by the producer')
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", 8001)) # DIFFERENT port than the consumer
SEND_INTERVAL_SECONDS = float(os.getenv("SEND_INTERVAL_SECONDS", 0.15))


def load_minio() -> pd.DataFrame:

    minio_endpoint = "http://localhost:9000"  # de schimbat la nev
    access_key = "IFSYL3fy8jSmMfK2To6G"
    secret_key = "WzhnEeyrtGmjtxE5QtSzK7ka0fZhgAgnqoykxm5g"
    bucket_name = "datasets-mlops"
    csv_key = "/kafka/kafka_simulation_dataset.csv" 

    s3 = boto3.client('s3',
                    endpoint_url=minio_endpoint,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key)

    obj = s3.get_object(Bucket=bucket_name, Key=csv_key)
    df = pd.read_csv(BytesIO(obj['Body'].read()))

    return df
    # print(df.head())


def produce(df: pd.DataFrame):
    producer = KafkaProducer(bootstrap_servers='localhost:9092',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))


    for index, row in df.iterrows():
        try:
            data = row.to_dict()
            producer.send('wind-data-testing-12', value=data)
            producer.flush()
            MESSAGES_SENT_TOTAL.inc()
            print(f"Sent: {data}")
            print("Current sent counter:", MESSAGES_SENT_TOTAL._value.get())  # debug
            sleep(SEND_INTERVAL_SECONDS)
        except Exception as e:
            PRODUCER_ERRORS_TOTAL.inc()
            print(f"Error sending message: {e}") # Simulate real-time


if __name__ == "__main__":
    print("STARTING PROMETHEUS SERVICE FOR PRODUCER")
    start_http_server(PROMETHEUS_PORT)
    
    df = load_minio()
    produce(df)
    input("Press Enter to exit.")

