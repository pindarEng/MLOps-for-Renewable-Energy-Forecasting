import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import boto3

minio_endpoint = "http://minio:9000"  # Use minio:9000 in container, or localhost:9000 if local test
access_key = "IFSYL3fy8jSmMfK2To6G"
secret_key = "WzhnEeyrtGmjtxE5QtSzK7ka0fZhgAgnqoykxm5g"
bucket_name = "datasets-mlops"

s3 = boto3.client('s3',
                  endpoint_url=minio_endpoint,
                  aws_access_key_id=access_key,
                  aws_secret_access_key=secret_key)


print("STARTING THE PREPROCESSING OF THE DATA")
#WE READ THE DATASET AFTER THE INITIAL.PY
df = pd.read_csv("dataset//processed//dataset.csv")
# print(df)

#AFTER WE DID EDA ON THE INITIAL DATASET WE FOUND OUT THAT WE NEED TO DROP MORE FEATURES
df = df.drop(columns=['DE_wind_offshore_capacity', "DE_wind_onshore_capacity","DE_wind_onshore_profile","DE_wind_offshore_profile","DE_wind_profile"])

#WE DEAL WITH THE NULL VALUES FOR WIND_CAPACITY AND WIND_GENERATION_ACTUAL (TARGET)
df["DE_wind_capacity"] = df["DE_wind_capacity"].ffill()
df["DE_wind_generation_actual"] = df['DE_wind_generation_actual'].fillna(df["DE_wind_generation_actual"].mean())

print(df.info())

#WE FORMAT THE DATASET TO CONTAIN YEAR,MONTH,DAY,HOUR RATHER THAN A UTC-TIMESTAMP FORMAT
df["utc_timestamp"] = pd.to_datetime(df['utc_timestamp'])

df['year'] = df['utc_timestamp'].dt.year

df['month'] = df['utc_timestamp'].dt.month

df['day'] = df['utc_timestamp'].dt.day

df['hour'] = df['utc_timestamp'].dt.hour

df.drop(columns=["utc_timestamp"],inplace=True)

#WE ARRANGE THE FINAL DATASET FOR BETTER READABILITY
df = df.loc[: , ["year","month","day","hour","DE_wind_generation_actual","DE_wind_capacity","DE_wind_speed","DE_temperature","DE_air_density"]]

df_kafka = df[df['year']>2019]

df_kafka_simulation = df_kafka[df_kafka['month'] <= 6]

print("kafka simulation")
print(df_kafka_simulation)
print()

df_kafka_testing = df_kafka[df_kafka['month'] > 6]
print("kafka testing")
print(df_kafka_testing)
df = df[df['year']<2020]
print()
print(df)
# print(df)
# print(df_kafka)
#WE SAVE THE DATASET TO A FINAL VERSION
df.to_csv("dataset//processed//final_dataset.csv",index=False)

print("saving historical (2015-2018) dataset")
s3.put_object(Bucket=bucket_name, Key="historical/historical_dataset.csv", Body=df.to_csv(index=False))
print("saving kafka simulation dataset")
s3.put_object(Bucket=bucket_name, Key="kafka/kafka_simulation_dataset.csv", Body=df_kafka_simulation.to_csv(index=False))
print("saving testing dataset")
s3.put_object(Bucket=bucket_name, Key="fastapi-test/kafka_testing_dataset.csv", Body=df_kafka_testing.to_csv(index=False))


#trb salvat 2015-2019 in historical/historical_dataset.csv
#si 2020 in kafka_simulation_dataset.csv

