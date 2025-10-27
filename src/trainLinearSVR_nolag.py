import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.model_selection import TimeSeriesSplit

import os
import boto3
from io import BytesIO
import datetime

class ModelWithScaler(mlflow.pyfunc.PythonModel):
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, context, model_input):
        # Scale the input using the scaler
        input_scaled = self.scaler.transform(model_input)
        return self.model.predict(input_scaled)

minio_endpoint = "http://minio:9000"  # minio:9000 - in the pipeline careful
access_key = "IFSYL3fy8jSmMfK2To6G"
secret_key = "WzhnEeyrtGmjtxE5QtSzK7ka0fZhgAgnqoykxm5g"
bucket_name = "datasets-mlops"

s3 = boto3.client('s3',
                  endpoint_url=minio_endpoint,
                  aws_access_key_id=access_key,
                  aws_secret_access_key=secret_key)

obj = s3.get_object(Bucket=bucket_name, Key="historical/historical_dataset.csv" )
df = pd.read_csv(BytesIO(obj['Body'].read()))

df = df[df['year']<2020]
print(df)

# logic for when we have and when we dont have batches.
try:
    obj_latest = s3.get_object(Bucket=bucket_name, Key="historical/latest_batches.csv")
    df_latest_batches = pd.read_csv(BytesIO(obj_latest['Body'].read()))

    split_idx = int(len(df_latest_batches) * 0.7)
    df_latest_train = df_latest_batches.iloc[:split_idx]
    df_latest_test = df_latest_batches[split_idx:]

    train_df = pd.concat([df, df_latest_train], ignore_index=True)
    test_df = df_latest_test.reset_index(drop=True)

    print("Using historical + batch data.")
    print(df_latest_batches)

except s3.exceptions.NoSuchKey:
    print("No batch file found, using historical data (2015-2018) and initial testing (2019)")

    train_df = df[df['year']<2019]
    test_df = df[df['year']>=2019]


print("train_df")
print(train_df)

print("test_df")
print(test_df)

features = ['year','month','day','hour','DE_wind_capacity', 'DE_wind_speed', 'DE_temperature', 'DE_air_density']
target = 'DE_wind_generation_actual'

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define parameter grid
tscv = TimeSeriesSplit(n_splits=3)

param_grid = {
    'C': [0.01,0.1, 1, 10,100],  # Regularization strength
    'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
    'epsilon': [0.0001, 0.001, 0.01, 0.1, 1],  # Tolerance for error
    'max_iter': [1000,2000,3500,5000]  # Increase iterations for convergence
}

grid_search = GridSearchCV(LinearSVR(), param_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

grid_search.fit(X_train_scaled, y_train)
best_svr = grid_search.best_estimator_

print("Best parameters:", grid_search.best_params_)

y_pred = best_svr.predict(X_test_scaled)
test_df['predicted_wind_generation'] = y_pred

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Root Mean Squared Error

# Compute R² Score
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Error': y_test - y_pred
})

date = datetime.datetime.now()
ver = date.strftime("%Y-%m-%d-%H:%M")
print(ver)


features_ = ['DE_wind_capacity', 'DE_wind_speed', 'DE_temperature', 'DE_air_density']
stats = train_df[features_].describe(percentiles=[.25,.5,.75]).transpose()
stats.to_json("training_feature_stats.json", indent=4)

mlflow.set_tracking_uri('http://mlflow:5000') #remember to change to mlflow:5000
mlflow.set_experiment("LinearSVR Wind Generation Prediction-1 __ no lag features")
print("setup complete")
with mlflow.start_run(run_name=f"linearsvr_model_nolag-{ver}"):
    print("started run")

    mlflow.sklearn.pyfunc.log_model("linearSVR_withScaler_batches_nolag",python_model=ModelWithScaler(model = best_svr, scaler= scaler))
    print("logged model")

    mlflow.log_params(best_svr.get_params())
    print("logged params")

    mlflow.log_metric("MSE",mse)
    print("logged MSE")
    mlflow.log_metric("RMSE",rmse)
    print("logged RMSE")
    mlflow.log_metric("R2",r2)
    print("logged R2")
    mlflow.log_artifact("training_feature_stats.json")

    model_info = mlflow.register_model(
        "runs:/{}/linearSVR_withScaler_batches_nolag".format(mlflow.active_run().info.run_id),
        name="model_linearSVR_scaler_batches_nolag"
    )
    print("model logged with mlflow")

os.remove("training_feature_stats.json")