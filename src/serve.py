import mlflow.sklearn
import mlflow.pyfunc
import mlflow
import pandas as pd

from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import RedirectResponse

from pydantic import BaseModel
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from io import BytesIO
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import boto3
from typing import Dict, List
from enum import Enum

# minio_endpoint = "http://minio:9000"  # Use minio:9000 in container, or localhost:9000 if local test
# access_key = "IFSYL3fy8jSmMfK2To6G"
# secret_key = "WzhnEeyrtGmjtxE5QtSzK7ka0fZhgAgnqoykxm5g"
# bucket_name = "datasets-mlops"

# s3 = boto3.client('s3',
#                   endpoint_url=minio_endpoint,
#                   aws_access_key_id=access_key,
#                   aws_secret_access_key=secret_key)

# obj = s3.get_object(Bucket=bucket_name, Key="historical/latest_batches.csv")

print("aplicatie testing")
app = FastAPI(title="Wind Prediction", version="1.0.0")

class ModelNameChoice(str, Enum):
    # Available prediction models 
    linear_svr_lag = "linear_svr_lag"
    xgboost_lag = "xgboost_lag"
    random_forest_lag = "random_forest_lag"

    linear_svr_nolag = "linear_svr_nolag"
    xgboost_nolag = "xgboost_nolag"
    random_forest_nolag = "random_forest_nolag"

batches_csv_path = "/app/dataset/fastapi-test/kafka_testing_dataset.csv"

features_base = ['DE_wind_capacity', 'DE_wind_speed', 'DE_temperature', 'DE_air_density']
time_features = ['year', 'month', 'day', 'hour']
lag_features = ['lag_1', 'lag_2', 'lag_3']

TARGET_COLUMN = 'DE_wind_generation_actual'

# drift_path = "src/temp/drift_init.html"
# batches_csv_path = "src/temp/latest_batches.csv" - deprecated

# COPY src/serve_model_svr_lag /app/model/svr_lag
# COPY src/serve_model_xgb_lag /app/model/xgb_lag
# COPY src/serve_model_randomForest_lag /app/model/randomForest_lag

# COPY src/serve_model_svr_nolag /app/model/svr_nolag
# COPY src/serve_model_xgb_nolag /app/model/xgb_nolag
# COPY src/serve_model_randomForest_nolag /app/model/randomForest_nolag

MODEL_CONFIG = {
    ModelNameChoice.linear_svr_lag.value: {
        "path": "/app/model/svr_lag",          # Path inside the container
        "prediction_col": "predicted_svr_lag",
        "use_lags": True
    },
    ModelNameChoice.xgboost_lag.value: {
        "path": "/app/model/xgb_lag",          # Path inside the container
        "prediction_col": "predicted_xgb_lag",
        "use_lags": True
    },
    ModelNameChoice.random_forest_lag.value: {
        "path": "/app/model/randomForest_lag",          # Path inside the container
        "prediction_col": "predicted_randomForest_lag",
        "use_lags": True
    },

    ModelNameChoice.linear_svr_nolag.value: {
        "path": "/app/model/svr_nolag",          # Path inside the container
        "prediction_col": "predicted_svr_nolag",
        "use_lags": False
    },
    ModelNameChoice.xgboost_nolag.value: {
        "path": "/app/model/xgb_nolag",          # Path inside the container
        "prediction_col": "predicted_xgb_nolag",
        "use_lags": False
    },
    ModelNameChoice.random_forest_nolag.value: {
        "path": "/app/model/randomForest_nolag",          # Path inside the container
        "prediction_col": "predicted_randomForest_nolag",
        "use_lags": False
    }
}
AVAILABLE_MODELS = [model.value for model in ModelNameChoice]

loaded_models: Dict[str, mlflow.pyfunc.PyFuncModel] = {}



df_batches = pd.read_csv(batches_csv_path)

df = df_batches.copy()

# split_idx = int(len(df_batches) * 0.7)
# df = df_batches.iloc[split_idx:].reset_index(drop=True)


print(df)
# mlflow.set_tracking_uri("http://mlflow:5000")
# model_uri = "/app/model"

def add_lags_if_needed(df: pd.DataFrame, use_lags: bool) -> pd.DataFrame:
    if use_lags:
        df['lag_1'] = df['DE_wind_generation_actual'].shift(1)
        df['lag_2'] = df['DE_wind_generation_actual'].shift(2)
        df['lag_3'] = df['DE_wind_generation_actual'].shift(3)
        df.dropna(inplace=True)
    return df


print(df)
# mlflow.set_tracking_uri("http://mlflow:5000")
# model_uri = "/app/model"


for model_key, config in MODEL_CONFIG.items():
    print(f"Loading model: {model_key}")
    model = mlflow.pyfunc.load_model(config["path"])
    
    print("Adding lags if needed...")
    df = add_lags_if_needed(df, config["use_lags"])
    
    if config["use_lags"]:
        X_test = df[features_base + lag_features]
    else:
        X_test = df[time_features + features_base]
    
    print(f"Predicting with {model_key}...")
    df[config["prediction_col"]] = model.predict(X_test)

# y_test = test_df['DE_wind_generation_actual']

# mse = mean_squared_error(y_test, predictions)
# rmse = np.sqrt(mse)  # Root Mean Squared Error
# r2 = r2_score(y_test, predictions)

# print(f"Mean Squared Error (MSE): {mse:.4f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
# print(f"R² Score: {r2:.4f}")
# print(predictions)

start_date = df.iloc[0][['year','month','day']].values
end_date = df.iloc[-1][['year','month','day']].values

start_date_str = f"{int(start_date[0]):04d}-{int(start_date[1]):02d}-{int(start_date[2]):02d}"
end_date_str = f"{int(end_date[0]):04d}-{int(end_date[1]):02d}-{int(end_date[2]):02d}"

print(f'Linear SVR: Wind Generation Prediction {start_date_str} - {end_date_str}')


@app.get("/info")
async def get_info():
    if not df.empty:
        example_params = {
             "year": int(df['year'].iloc[0]),
             "month": int(df['month'].iloc[0]),
             "days_range": [int(df['day'].min()), int(df['day'].max())]
         }
    else:
        example_params = {"error": "No data loaded/processed"}

    return {
        "service_title": app.title,
        "version": app.version,
        "available_models": AVAILABLE_MODELS,
        "processed_data_range": f"{start_date_str} to {end_date_str}",
        "example_query_params": example_params,
        "plot_endpoint_structure": "/wind-generation-plot/{model_name}/{year}/{month}/{start_day}/{end_day}",
        "note": "Plot shows predictions for the selected model on the processed dataset."
    }

def generate_plot(model_name_str: str, year,month,range_day_start,range_day_stop):
    subset = df[(df['year'] == year) &
                    (df['month'] == month) &
                    (df['day'] >=range_day_start) & (df['day']<=range_day_stop)]
    
    prediction_col = MODEL_CONFIG[model_name_str]["prediction_col"]
    plot_title = f'{model_name_str.upper()}: Actual vs Predicted ({year}-{month:02d}-{range_day_start:02d} to {range_day_stop:02d})'
    y_true = subset[TARGET_COLUMN]
    y_pred = subset[prediction_col]

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    plt.figure(figsize=(10, 6))
    plt.plot(subset.index, subset[TARGET_COLUMN], label="Actual", color="blue", marker='.', markersize=4)
    plt.plot(subset.index, subset[prediction_col], label=f"Predicted ({model_name_str})", color="red", linestyle="--", marker='x', markersize=4)
    
    metrics_text = f"R²: {r2:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}"
    plt.text(0.95, 0.95, metrics_text, transform=plt.gca().transAxes, 
             horizontalalignment='right', verticalalignment='top', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))

    plt.xlabel('Index')
    plt.legend()
    plt.ylabel('Wind Generation')
    plt.xticks(rotation=45)
    plt.grid()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return img
@app.get("/wind-generation-plot/{year}/{month}/{range_day_start}/{range_day_stop}")
async def wind_generation_plot(model_name: ModelNameChoice, year: int, month: int, range_day_start: int, range_day_stop: int):
    img = generate_plot(model_name.value,year,month,range_day_start,range_day_stop)
    return StreamingResponse(img, media_type="image/png")


# @app.get("/drift-report")
# def serve_drift_report():
#     return FileResponse("src/temp/drift_init.html", media_type="text/html")

@app.get("/predictions/{model_name}")
async def get_predictions(model_name: ModelNameChoice):
    return df[[*features_base, TARGET_COLUMN, MODEL_CONFIG[model_name.value]["prediction_col"]]].to_dict(orient="records")
