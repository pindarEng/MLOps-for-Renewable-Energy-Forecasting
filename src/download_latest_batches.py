import json
import pandas as pd
import os
import boto3
from datetime import datetime, timedelta
from io import StringIO # 


BUCKET_NAME = "datasets-mlops"
BATCH_PREFIX = "batches/"
DRIFT_CHECK_WINDOW_DAYS = 7


def get_minio_client():
    return boto3.client(
        's3',
        endpoint_url="http://minio:9000",
        aws_access_key_id="IFSYL3fy8jSmMfK2To6G",
        aws_secret_access_key="WzhnEeyrtGmjtxE5QtSzK7ka0fZhgAgnqoykxm5g")



def recent_batches(s3_client, bucket, prefix, days_back):
    """Lists, downloads, and aggregates recent batches from MinIO."""
    cutoff_date = datetime.now() - timedelta(days=days_back)
    print(f"Fetching batches newer than {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")

    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    recent_batch_keys = []
    for page in pages:
        if "Contents" in page:
            for obj in page['Contents']:
                # assumes filename contains timestamp parsable info
                if obj['LastModified'].replace(tzinfo=None) >= cutoff_date:
                     # checks for .csv files
                    if obj['Key'].endswith('.csv') and prefix in obj['Key']:
                         recent_batch_keys.append(obj['Key'])

    if not recent_batch_keys:
        print("No recent batches found within the specified window.")
        return [] # empty list
    
    print(f"Found {len(recent_batch_keys)} recent batch files.")
    return recent_batch_keys

def download_aggregate_recent_batches(s3_client, bucket, keys):
    
    all_batches_df = []
    for key in keys:
        try:
            print(f"Downloading and reading: {key}")
            csv_obj = s3_client.get_object(Bucket=bucket, Key=key)
            body = csv_obj['Body']
            csv_string = body.read().decode('utf-8')
            # Use StringIO to read the string as if it were a file
            batch_df = pd.read_csv(StringIO(csv_string))
            all_batches_df.append(batch_df)
        except Exception as e:
            print(f"Warning: Could not read batch file {key}. Error: {e}")

    if not all_batches_df:
         print("Could not read any of the recent batch files.")
         return pd.DataFrame()

    aggregated_df = pd.concat(all_batches_df, ignore_index=True)
    print(f"Aggregated data shape: {aggregated_df.shape}")
    return aggregated_df

def move_batches_to_deprecated(s3_client, bucket, keys, prefix="deprecated/"):
    for key in keys:
        filename = os.path.basename(key)
        new_key = f"{prefix}{filename}"
        print(f"Moving {key} -> {new_key}")

        try:
            #copy object to new location // move to deprecated
            s3_client.copy_object(
                Bucket=bucket,
                CopySource={'Bucket': bucket, 'Key': key},
                Key=new_key
            )

            #delete original object // delete batches
            s3_client.delete_object(Bucket=bucket, Key=key)
        except Exception as e:
            print(f"Error moving {key} to deprecated: {e}")



if __name__ == "__main__":
    s3 = get_minio_client()

    recent_keys = recent_batches(s3, BUCKET_NAME, BATCH_PREFIX, DRIFT_CHECK_WINDOW_DAYS)

    if not recent_keys:
        print("No recent data to analyze. Exiting.")
        exit()

    df = download_aggregate_recent_batches(s3, BUCKET_NAME, recent_keys)

    print(df)

    features_ = ['DE_wind_capacity', 'DE_wind_speed', 'DE_temperature', 'DE_air_density']
    
    
    stats = df[features_].describe(percentiles=[.25,.5,.75]).transpose()
    

    seasonal_stats = {}

    for (year, month), group in df.groupby(["year", "month"]):
        stats = group[features_].describe().to_dict() 
        
        year_str = str(year)
        
        if year_str not in seasonal_stats:
            seasonal_stats[year_str] = {}
        
        month_str = str(int(month)).zfill(2)

        seasonal_stats[year_str][month_str] = stats

    js = json.dumps(seasonal_stats, indent=4)
    # with open("jsons/seasonal_stats.json", "w") as f:
    #     f.write(js)


    # stats.to_json("drift/latest_batches_feature_stats.json", indent=4)
    s3.put_object(Bucket="datasets-mlops", Key="drift/latest_batches_feature_stats.json", Body=js)

    new_key = f"historical/latest_batches.csv" 
    s3.put_object(Bucket=BUCKET_NAME, Key=new_key, Body=df.to_csv(index=False))

    metadata = {
    "created_at": datetime.now().isoformat(),
    "files_used": recent_keys,
    "row_count": len(df),
    "columns": list(df.columns)
    }

    s3.put_object(Bucket=BUCKET_NAME, Key="historical/consolidated_metadata.json", Body=json.dumps(metadata, indent=4))


    #save a local file also. mostly for the serve.py havent got the logic for a more sophisticated way yet ...
    df.to_csv(f"src/temp/latest_batches.csv", index=False)

    move_batches_to_deprecated(s3, BUCKET_NAME, recent_keys)
