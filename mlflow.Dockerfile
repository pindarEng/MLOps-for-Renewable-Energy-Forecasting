FROM ghcr.io/mlflow/mlflow

RUN apt update && \ 
         apt install -y gcc libpq-dev && \
         pip install psycopg2 && \
         pip install boto3