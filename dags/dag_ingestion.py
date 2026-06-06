"""
DAG Airflow — Ingestion des images depuis GitHub vers MinIO.
Fréquence : quotidienne (ou déclenchée manuellement).
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mlops-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

def task_download_images(**context):
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.data.download import run_ingestion
    records = run_ingestion()
    context["ti"].xcom_push(key="n_images", value=len(records))
    return records

def task_save_features(**context):
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.features.feature_store import save_features
    records = context["ti"].xcom_pull(task_ids="download_images")
    path = save_features(records)
    return path

with DAG(
    dag_id="plant_data_ingestion",
    default_args=default_args,
    description="Télécharge et préprocess les images plantes vers MinIO",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "data"],
) as dag:

    download = PythonOperator(
        task_id="download_images",
        python_callable=task_download_images,
    )

    save = PythonOperator(
        task_id="save_to_feature_store",
        python_callable=task_save_features,
    )

    download >> save
