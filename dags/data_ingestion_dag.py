"""
dags/data_ingestion_dag.py
DAG Airflow : télécharge les images depuis GitHub et les stocke dans MinIO.
Se déclenche manuellement ou tous les lundis.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import sys
import os

sys.path.insert(0, "/opt/airflow/src")

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="data_ingestion",
    default_args=default_args,
    description="Télécharge les images dandelion/grass et les stocke dans MinIO",
    schedule_interval="@weekly",
    start_date=days_ago(1),
    catchup=False,
    tags=["data", "ingestion"],
) as dag:

    def download_images():
        from data.download import run
        records = run()
        print(f"✅ {len(records)} images téléchargées et stockées dans MinIO.")
        return len(records)

    def save_to_feature_store(**context):
        from features.feature_store import save_metadata
        # Récupère les records via XCom
        ti = context["ti"]
        n = ti.xcom_pull(task_ids="download_images")
        print(f"Feature store mis à jour avec {n} entrées.")

    def validate_data():
        from minio import Minio
        client = Minio(
            os.getenv("MINIO_ENDPOINT", "minio:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=False,
        )
        bucket = os.getenv("MINIO_BUCKET", "plant-classifier")
        objects = list(client.list_objects(bucket, prefix="images/", recursive=True))
        print(f"✅ Validation : {len(objects)} images présentes dans MinIO.")
        assert len(objects) > 0, "Aucune image trouvée dans MinIO !"
        return len(objects)

    task_download = PythonOperator(
        task_id="download_images",
        python_callable=download_images,
    )

    task_feature_store = PythonOperator(
        task_id="save_to_feature_store",
        python_callable=save_to_feature_store,
        provide_context=True,
    )

    task_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    task_download >> task_feature_store >> task_validate
