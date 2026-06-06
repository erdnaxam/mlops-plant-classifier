"""
DAG Airflow — Entraînement du modèle + déploiement automatique.
Triggers : hebdomadaire ou sur signal de drift détecté.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "mlops-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

def task_check_data(**context):
    """Vérifie que des données sont disponibles dans le feature store."""
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.features.feature_store import load_latest_features
    df = load_latest_features()
    n = len(df)
    context["ti"].xcom_push(key="n_samples", value=n)
    return "train_model" if n >= 100 else "skip_training"

def task_train(**context):
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.model.train import train
    run_id = train(epochs=5)
    context["ti"].xcom_push(key="mlflow_run_id", value=run_id)
    return run_id

def task_evaluate(**context):
    """Log les métriques de validation et vérifie le seuil de qualité."""
    import mlflow, os
    run_id = context["ti"].xcom_pull(task_ids="train_model")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    acc = run.data.metrics.get("accuracy", 0)
    if acc < 0.85:
        raise ValueError(f"Accuracy trop faible : {acc:.3f} < 0.85")
    return acc

def task_promote_model(**context):
    """Passe le modèle en stage 'Production' dans MLflow."""
    import mlflow, os
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("plant-classifier", stages=["None"])
    if versions:
        client.transition_model_version_stage(
            name="plant-classifier",
            version=versions[0].version,
            stage="Production",
        )

with DAG(
    dag_id="plant_model_training",
    default_args=default_args,
    description="Entraîne, évalue et promeut le modèle de classification",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["training", "ml"],
) as dag:

    check = BranchPythonOperator(
        task_id="check_data_availability",
        python_callable=task_check_data,
    )

    skip = EmptyOperator(task_id="skip_training")

    train = PythonOperator(
        task_id="train_model",
        python_callable=task_train,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate,
    )

    promote = PythonOperator(
        task_id="promote_to_production",
        python_callable=task_promote_model,
    )

    check >> [train, skip]
    train >> evaluate >> promote
