"""
dags/retrain_dag.py
DAG Airflow : réentraîne le modèle et le déploie automatiquement.
Triggers : hebdomadaire + si accuracy < seuil (data drift ou déclin de performance).
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
import sys
import os

sys.path.insert(0, "/opt/airflow/src")
sys.path.insert(0, "/opt/airflow/retrain")

ACCURACY_THRESHOLD = 0.85  # Seuil de performance minimum

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="continuous_training",
    default_args=default_args,
    description="Réentraîne et déploie le modèle automatiquement",
    schedule_interval="@weekly",
    start_date=days_ago(1),
    catchup=False,
    tags=["training", "mlops", "ct"],
) as dag:

    def check_model_performance(**context):
        """Vérifie si le modèle actuel est encore performant."""
        import mlflow
        import os

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

        try:
            client = mlflow.tracking.MlflowClient()
            model_name = os.getenv("MODEL_NAME", "plant-classifier")
            versions = client.get_latest_versions(model_name, stages=["Production"])

            if not versions:
                print("Aucun modèle en production → réentraînement nécessaire.")
                return "train_model"

            run_id = versions[0].run_id
            run = client.get_run(run_id)
            current_accuracy = run.data.metrics.get("val_accuracy", 0)

            print(f"Accuracy actuelle: {current_accuracy:.4f} (seuil: {ACCURACY_THRESHOLD})")

            if current_accuracy < ACCURACY_THRESHOLD:
                print("⚠️ Performance insuffisante → réentraînement déclenché.")
                return "train_model"
            else:
                print("✅ Modèle performant → pas de réentraînement nécessaire.")
                return "skip_training"

        except Exception as e:
            print(f"Erreur lors de la vérification: {e} → réentraînement par précaution.")
            return "train_model"

    def train_new_model(**context):
        """Lance l'entraînement du modèle."""
        from model.train import run
        run_id = run()
        context["ti"].xcom_push(key="run_id", value=run_id)
        print(f"✅ Modèle entraîné — Run ID: {run_id}")

    def promote_to_production(**context):
        """Passe le nouveau modèle en Production dans MLflow."""
        import mlflow
        ti = context["ti"]
        run_id = ti.xcom_pull(task_ids="train_model", key="run_id")

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        client = mlflow.tracking.MlflowClient()
        model_name = os.getenv("MODEL_NAME", "plant-classifier")

        # Archive les anciennes versions
        old_versions = client.get_latest_versions(model_name, stages=["Production"])
        for v in old_versions:
            client.transition_model_version_stage(model_name, v.version, "Archived")

        # Passe la nouvelle version en Production
        latest = client.get_latest_versions(model_name, stages=["None"])
        if latest:
            client.transition_model_version_stage(model_name, latest[0].version, "Production")
            print(f"✅ Modèle v{latest[0].version} promu en Production.")

    def run_monitoring(**context):
        """Génère un rapport Evidently après déploiement."""
        from monitoring.monitor import generate_report
        generate_report()
        print("✅ Rapport de monitoring généré.")

    # ─── Tasks ──────────────────────────────────────────────────────
    check_performance = BranchPythonOperator(
        task_id="check_model_performance",
        python_callable=check_model_performance,
        provide_context=True,
    )

    skip_training = DummyOperator(task_id="skip_training")

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_new_model,
        provide_context=True,
    )

    promote = PythonOperator(
        task_id="promote_to_production",
        python_callable=promote_to_production,
        provide_context=True,
    )

    monitor = PythonOperator(
        task_id="run_monitoring",
        python_callable=run_monitoring,
        provide_context=True,
    )

    end = DummyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    check_performance >> [train, skip_training]
    train >> promote >> monitor >> end
    skip_training >> end
