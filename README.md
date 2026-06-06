# 🌿 Plant Image Classifier — MLOps Bloc 4

Binary image classification: **Dandelion vs Grass**

## Architecture

```
Data (GitHub URLs) → Airflow DAG → MinIO (S3) → FastAI Training → MLflow Registry → FastAPI → Gradio WebApp
                                                                          ↓
                                                              Evidently Monitoring → Retraining DAG
```

## Stack technique

| Composant        | Technologie              |
|------------------|--------------------------|
| Modèle ML        | FastAI (ResNet18)        |
| Orchestration    | Apache Airflow           |
| Model Registry   | MLflow + MinIO (S3)      |
| API Serving      | FastAPI                  |
| WebApp           | Gradio                   |
| Monitoring       | Evidently                |
| CI/CD            | GitHub Actions           |
| Conteneurisation | Docker Compose           |

## Lancement rapide (dev local)

```bash
# 1. Cloner le repo
git clone https://github.com/VOTRE_USERNAME/mlops-plant-classifier.git
cd mlops-plant-classifier

# 2. Copier et configurer les variables d'environnement
cp .env.example .env

# 3. Lancer tous les services
docker compose up -d

# 4. Accès aux interfaces
# Airflow    → http://localhost:8080  (admin/admin)
# MLflow     → http://localhost:5000
# MinIO      → http://localhost:9001  (minioadmin/minioadmin)
# API        → http://localhost:8000/docs
# Gradio App → http://localhost:7860
```

## Structure du projet

```
.
├── src/
│   ├── data/          # Téléchargement et preprocessing des images
│   ├── model/         # Entraînement FastAI + logging MLflow
│   └── features/      # Feature store (fichiers locaux)
├── api/               # FastAPI serving
├── dags/              # Airflow DAGs (ingestion + réentraînement)
├── monitoring/        # Evidently drift reports
├── retrain/           # Scripts de réentraînement automatisé
├── tests/             # Tests unitaires et d'intégration
├── notebooks/         # Exploration et expérimentations
├── .github/workflows/ # CI/CD GitHub Actions
├── docker-compose.yml
└── requirements.txt
```

## Résultats du modèle

| Métrique  | Valeur  |
|-----------|---------|
| Accuracy  | ~97%    |
| F1-score  | ~0.97   |
| Recall    | ~0.96   |

> Modèle : ResNet18 fine-tuné sur 400 images (200 dandelion + 200 grass)
