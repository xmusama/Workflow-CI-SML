# Workflow-CI

Repository for Kriteria 3 (MLflow Project + CI retraining).

## Struktur
- MLProject/MLProject
- MLProject/conda.yaml
- MLProject/modelling.py
- MLProject/dataset_preprocessing/processed.csv
- MLProject/DockerHub.txt
- .github/workflows/ci.yml

## Menjalankan lokal
```bash
pip install mlflow pandas numpy scikit-learn joblib matplotlib
mlflow run MLProject --env-manager=local
```

## Secrets
Set these repository secrets:
- DOCKERHUB_USERNAME
- DOCKERHUB_TOKEN
