# Crypto Prediction Monorepo

Monorepo for data preprocessing, model training, API, UI & infra.

## Repository Structure

```text
/ (root)
├── data/                      # raw + processed datasets, DVC metadata (if used)
│   ├── kaggle-raw/
│   └── processed/
│
├── models/                    # training scripts, notebooks, checkpoints
│   ├── notebooks/
│   └── src/                   # reusable model‑building code
│
├── backend/                   # FastAPI service + business logic
│   ├── app/
│   ├── tests/
│   └── Dockerfile
│
├── frontend/                  # React dashboard
│   ├── src/
│   ├── public/
│   └── Dockerfile
│
├── infra/                     # Terraform / CloudFormation / k8s manifests
│   └── ...
│
├── .github/                   # workflows, ISSUE_TEMPLATE.md, PULL_REQUEST_TEMPLATE.md
│   └── workflows/
│
├── .gitignore
├── README.md
└── docker-compose.yml         # for local dev: backend + redis + db + frontend
```
