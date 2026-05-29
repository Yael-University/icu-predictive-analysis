# Cheapest research deployment: Lambda container + Function URL

This bundle is the **low-traffic / cheapest practical** version of your mortality API:

UI or Postman -> **Lambda Function URL** -> Lambda container -> scikit-learn fusion bundle

## Why this architecture

Use this when traffic is low and you want the monthly floor as close to zero as possible.

- **No ALB**
- **No ECS service**
- **No EC2**
- **No NAT gateway**
- **No VPC**
- **No always-on compute**

You pay mainly for Lambda invocations/compute plus tiny ECR image storage.

## What is in this bundle

- `train/export_fusion_bundle.py` — trains and exports a 5-model bundle
- `app/handler.py` — Lambda HTTP API handler
- `common/features.py` — feature engineering shared logic
- `Dockerfile` — Lambda container image
- `deploy/deploy_lambda.sh` — one deploy/update script
- `sample_request.json` — example request payload
- `tests/local_invoke.py` — quick local smoke test

## Models in the exported bundle

This bundle trains these 5 models:

1. calibrated `LinearSVC`
2. decision tree
3. random forest
4. gradient boosting
5. logistic regression

`GradientBoostingClassifier` is being used as the 4th slot because your fourth uploaded model file was not provided.

## Request format

`POST /predict`

```json
{
  "patient": {
    "demographics": {
      "age": 67,
      "gender": "M",
      "race": "WHITE",
      "language": "ENGLISH"
    },
    "admission": {
      "admission_type": "EMERGENCY",
      "admission_location": "EMERGENCY ROOM",
      "insurance": "Medicare",
      "marital_status": "MARRIED",
      "has_ed_visit": true,
      "ed_los_hours": 4.5
    },
    "diagnoses": ["I10", "E11.9", "J18.9"],
    "procedures": ["5A1955Z", "0BH17EZ"],
    "medications": ["aspirin", "metformin", "ceftriaxone"]
  }
}
```

## Response format

```json
{
  "predicted_class": 1,
  "death_probability": 0.271234,
  "death_percentage": 27.12,
  "vote_fraction": 0.6,
  "votes": {
    "svm_calibrated": 1,
    "decision_tree": 0,
    "random_forest": 1,
    "gradient_boosting": 1,
    "logistic_regression": 0
  },
  "model_probabilities": {
    "svm_calibrated": 0.31,
    "decision_tree": 0.18,
    "random_forest": 0.29,
    "gradient_boosting": 0.36,
    "logistic_regression": 0.22
  }
}
```

## Step 0 — prerequisites

Install these on your machine:

- Python 3.11+
- Docker
- AWS CLI v2
- An AWS account with permission to use IAM, Lambda, and ECR

Configure AWS CLI first:

```bash
aws configure
```

## Step 1 — train and export the model bundle

From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then export the model bundle from your MIMIC-style CSV directory:

```bash
python train/export_fusion_bundle.py \
  --data-dir /path/to/your/mimic-folder \
  --output model_artifacts/fusion_bundle.joblib
```

That folder should contain at least:

- `admissions.csv`
- `patients.csv`
- `diagnoses_icd.csv`
- `procedures_icd.csv`

Optional:

- `prescriptions.csv`

If `prescriptions.csv` is missing, the training script automatically drops medication count features.

## Step 2 — smoke test locally before AWS

```bash
python tests/local_invoke.py
```

If you want to use your own key locally:

```bash
API_KEY=my-local-key python tests/local_invoke.py
```

## Step 3 — deploy to AWS

Set a few variables first.

macOS/Linux:

```bash
export APP_NAME=mortality-api
export AWS_REGION=us-east-1
export API_KEY='replace-with-a-long-random-secret'
export ALLOWED_ORIGIN='*'
```

PowerShell:

```powershell
$env:APP_NAME='mortality-api'
$env:AWS_REGION='us-east-1'
$env:API_KEY='replace-with-a-long-random-secret'
$env:ALLOWED_ORIGIN='*'
```

Then run:

```bash
bash deploy/deploy_lambda.sh
```

What the script does:

1. creates or reuses an ECR repo
2. builds the container image
3. pushes the image to ECR
4. creates or reuses an IAM execution role
5. creates or updates the Lambda function
6. caps reserved concurrency at 2
7. creates a Function URL
8. grants the public Function URL invoke permissions required for `AuthType=NONE`

## Step 4 — test the live API

The deploy script prints the Function URL.

Health check:

```bash
curl "https://YOUR_ID.lambda-url.us-east-1.on.aws/health"
```

Prediction:

```bash
curl -X POST "https://YOUR_ID.lambda-url.us-east-1.on.aws/predict" \
  -H "content-type: application/json" \
  -H "x-api-key: replace-with-a-long-random-secret" \
  -d @sample_request.json
```

## Step 5 — connect the UI

Your friend’s UI should call:

- `GET /health` for readiness checks
- `POST /predict` for inference

Headers:

- `content-type: application/json`
- `x-api-key: <your secret>`

## Recommended settings

These are set by default in `deploy/deploy_lambda.sh`:

- architecture: `x86_64`
- memory: `2048 MB`
- timeout: `30 sec`
- ephemeral storage: `1024 MB`
- reserved concurrency: `2`

These are intentionally conservative so the first deploy is more likely to work.

## Why I chose x86_64 instead of ARM first

ARM can be cheaper, but `x86_64` is the least painful first deployment path for scikit-learn / pandas wheels. Once the API is stable, you can try ARM later.

## How to keep cost low

- keep it **outside a VPC**
- do **not** add a NAT gateway
- do **not** use provisioned concurrency
- keep reserved concurrency low like `2`
- keep the function URL instead of API Gateway unless you need more gateway features

## Security notes

This setup uses a Lambda Function URL with `AuthType=NONE`, which means the HTTPS endpoint is publicly reachable. The app itself protects `/predict` with a shared secret in the `x-api-key` header.

That is good enough for a small research tool, but you should still:

- use a long random API key
- rotate it if it leaks
- set `ALLOWED_ORIGIN` to your real frontend origin instead of `*`
- never put PHI in logs
- never commit the API key to GitHub

If you later want stronger auth, the easiest upgrade path is API Gateway + JWT or IAM.

## Updating the model later

When you retrain:

1. regenerate `model_artifacts/fusion_bundle.joblib`
2. rerun `bash deploy/deploy_lambda.sh`

That rebuilds the image and updates the Lambda function.

## Troubleshooting

### 1. Function deploys but `/predict` returns 500

Check logs:

```bash
aws logs tail /aws/lambda/mortality-api --follow
```

### 2. You get 401 Unauthorized

Your `x-api-key` header does not match the `API_KEY` environment variable on the Lambda function.

### 3. CORS errors from the browser

Set:

```bash
export ALLOWED_ORIGIN='https://your-frontend-domain.com'
```

Then redeploy.

### 4. Model file not found

Make sure this file exists before `docker build`:

```text
model_artifacts/fusion_bundle.joblib
```

### 5. Cold starts feel slow

That is normal for a low-cost Lambda container setup. For a research tool with low traffic, it is usually acceptable.

## Optional next improvement

Once this is running, the next best step is to add:

- a tiny frontend example
- request schema validation
- a `/version` endpoint
- model metadata in the UI
- GitHub Actions for auto-deploy on push
