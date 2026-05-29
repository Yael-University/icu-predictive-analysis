#!/usr/bin/env bash
set -euo pipefail

# Load .env from the project root if it exists
if [ -f ".env" ]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

APP_NAME="${APP_NAME:-mortality-api}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ARCHITECTURE="${ARCHITECTURE:-x86_64}"
MEMORY_SIZE="${MEMORY_SIZE:-2048}"
TIMEOUT="${TIMEOUT:-120}"
EPHEMERAL_STORAGE_MB="${EPHEMERAL_STORAGE_MB:-1024}"
ALLOWED_ORIGIN="${ALLOWED_ORIGIN:-*}"
API_KEY="${API_KEY:-change-me-now}"
GEMINI_API_KEY="${GEMINI_API_KEY:-}"
ROLE_NAME="${ROLE_NAME:-${APP_NAME}-lambda-role}"
REPOSITORY_NAME="${REPOSITORY_NAME:-${APP_NAME}}"
FUNCTION_NAME="${FUNCTION_NAME:-${APP_NAME}}"
RESERVED_CONCURRENCY="${RESERVED_CONCURRENCY:-2}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REPO_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPOSITORY_NAME}"
IMAGE_URI="${REPO_URI}:latest"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

echo "==> Ensuring ECR repository exists"
aws ecr describe-repositories --repository-names "${REPOSITORY_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "${REPOSITORY_NAME}" --region "${AWS_REGION}" >/dev/null

echo "==> Logging in to ECR"
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "==> Building container image"
docker build --platform "linux/amd64" --provenance=false -t "${FUNCTION_NAME}:latest" .
docker tag "${FUNCTION_NAME}:latest" "${IMAGE_URI}"

echo "==> Pushing image"
docker push "${IMAGE_URI}"

echo "==> Ensuring IAM role exists"
if ! aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document file://deploy/lambda-trust-policy.json >/dev/null
fi

aws iam attach-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole >/dev/null

echo "==> Waiting a moment for IAM role propagation"
sleep 10

echo "==> Creating or updating Lambda function"
if aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  aws lambda update-function-code \
    --function-name "${FUNCTION_NAME}" \
    --image-uri "${IMAGE_URI}" \
    --architectures "${ARCHITECTURE}" \
    --region "${AWS_REGION}" >/dev/null

  aws lambda wait function-updated-v2 --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}"

  aws lambda update-function-configuration \
    --function-name "${FUNCTION_NAME}" \
    --memory-size "${MEMORY_SIZE}" \
    --timeout "${TIMEOUT}" \
    --ephemeral-storage "{\"Size\": ${EPHEMERAL_STORAGE_MB}}" \
    --environment "Variables={API_KEY=${API_KEY},ALLOWED_ORIGIN=${ALLOWED_ORIGIN},MODEL_PATH=/var/task/model_artifacts/fusion_bundle.joblib,GEMINI_API_KEY=${GEMINI_API_KEY}}" \
    --region "${AWS_REGION}" >/dev/null
else
  aws lambda create-function \
    --function-name "${FUNCTION_NAME}" \
    --package-type Image \
    --code ImageUri="${IMAGE_URI}" \
    --role "${ROLE_ARN}" \
    --memory-size "${MEMORY_SIZE}" \
    --timeout "${TIMEOUT}" \
    --architectures "${ARCHITECTURE}" \
    --ephemeral-storage "{\"Size\": ${EPHEMERAL_STORAGE_MB}}" \
    --environment "Variables={API_KEY=${API_KEY},ALLOWED_ORIGIN=${ALLOWED_ORIGIN},MODEL_PATH=/var/task/model_artifacts/fusion_bundle.joblib,GEMINI_API_KEY=${GEMINI_API_KEY}}" \
    --region "${AWS_REGION}" >/dev/null
fi

aws lambda wait function-active-v2 --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}"

echo "==> Capping concurrency to protect against runaway cost"
aws lambda put-function-concurrency \
  --function-name "${FUNCTION_NAME}" \
  --reserved-concurrent-executions "${RESERVED_CONCURRENCY}" \
  --region "${AWS_REGION}" >/dev/null 2>&1 || \
  echo "    WARNING: Could not set reserved concurrency (account unreserved pool too small); function will use unreserved concurrency."

echo "==> Creating or updating Function URL"
if aws lambda get-function-url-config --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  aws lambda update-function-url-config \
    --function-name "${FUNCTION_NAME}" \
    --auth-type NONE \
    --cors file://deploy/function-url-cors.json \
    --region "${AWS_REGION}" >/dev/null
else
  aws lambda create-function-url-config \
    --function-name "${FUNCTION_NAME}" \
    --auth-type NONE \
    --cors file://deploy/function-url-cors.json \
    --region "${AWS_REGION}" >/dev/null
fi

set +e
aws lambda add-permission \
  --function-name "${FUNCTION_NAME}" \
  --statement-id FunctionURLAllowPublicInvokeURL \
  --action lambda:InvokeFunctionUrl \
  --principal "*" \
  --function-url-auth-type NONE \
  --region "${AWS_REGION}" >/dev/null 2>&1

aws lambda add-permission \
  --function-name "${FUNCTION_NAME}" \
  --statement-id FunctionURLAllowPublicInvokeFunction \
  --action lambda:InvokeFunction \
  --principal "*" \
  --invoked-via-function-url \
  --region "${AWS_REGION}" >/dev/null 2>&1
set -e

FUNCTION_URL="$(aws lambda get-function-url-config --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}" --query FunctionUrl --output text)"

echo

echo "Deploy complete."
echo "Function URL: ${FUNCTION_URL}"
echo "Remember your x-api-key: ${API_KEY}"
