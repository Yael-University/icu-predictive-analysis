FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY app ./app
COPY common ./common
COPY model_artifacts ./model_artifacts

CMD ["app.handler.lambda_handler"]
