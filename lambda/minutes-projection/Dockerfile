# Use specific platform architecture for Lambda
FROM public.ecr.aws/lambda/python:3.11

# Copy requirements and all Python modules
COPY requirements.txt ${LAMBDA_TASK_ROOT}
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY config.py ${LAMBDA_TASK_ROOT}
COPY s3_utils.py ${LAMBDA_TASK_ROOT}
COPY actuals_updater.py ${LAMBDA_TASK_ROOT}
COPY projection_models.py ${LAMBDA_TASK_ROOT}
COPY injury_system.py ${LAMBDA_TASK_ROOT}
COPY process_model.py ${LAMBDA_TASK_ROOT}
COPY lineup_optimizer.py ${LAMBDA_TASK_ROOT}

# Install Python dependencies
RUN pip install -r requirements.txt

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"]
