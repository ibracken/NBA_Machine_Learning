# Use specific platform architecture for Lambda
FROM public.ecr.aws/lambda/python:3.11

# Copy requirements and lambda function
COPY requirements.txt ${LAMBDA_TASK_ROOT}
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Install Python dependencies
RUN pip install -r requirements.txt

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"]