# Use specific platform architecture for Lambda
FROM public.ecr.aws/lambda/python:3.11

# Install Chrome and ChromeDriver using Chrome for Testing
RUN yum update -y && \
    yum install -y wget unzip && \
    yum install -y atk cups-libs gtk3 libXcomposite libXcursor libXdamage libXext libXi libXrandr libXss libXtst pango at-spi2-atk libXt xorg-x11-server-Xvfb xorg-x11-xauth dbus-glib dbus-glib-devel nss alsa-lib && \
    wget -q https://storage.googleapis.com/chrome-for-testing-public/120.0.6099.109/linux64/chrome-linux64.zip && \
    unzip chrome-linux64.zip && \
    mv chrome-linux64 /opt/chrome && \
    chmod +x /opt/chrome/chrome && \
    wget -q https://storage.googleapis.com/chrome-for-testing-public/120.0.6099.109/linux64/chromedriver-linux64.zip && \
    unzip chromedriver-linux64.zip && \
    mv chromedriver-linux64/chromedriver /opt/chromedriver && \
    chmod +x /opt/chromedriver

# Copy requirements and lambda function
COPY requirements.txt ${LAMBDA_TASK_ROOT}
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Install Python dependencies
RUN pip install -r requirements.txt

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"]
