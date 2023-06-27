FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.h5 .
COPY inference.py .
COPY train.py .
COPY readme.md .

# Disable TensorFlow logging outputs
ENV TF_CPP_MIN_LOG_LEVEL=3
