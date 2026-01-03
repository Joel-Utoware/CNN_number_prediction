FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV/PIL
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy folders
COPY ./api ./api
COPY ./model ./model

EXPOSE 8000


# we provide a default start command here just in case.
# old:CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}