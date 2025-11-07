FROM python:3.11-slim

# Install OpenJDK (headless) for JPype
RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV JPYPE_JVM=$JAVA_HOME/lib/server/libjvm.so

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt
COPY . .

# Uvicorn binds to Cloud Run's dynamic $PORT
ENV PYTHONUNBUFFERED=1
CMD ["sh","-c","uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}"]