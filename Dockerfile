FROM python:3.11.9-slim
ADD src /app
ADD requirements.txt /app
WORKDIR /app
RUN pip install 'transformers[torch]'
RUN pip install -r requirements.txt

# Generic configuration
ENV OPENAI_API_KEY=""
ENV NLP_CLOUD_API_KEY=""


# Pool configuration
ENV POOL_ADDRESS="127.0.0.1"
ENV POOL_PORT="5000"
ENV HF_HOME="/cache/hugginface"
ENV CACHE_PATH="/cache"

# Embeddings configuration
ENV EMBEDDINGS_TRANSFORMERS_DEVICE="-1"
ENV EMBEDDINGS_MODEL="intfloat/multilingual-e5-base"
ENV EMBEDDINGS_MAX_TEXT_LENGTH=512
ENV EMBEDDINGS_ADD_MARKERS_TO_SENTENCES="true"

# Logging configuration
ENV LOG_LEVEL="debug"
ENV OPENOBSERVE_ENDPOINT=""
ENV OPENOBSERVE_ORG="default"
ENV OPENOBSERVE_STREAM="default"
ENV OPENOBSERVE_BASICAUTH=""
ENV OPENOBSERVE_USERNAME=""
ENV OPENOBSERVE_PASSWORD=""
ENV OPENOBSERVE_BATCHSIZE="21"
ENV OPENOBSERVE_FLUSH_INTERVAL="5000"
ENV OPENOBSERVE_LOG_LEVEL="debug"




VOLUME /cache




CMD ["python", "-u", "main.py"]
