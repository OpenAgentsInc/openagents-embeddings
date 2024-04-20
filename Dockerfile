FROM python:3.11.9-slim
ADD src /app
ADD requirements.txt /app
WORKDIR /app
RUN pip install 'transformers[torch]'
RUN pip install -r requirements.txt

ENV POOL_ADDRESS="127.0.0.1"
ENV POOL_PORT="5000"
ENV TRANSFORMERS_DEVICE="-1"
ENV CACHE_PATH="/cache"
ENV HF_HOME="/cache/hugginface"
ENV MODEL="intfloat/multilingual-e5-base"

VOLUME /cache

CMD ["python", "-u", "main.py"]
