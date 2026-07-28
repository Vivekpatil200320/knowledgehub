FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch is a transitive dependency of sentence-transformers (the local reranker).
# Installing requirements.txt directly resolves the default PyPI wheel, which is the
# CUDA build — several GB of nvidia-cu*/triton packages this CPU-only container never
# uses, turning an ~80s image build into 15+ minutes. Installing the CPU wheel from
# PyTorch's own index first satisfies that dependency before pip gets a chance to
# reach for the CUDA one.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY evals/ ./evals/

RUN mkdir -p /app/uploads /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
