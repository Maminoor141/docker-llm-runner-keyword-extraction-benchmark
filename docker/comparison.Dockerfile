FROM python:3.11-slim

WORKDIR /app

COPY docker/requirements/requirements_comparison.txt .

RUN pip install --no-cache-dir -r requirements_comparison.txt

COPY comparison.py .

COPY dataset/ ./dataset/

CMD ["python", "comparison.py"]