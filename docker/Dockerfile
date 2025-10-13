FROM python:3.11-slim

WORKDIR /app

COPY docker/requirements/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY Keyword_extractor.py .

CMD ["python", "Keyword_extractor.py"]