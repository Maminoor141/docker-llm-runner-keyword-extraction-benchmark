# Start from the official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the new requirements file
COPY docker/requirements/requirements_api.txt .

# Install all Python libraries
RUN pip install --no-cache-dir -r requirements_api.txt

# Download the spaCy model AND all necessary NLTK packages
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader punkt stopwords punkt_tab

# Copy the API script into the container
COPY api_service/api.py .

# Expose the port the app will run on
EXPOSE 5001

# Set the command to run the Flask application
CMD ["flask", "--app", "api", "run", "--host=0.0.0.0", "--port=5001"]