# Start from an official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY docker/requirements/requirements_baseline.txt .

# Install the Python libraries
RUN pip install --no-cache-dir -r requirements_baseline.txt

# Download the spaCy model AND all necessary NLTK packages
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader punkt stopwords punkt_tab # ADDED punkt_tab HERE

# Copy the baseline script into the container
COPY baseline_script.py .

# Set the command to run when the container starts
CMD ["python", "baseline_script.py"]