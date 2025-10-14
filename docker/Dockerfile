# Start from an official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY docker/requirements/requirements.txt .

# Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the keyword extraction script into the container
COPY Keyword_extractor.py .

# Set the command to run when the container starts
CMD ["python", "Keyword_extractor.py"]