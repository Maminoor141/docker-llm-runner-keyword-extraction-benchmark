# Start from an official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY docker/requirements/requirements_comparison.txt .

# Install the Python libraries
RUN pip install --no-cache-dir -r requirements_comparison.txt

# Copy the comparison script into the container
COPY comparison.py .

# Copy the dataset folder from the host into the container.
COPY dataset/ ./dataset/

# Set the command to run when the container starts
CMD ["python", "comparison.py"]