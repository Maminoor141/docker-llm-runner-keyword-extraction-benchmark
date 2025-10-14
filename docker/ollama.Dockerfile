# Start from an official Python image
FROM ollama/ollama

# Copy the entrypoint script from the docker directory into the container root
COPY docker/entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable inside the container
RUN chmod +x /entrypoint.sh

# Set the entrypoint to run this script when the container starts
ENTRYPOINT ["/entrypoint.sh"]