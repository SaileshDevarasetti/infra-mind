FROM python:3.10-slim
WORKDIR /app

# Copy project
COPY . /app

# Install runtime dependencies
RUN pip install --no-cache-dir -r infra_mind/requirements.txt

# Expose default port used by the FastAPI server
EXPOSE 8080

# Default command: run the API server
CMD ["uvicorn", "infra_mind.api.server:app", "--host", "0.0.0.0", "--port", "8080"]
