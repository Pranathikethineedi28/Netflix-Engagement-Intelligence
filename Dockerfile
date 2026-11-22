# Engagement-risk prediction API.
# The model is trained during the build from the data in this repo, so the image never depends on a
# checked-in binary and every build is reproducible from code + data.
FROM python:3.12-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY data/user_master_table.csv data/user_master_table.csv
RUN python -m src.train --no-mlflow

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
