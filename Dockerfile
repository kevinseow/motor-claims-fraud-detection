FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "job_runner.py"]
CMD []
