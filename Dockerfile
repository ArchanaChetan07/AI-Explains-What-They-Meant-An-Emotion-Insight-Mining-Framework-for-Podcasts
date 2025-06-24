# Use official Python base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else into /app
COPY . .

# Expose Flask port
EXPOSE 5000

# Run Flask
CMD ["python", "app/app.py"]
