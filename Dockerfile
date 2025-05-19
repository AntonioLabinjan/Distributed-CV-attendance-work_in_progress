FROM python:3.12-slim

WORKDIR /app

# Copy all stuff in the container
COPY server.py requirements.txt /app/
COPY dataset /app/dataset

# Install all dependecies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --verbose

EXPOSE 6010

CMD ["python", "server.py"]

