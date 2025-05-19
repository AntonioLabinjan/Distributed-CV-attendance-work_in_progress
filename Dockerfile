FROM python:3.12-slim

WORKDIR /app

# Kopiraj sve potrebne fajlove u container
COPY server.py requirements.txt /app/
COPY dataset /app/dataset

# Instaliraj dependencies iz requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --verbose

EXPOSE 6010

CMD ["python", "server.py"]

