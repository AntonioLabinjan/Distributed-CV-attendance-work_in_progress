# 1. Bazni image
FROM python:3.12-slim

# 2. Postavi radni direktorij
WORKDIR /app

# 3. Kopiraj requirements i instaliraj ovisnosti
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Kopiraj cijeli projekt
COPY . .

# 5. Kopiraj dodatne foldere (dataset i credentials)
# (Ova naredba će biti redundantna ako su ti već u rootu i uključeni u COPY . .,
# ali dodajemo je za sigurnost/čist primjer.)
COPY dataset/ dataset/
COPY credentials/ credentials/

# 6. Expose porta
EXPOSE 8000

# 7. Pokreni aplikaciju
CMD ["uvicorn", "server_FASTAPI:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
