# 1. Bazni image
FROM python:3.12-slim

# 2. Postavi radni direktorij
WORKDIR /app

# 3. Kopiraj requirements i instaliraj ovisnosti
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Kopiraj ostatak projekta
COPY . .

# 5. Učitaj env varijable (ako koristiš python-dotenv, FastAPI će ih automatski pročitati iz .env ako to ručno napraviš u kodu)
# Ako želiš da Docker direktno koristi ENV, koristi ENV naredbu (alternativa)
# ENV MY_SECRET=12345 

# 6. Expose porta (FastAPI default)
EXPOSE 8000

# 7. Pokreni aplikaciju
CMD ["uvicorn", "server_FASTAPI:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
