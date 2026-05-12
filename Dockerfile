# Python 3.10 use karo
FROM python:3.10-slim

# Server pe /app folder mein kaam karo
WORKDIR /app

# Pehle requirements install karo
COPY requirements.txt .
RUN pip install -r requirements.txt

# Saari files copy karo
COPY . .

# Port 7860 open karo (HuggingFace ka default)
EXPOSE 7860

# Server start karo
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]