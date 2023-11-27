FROM python:3.11

RUN pip install --upgrade pip

RUN mkdir -p /app/languagedetector

WORKDIR /app/languagedetector/

COPY requirements.txt /app/languagedetector/

RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "80"]