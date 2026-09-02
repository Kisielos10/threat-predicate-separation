FROM python:3.12-alpine
WORKDIR /app
COPY app.py .
ENV LOG_PATH=/logs/requests.jsonl PORT=8081
EXPOSE 8081
CMD ["python", "-u", "app.py"]
