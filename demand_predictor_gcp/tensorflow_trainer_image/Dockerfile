
FROM gcr.io/deeplearning-platform-release/base-cpu
RUN pip install -U fire tensorflow==2.1.1
WORKDIR /app
COPY train.py .

ENTRYPOINT ["python", "train.py"]
