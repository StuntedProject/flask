FROM python:3.8-slim-buster

WORKDIR /app/ml-backend

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirement.txt requirement.txt
RUN pip3 install -r requirement.txt 

COPY . . 

CMD ["python3", "app.py"] 