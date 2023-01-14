FROM debian:stable-slim
USER root

LABEL version="1.0"
LABEL description="yolov7をtfliteモデルで実行"

RUN apt-get update
RUN apt-get -y install pip
RUN apt-get -y install git && apt-get -y install libgl1-mesa-dev && apt-get -y install libglib2.0-0
RUN pip install -U pip
RUN pip install --no-cache-dir opencv-python tflite_runtime