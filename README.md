# yolov7-tflite
Infer image and video with yolov7's tflite model

## 1. Dev env
### 1.1 pip install
```bash
pip install -U pip && pip install opencv-python tflite_runtime
```
### 1.2 Docker
Dockerfile
```
docker build -t tatsuya060504/yolov7-tflite:wsl2 .
docker run -it --name=yolov7-tflite -v /home/<user>/yolov7-tflite:/home tatsuya060504/yolov7-tflite:wsl2
```
Dockerhub
```bash
docker pull tatsuya060504/yolov7-tflite:wsl2
docker run -it --name=yolov7-tflite -v /home/<user>/yolov7-tflite:/home tatsuya060504/yolov7-tflite:wsl2
```

## 2. tflite model download
```bash
cd model
sh download_yolov7_tiny_640x640_tflite.sh
```

## 3. Inference
```bash
# image
python tflite_inference.py -i <image path>

# video
python tflite_inference.py --mo video -i <video path>

# Webcam
python tflite_inference.py -mo webcam -i 0
```
## 4. About me
BLOG: https://chantastu.hatenablog.com/archive
