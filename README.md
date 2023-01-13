# yolov7-tflite
Infer image and video with yolov7's tflite model

## 1. pip install
```bash
pip install -U pip && pip install opencv-python tflite_runtime
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

# vieo
python tflite_inference.py --mo video -i <video path>
```
## 4. About me
BLOG: https://chantastu.hatenablog.com/archive
