# yolov7-tflite
Infer image and video with yolov7's tflite model

## 1. pip install
```bash
pip install -U pip && pip install opencv-python tflite_runtime
```
## 2. Inference
```bash
# image
python tflite_inference.py -i <image path>

# vieo
python tflite_inference.py --mo video -i <video path>
```
