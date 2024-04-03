import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

from yolov7.coco_classes import COCO_CLASSES
from yolov7.color_list import _COLORS


class Yolov7tflite(object):
    def __init__(
        self,
        model_path,
        img_size,
        num_of_threads = 2,
        class_score_th=0.3,
        nms_th=0.45,
        nms_score_th=0.1,
        with_p6=False,
        extract=None
    ):
        
        self.img_size = img_size
        # 閾値
        self.class_score_th = class_score_th
        self.nms_th = nms_th
        self.nms_score_th = nms_score_th

        self.with_p6 = with_p6
        self.extract = extract

        # モデル読み込み
        self.interpreter = self.make_interpreter(model_path, num_of_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        _, height, width, channel = self.input_details[0]["shape"]
        self.input_shape = (height, width)
        print("Interpreter(height, width, channel): ", height, width, channel)

    def make_interpreter(
            self, model_file, num_of_threads, delegate_library=None, delegate_option=None
        ):
            if delegate_library is not None:
                return tflite.Interpreter(
                    model_path=model_file, experimental_delegates=[
                        tflite.load_delegate(delegate_library, options=delegate_option)],
                )
            else:
                print("delegate_library is None")
                return tflite.Interpreter(model_path=model_file, num_threads=num_of_threads)

    def inference(self, img):
        # 前処理
        image, ratio, dwdh = self.preproc(img)
        image = image.reshape(-1, self.input_shape[0], self.input_shape[1],3)

        # 推論実施
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            image,
        )
        self.interpreter.invoke()
        
        results = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        reslut_img = self.visual(img, results, ratio, dwdh)
        return reslut_img

    def preproc(self,img):
        image = img.copy()
        image, ratio, dwdh = self.letterbox(image, new_shape=self.img_size, auto=False)
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255
        im.shape
        
        return im, ratio, dwdh
    
    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)
    
    def visual(self, img, outputs, ratio, dwdh):
        ori_images = [img.copy()]
<<<<<<< HEAD
        
=======
        print(outputs[0])
>>>>>>> 48c9c08080a012e8a3fa28ed5188f049e083d300
        if self.extract is not None:
            extract_cls_i = COCO_CLASSES.index(self.extract)
            
            for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
                if int(cls_id) == extract_cls_i:
                    if self.class_score_th is not None:
                        if score > self.class_score_th:
                            self.vis(ori_images, batch_id,x0,y0,x1,y1,cls_id,score, ratio, dwdh)
                    else:
                        self.vis(ori_images, batch_id,x0,y0,x1,y1,cls_id,score, ratio, dwdh,)
        else:
            for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
                if self.class_score_th is not None:
                    if score > self.class_score_th:
                        self.vis(ori_images, batch_id,x0,y0,x1,y1,cls_id,score, ratio, dwdh)
                else:
                    self.vis(ori_images, batch_id,x0,y0,x1,y1,cls_id,score, ratio, dwdh)
            
        return ori_images[0]
    
    def vis(self, ori_images, batch_id,x0,y0,x1,y1,cls_id,score, ratio, dwdh):
        image = ori_images[int(batch_id)]
        
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        
        #class_name list
        names = list(COCO_CLASSES)
        
        cls_id = int(cls_id)
        score = round(float(score),3)
        name = names[cls_id]
        
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        
        cv2.rectangle(image,box[:2],box[2:],color,2)
        cv2.rectangle(
            image,
            (box[0], box[1] + 1),
            (box[0] + txt_size[0] + 1, box[1] + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(image,text,(box[0], box[1] + txt_size[1]),font,0.4,txt_color,thickness=1) 
