import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import random
import ctypes
import pycuda.driver as cuda
import time
import math

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
host_inputs  = []
cuda_inputs  = []
host_outputs = []
cuda_outputs = []
bindings = []


class YoloTRT():
    def __init__(self, library, engine, conf, yolo_ver):
        self.CONF_THRESH = conf 
        self.IOU_THRESHOLD = 0.4
        self.LEN_ALL_RESULT = 38001
        self.LEN_ONE_RESULT = 38
        self.yolo_version = yolo_ver
        self.categories = ["car"]
        
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

        ctypes.CDLL(library)

        with open(engine, 'rb') as f:
            serialized_engine = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.batch_size = self.engine.max_batch_size

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.input_w = self.engine.get_binding_shape(binding)[-1]
                self.input_h = self.engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

    def PreProcessImg(self, img):
        image_raw = img
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128))
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def Inference(self, img):
        input_image, image_raw, origin_h, origin_w = self.PreProcessImg(img)
        np.copyto(host_inputs[0], input_image.ravel())
        stream = cuda.Stream()
        self.context = self.engine.create_execution_context()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        t1 = time.time()
        self.context.execute_async(self.batch_size, bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        t2 = time.time()
        output = host_outputs[0]
                
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.PostProcess(output[i * self.LEN_ALL_RESULT: (i + 1) * self.LEN_ALL_RESULT], origin_h, origin_w)
            
        det_res = []
        for j in range(len(result_boxes)):
            box = result_boxes[j]
            det = dict()
            det["class"] = self.categories[int(result_classid[j])]
            det["conf"] = result_scores[j]
            det["box"] = box 
            det_res.append(det)
            self.PlotBbox(box, img, label="{}:{:.2f}".format(self.categories[int(result_classid[j])], result_scores[j]),)
        
        # 加上座標框在圖片中央
        self.PlotCord(img)

        return det_res, t2-t1

    def PostProcess(self, output, origin_h, origin_w):
        num = int(output[0])
        if self.yolo_version == "v5":
            pred = np.reshape(output[1:], (-1, self.LEN_ONE_RESULT))[:num, :]
            pred = pred[:, :6]
        elif self.yolo_version == "v7":
            pred = np.reshape(output[1:], (-1, 6))[:num, :]
        
        boxes = self.NonMaxSuppression(pred, origin_h, origin_w, conf_thres=self.CONF_THRESH, nms_thres=self.IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid
    
    def NonMaxSuppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        boxes = prediction[prediction[:, 4] >= conf_thres]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        confs = boxes[:, 4]
        boxes = boxes[np.argsort(-confs)]
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes
    
    def xywh2xyxy(self, origin_h, origin_w, x):
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h
        return y
    
    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou
    
    def PlotBbox(self, x, img, color=None, label=None, line_thickness=None):
        tl = (line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

'''
        # 計算車輛座標
        target_x, target_y = self.get_target_position(
            int(x[0]),
            int(x[1]), 
            int(x[2]), 
            int(x[3]),
        )
'''
        # label = label + ' (' + str(target_x) + ', '+ str(target_y) + ')'

        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA,)
    
    '''
    def get_target_position(
        self,
        bbox_x1, 
        bbox_y1, 
        bbox_x2, 
        bbox_y2, 
        drone_x=0, 
        drone_y=0,
        drone_direction = 0,
        height=60, 
        hfov_degree=90,
        vfov_degree=90,
        image_w = 1000,
        image_h = 1000,
    ):
        hfov_rad = hfov_degree*math.pi/180
        vfov_rad = vfov_degree*math.pi/180

        # 目標在圖片上的座標
        bbox_x = bbox_x2 - bbox_x1
        bbox_y = bbox_y2 - bbox_y1

        # 車輛在圖片上的XY位置
        car_in_drone_x = bbox_x - image_w/2
        car_in_drone_y = bbox_y - image_h/2

        # 車輛在圖片上的極座標
        r = (car_in_drone_x**2 + car_in_drone_y**2)**0.2
        theta_rad = math.atan(car_in_drone_y/car_in_drone_x)
        theta_deg = 180 * theta_rad/math.pi

        # 實際地理寬度 W，物體到中心的實際地理距離R， image_w圖片寬度
        R = 2*height*math.tan(hfov_rad/2)*r/image_w

        # 實際車輛方位與北方順時針夾角
        car_direction_deg = drone_direction + theta_deg
        car_direction_rad = car_direction_deg*math.pi/180

        car_delta_x = R * math.sin(car_direction_rad)
        car_delta_y = R * math.cos(car_direction_rad)

        # 車輛實際地理座標
        ##################################這裡需要再修改成TWD97運算#######################################
        car_x = round(drone_x + car_delta_x, 3)
        car_y = round(drone_y + car_delta_y, 3)

        # target_x = drone_x + 2*height*math.tan(hfov_rad/2)*(bbox_x-0.5)
        # target_y = drone_y + 2*height*math.tan(vfov_rad/2)*(0.5-bbox_y)
        return car_x, car_y
    '''

    def PlotCord(self, img):
        cord_x, cord_y = 248170.787, 2652129.936 
        text = '(' + str(cord_x) + ', ' + str(cord_y) + ')'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 0, 0)  # 藍色
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness=1)
        height, width, channels = img.shape
        x = (width - text_width) // 2
        y = (height + text_height + baseline) // 2
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness=2)