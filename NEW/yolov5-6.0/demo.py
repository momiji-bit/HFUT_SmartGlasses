import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync
from utils.augmentations import letterbox

# 设置参数
imgsz = 640
weights = './yolov5n.pt'
device = 'cpu'
half = False
line_thickness = 2
hide_labels = False
hide_conf = False

# 初始化
stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

# 载入模型
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, 'module') else model.names  # get class names
imgsz = check_img_size(imgsz, s=stride)  # check image size


cap = cv2.VideoCapture(0)
assert cap.isOpened(), f'Failed to open cam !'
# cudnn.benchmark = True  # set True to speed up constant image size inference

while True:
    t0 = time_sync()
    success, img = cap.read()

    # ==================================================================================================================
    # yolo v5
    img0 = img.copy()
    img = [letterbox(img, imgsz, stride=stride)[0]]  # 保持图片的长宽比例，剩下的部分采用灰色填充
    # Stack
    img = np.stack(img, 0)
    # Convert
    img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    img = np.ascontiguousarray(img)  # 将内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
    img = torch.from_numpy(img).to(device)  # 从numpy.ndarray创建一个张量
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    pred = model(img)[0]  # 预测
    det = non_max_suppression(prediction=pred, conf_thres=0.6, iou_thres=0.45, max_det=1000)[0]  # NMS非极大值抑制
    s = 'cam: '
    s += '%gx%g ' % img.shape[2:]  # print string
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(img0, line_width=line_thickness, example=str(names))
    if len(pred):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))
    t1 = time_sync()
    # Print time (inference-only)
    print(f'{s}Done. ({t1 - t0:.3f}s)')
    cv2.imshow('Demo', img0)
    cv2.waitKey(1)  # 1 millisecond
    # ==================================================================================================================


