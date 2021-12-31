import cv2
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync
from utils.augmentations import letterbox

# ======================================================================================================================
# 设置参数
imgsz = 640
weights = './yolov5l.pt'
device = 'cpu'
half = False
line_thickness = 2
hide_labels = False
hide_conf = False
world_distance = [25,50,100,150,200,250,300,350,400,450]
raw_distance = [97,145,250,320,377,437,467,511,571, 620]
coff = np.polyfit(raw_distance,world_distance,2)  # Ax + B
A, B ,C= coff

# 初始化
stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
# half &= device.type != 'cpu'

# 载入模型
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, 'module') else model.names  # get class names
imgsz = check_img_size(imgsz, s=stride)  # check image size

# ----------------------------------------------------------------------------------------------------------------------
# 相机参数
size = (imgsz, int((imgsz*9)/16))  # 图像尺寸
# distCoeffs 畸变系数向量 (k_1, k_2, p_1, p_2, k_1) k径向畸变 p切向畸变
# 左相机矩阵
left_camera_matrix = np.array([[8.315761264277194e+02, -0.067100532264039, 6.270773295697649e+02],
                               [0, 8.317328040960693e+02, 4.950599408765926e+02],
                               [0., 0, 1.0000]])
# 左相机失真
left_distortion = np.array(
    [[-0.052497003322815, 0.219130301798893, -3.773141760469683e-04, -0.001161698975305, -0.247900058208785]])

# 右相机矩阵
right_camera_matrix = np.array([[8.365744888406663e+02, 0.337600854848549, 6.496769813852944e+02],
                                [0, 8.367057793927418e+02, 4.919242249466425e+02],
                                [0., 0, 1.0000]])
# 右相机失真
right_distortion = np.array(
    [[-0.055965519717472, 0.238054946405461, -2.395780095168055e-04, -4.380167767466566e-04, -0.288577515484556]])

# 相机旋转矩阵
R = np.matrix([[0.999936367261596, -2.560085328833769e-04, 0.011278115414964],
               [2.352383714212434e-04, 0.999998274148975, 0.001842922130743],
               [-0.011278567754408, -0.001840151815058, 0.999934701743422]])
# 相机平移矢量
T = np.array([-1.198999022890396e+02, 0.022648001773392, -0.885207100250237])
# R T 都可以通过 cv2.stereoCalibrate() 计算得出

# 立体校正教程 https://www.cnblogs.com/zhiyishou/p/5767592.html
# 进行立体校正 stereoRectify() https://blog.csdn.net/zfjBIT/article/details/94436644
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion,
                                                                  size, R, T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

minDisparity = 0  # 最小视差值:最小可能的视差值,通常为0，但有时校正算法可以移动图像，因此需要相应调整此参数
numDisparities = 16 * 13  # 视差范围:最大视差减去最小视差,该值始终大于零,在当前的实现中，这个参数必须能被 16 整除
blockSize = 1  # 匹配块大小(SADWindowSize): 它必须是一个奇数 >=1，通常应该在 3-11 范围内
P1 = 8 * 3 * blockSize * blockSize  # 第一个参数控制视差平滑度，对相邻像素之间正负 1 的视差变化惩罚
P2 = 4 * P1  # 第二个参数控制视差平滑度，值越大视差越平滑，相邻像素之间视差变化超过 1 的惩罚
disp12MaxDiff = -1  # 左右视差检查中允许的最大差异（以整数像素为单位）。 将其设置为 -1 以禁用检查。
preFilterCap = None  # 预过滤图像像素的截断值:该算法首先计算每个像素的 x 导数，并按 [-preFilterCap, preFilterCap] 间隔裁剪其值。将结果值传递给 Birchfield-Tomasi 像素成本函数
uniquenessRatio = 3  # 视差唯一性百分比， 视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15
speckleWindowSize = 0  # 平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50-200的范围内
speckleRange = 2  # 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
mode = None  # 将其设置为 StereoSGBM::MODE_HH 以运行完整的两遍动态编程算法。 它将消耗 O(W*H*numDisparities) 字节，这对于 640x480 立体声来说很大，对于 HD 尺寸的图片来说很大。 默认情况下，它设置为 false
SGBM_stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
                                        numDisparities=numDisparities,
                                        blockSize=blockSize,
                                        P1=P1,
                                        P2=P2,
                                        disp12MaxDiff=disp12MaxDiff,
                                        preFilterCap=preFilterCap,
                                        uniquenessRatio=uniquenessRatio,
                                        speckleWindowSize=speckleWindowSize,
                                        speckleRange=speckleRange,
                                        mode=mode
                                        )
# ======================================================================================================================

cap = cv2.VideoCapture(0)
assert cap.isOpened(), f'Failed to open cam !'
cudnn.benchmark = True  # set True to speed up constant image size inference

# cap.set();  # 画面宽
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)  # 画面高
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # 画面高
cap.set(cv2.CAP_PROP_FPS, 60)
# cap.set(cv2.CV_CAP_PROP_FOURCC, cv2.CV_FOURCC('M', 'J', 'P', 'G')); # 压缩格式 mjpg
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    success, img = cap.read()

    # 分离左右图像,调整图像大小
    img_L = img[0:frame_h, 0:int(frame_w / 2)]  # yolo
    img_R = img[0:frame_h, int(frame_w / 2):frame_w]
    img_L = cv2.resize(img_L, dsize=size)
    img_R = cv2.resize(img_R, dsize=size)

    # ================================================SGBM==============================================================
    img1_rectified = cv2.remap(img_L, left_map1, left_map2, cv2.INTER_LINEAR)  # 图像校准
    img2_rectified = cv2.remap(img_R, right_map1, right_map2, cv2.INTER_LINEAR)
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
    disp = SGBM_stereo.compute(imgL, imgR).astype(np.float32) / 16.0  # disparity返回视差
    dislist = cv2.reprojectImageTo3D(disp, Q)  # 返回_3dImage 3D图像
    # cv2.imshow('l', img1_rectified)
    # cv2.imshow('r', img2_rectified)
    # cv2.waitKey(1)
    # continue
    # ================================================yolo v5===========================================================
    t0 = time_sync()
    img0 = img_L.copy()
    img_L = [letterbox(img_L, imgsz, stride=stride)[0]]  # 保持图片的长宽比例，剩下的部分采用灰色填充
    # Stack
    img_L = np.stack(img_L, 0)
    # Convert
    img_L = img_L[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    img_L = np.ascontiguousarray(img_L)  # 将内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
    img_L = torch.from_numpy(img_L).to(device)  # 从numpy.ndarray创建一个张量
    img_L = img_L.half() if half else img_L.float()  # uint8 to fp16/32
    img_L = img_L / 255.0  # 0 - 255 to 0.0 - 1.0
    pred = model(img_L)[0]  # 预测
    det = non_max_suppression(prediction=pred, conf_thres=0.3, iou_thres=0.45, max_det=20)[0]  # NMS非极大值抑制
    s = 'cam: '
    s += '%gx%g ' % img_L.shape[2:]  # print string
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(img0, line_width=line_thickness, example=str(names))
    if len(pred):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img_L.shape[2:], det[:, :4], img0.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            # ----------------------------------------------------------------------------------------------------------
            point1 = (int(xyxy[0] + 1 * (xyxy[2] - xyxy[0]) / 4), int(xyxy[1] + 1 * (xyxy[3] - xyxy[1]) / 4))
            point2 = (int(xyxy[0] + 1 * (xyxy[2] - xyxy[0]) / 4), int(xyxy[1] + 2 * (xyxy[3] - xyxy[1]) / 4))
            point3 = (int(xyxy[0] + 1 * (xyxy[2] - xyxy[0]) / 4), int(xyxy[1] + 3 * (xyxy[3] - xyxy[1]) / 4))
            point4 = (int(xyxy[0] + 2 * (xyxy[2] - xyxy[0]) / 4), int(xyxy[1] + 1 * (xyxy[3] - xyxy[1]) / 4))
            point5 = (int(xyxy[0] + 2 * (xyxy[2] - xyxy[0]) / 4), int(xyxy[1] + 2 * (xyxy[3] - xyxy[1]) / 4))
            point6 = (int(xyxy[0] + 2 * (xyxy[2] - xyxy[0]) / 4), int(xyxy[1] + 3 * (xyxy[3] - xyxy[1]) / 4))
            point7 = (int(xyxy[0] + 3 * (xyxy[2] - xyxy[0]) / 4), int(xyxy[1] + 1 * (xyxy[3] - xyxy[1]) / 4))
            point8 = (int(xyxy[0] + 3 * (xyxy[2] - xyxy[0]) / 4), int(xyxy[1] + 2 * (xyxy[3] - xyxy[1]) / 4))
            point9 = (int(xyxy[0] + 3 * (xyxy[2] - xyxy[0]) / 4), int(xyxy[1] + 3 * (xyxy[3] - xyxy[1]) / 4))
            point_list = [point1, point2, point3, point4, point5, point6, point7, point8, point9]
            distance = []
            for point in point_list:
                # cv2.circle(img=demo_img, center=point, radius=2, color=colors(c, True), thickness=1)
                dist = (dislist[point[1]][point[0]] / 5)[-1]
                cv2.circle(img=img0, center=point, radius=1, color=(0, 0, 225), thickness=1)
                if 0 <= dist < 10000:
                    distance.append(dist)
            distance.sort()
            if len(distance) <= 3:
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {int(conf * 100)}% null')
            else:
                dist = (distance[0] + distance[1] + distance[2])/3
                CM_distance = A*dist**2 + B*dist + C
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {int(conf * 100)}% {int(CM_distance)}cm')
                # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {int(conf * 100)}% {int(dist)}cm')
            # ----------------------------------------------------------------------------------------------------------
            # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

    t1 = time_sync()
    # Print time (inference-only)
    print(f'{s}Done. ({t1 - t0:.3f}s)')

    cv2.imshow('Demo', img0)
    x = (disp - minDisparity) / numDisparities
    cv2.imshow('SGNM_disparity', x)
    cv2.waitKey(1)  # 1 millisecond
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
    # ==================================================================================================================


