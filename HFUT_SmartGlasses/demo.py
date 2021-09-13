import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_coords, increment_path
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

# ===============================================================================================================相机参数
w = 640
h = 360
size = (w, h)  # 图像尺寸
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

# ===============================================================================================================相机参数

# ===========================================================================================================SGBM立体匹配
minDisparity = 0  # 最小视差值:最小可能的视差值,通常为0，但有时校正算法可以移动图像，因此需要相应调整此参数
numDisparities = 16 * 9  # 视差范围:最大视差减去最小视差,该值始终大于零,在当前的实现中，这个参数必须能被 16 整除
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
@torch.no_grad()  # 数据不需要计算梯度，也不会进行反向传播
def run(weights='yolov5s.pt',  # model.pt path(s)   模型路径
        source=['0'],  # ['0']单目摄像头 ['0', '1']双目 ...
        imgsz=w,  # inference size (pixels)   图片像素大小
        conf_thres=0.5,  # confidence threshold    置信度阀值
        iou_thres=0.5,  # NMS IOU threshold    极大值抑制IOU阀值
        max_det=1000,  # maximum detections per image   每张图片最大预测框个数
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu     CPU/GPU加速
        classes=None,  # filter by class: --class 0, or --class 0 2 3   筛选器只保留指定目标 person=0,bicycle=1...
        agnostic_nms=False,  # class-agnostic NMS   进行nms是否也去除不同类别之间的框，默认False
        augment=False,  # augmented inference   增强推理，推理的时候进行多尺度，翻转等操作(TTA)推理
        visualize=False,  # visualize features  可视化功能
        project='runs/detect',  # save results to project/name  保存地址
        name='exp',  # save results to project/name 保存目录名称
        exist_ok=True,  # existing project/name ok, do not increment   是否覆盖已有项目
        half=False,  # use FP16 half-precision inference    使用FP16半精度推理
        ):
    # Directories   目录
    # increment_path() 增加文件或目录路径
    # save_dir 运行结果保存地址 runs/detect/exp
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run

    # Initialize 初始化
    device = select_device(device)  # 返回使用的设备 cuda:* 或 cpu
    half &= device.type != 'cpu'  # half precision only supported on CUDA 半精度仅支持CUDA

    # Load model 加载模型
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride 输入和特征图之间的尺度比值
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names (列表: coco 80个不同类型)
    if half:
        model.half()  # to FP16
    cudnn.benchmark = True  # set True to speed up constant image size inference  将True设置为加速常量图像大小推断
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    # 2560*720 中 左图像为目标检测运算图像，左相机为主相机

    # 预测阶段
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    cap = cv2.VideoCapture(0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        t1 = time_sync()
        ref, frame = cap.read()
        img_L = frame[0:frame_h, 0:int(frame_w / 2)]
        img_R = frame[0:frame_h, int(frame_w / 2):frame_w]
        img_L = cv2.resize(img_L, dsize=(w, h))
        right_frame = cv2.resize(img_R, dsize=(w, h))

        s = np.stack([letterbox(img_L, imgsz, stride=stride)[0].shape], 0)  # shapes
        rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal 如果所有形状都相等，则使用矩形推理
        img = [letterbox(img_L, imgsz, auto=rect, stride=stride)[0]]
        img = np.stack(img, axis=0)
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)  # CPU GPU加速
        img = img.half() if half else img.float()  # uint8 to fp16/32 提升精度
        img /= 255.0  # 0 - 255 to 0.0 - 1.0 归一化
        if img.ndimension() == 3:  # 返回tensor的维度（整数）
            img = img.unsqueeze(0)  # 对张量的维度进行增加的操作
        pred = model(img, augment=augment,
                     visualize=increment_path(save_dir / Path(source).stem, mkdir=True) if visualize else False)[0]
        # NMS非极大值抑制
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        img1_rectified = cv2.remap(img_L, left_map1, left_map2, cv2.INTER_LINEAR)  # 图像校准
        img2_rectified = cv2.remap(right_frame, right_map1, right_map2, cv2.INTER_LINEAR)
        imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
        disp = SGBM_stereo.compute(imgL, imgR).astype(np.float32) / 16.0  # disparity返回视差
        dislist = cv2.reprojectImageTo3D(disp, Q)  # 返回_3dImage 3D图像

        # 目标检测
        for i, det in enumerate(pred):  # 遍历每一个预测框
            demo_img = img_L.copy()
            if len(det):  # 遍历预测框
                # Rescale boxes from img_size to im0 size  将框从 img_size 重新缩放为 im0 大小
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], demo_img.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class (c对应每个种类)

                    point1 = (int(xyxy[0] + 1*(xyxy[2]-xyxy[0])/4), int(xyxy[1] + 1*(xyxy[3]-xyxy[1])/4))
                    point2 = (int(xyxy[0] + 1*(xyxy[2]-xyxy[0])/4), int(xyxy[1] + 2*(xyxy[3]-xyxy[1])/4))
                    point3 = (int(xyxy[0] + 1*(xyxy[2]-xyxy[0])/4), int(xyxy[1] + 3*(xyxy[3]-xyxy[1])/4))
                    point4 = (int(xyxy[0] + 2*(xyxy[2]-xyxy[0])/4), int(xyxy[1] + 1*(xyxy[3]-xyxy[1])/4))
                    point5 = (int(xyxy[0] + 2*(xyxy[2]-xyxy[0])/4), int(xyxy[1] + 2*(xyxy[3]-xyxy[1])/4))
                    point6 = (int(xyxy[0] + 2*(xyxy[2]-xyxy[0])/4), int(xyxy[1] + 3*(xyxy[3]-xyxy[1])/4))
                    point7 = (int(xyxy[0] + 3*(xyxy[2]-xyxy[0])/4), int(xyxy[1] + 1*(xyxy[3]-xyxy[1])/4))
                    point8 = (int(xyxy[0] + 3*(xyxy[2]-xyxy[0])/4), int(xyxy[1] + 2*(xyxy[3]-xyxy[1])/4))
                    point9 = (int(xyxy[0] + 3*(xyxy[2]-xyxy[0])/4), int(xyxy[1] + 3*(xyxy[3]-xyxy[1])/4))
                    point_list = [point1, point2, point3, point4, point5, point6, point7, point8, point9]
                    distance = []
                    for point in point_list:
                        cv2.circle(img=demo_img, center=point, radius=2, color=colors(c, True), thickness=2)
                        dist = (dislist[point[1]][point[0]] / 5)[-1]
                        dist = dist * 0.373
                        if 0 <= dist < 1000:
                            distance.append(dist)
                    distance.sort()
                    if len(distance) < 3:
                        label = f'{names[c]} {int(conf * 100)}% null'
                    else:
                        dist = distance[1]
                        label = f'{names[c]} {int(conf * 100)}% {dist / 100:.2f}m'
                    plot_one_box(xyxy, demo_img, label=label, color=colors(c, True), line_thickness=2)
        t2 = time_sync()
        print(f'{t2 - t1:.3f}s Done')
        # demo_img = cv2.resize(demo_img, (960, 576))
        cv2.imshow('Demo', demo_img)
        x = (disp - minDisparity) / numDisparities
        cv2.imshow('SGNM_disparity', x)
        cv2.waitKey(1)  # 1 millisecond

run()