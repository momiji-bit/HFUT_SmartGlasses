import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_coords, increment_path, set_logging
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_sync

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path


def nothing(x):
    pass


cv2.namedWindow("SGNM_disparity")
cv2.createTrackbar("minDisparity", 'SGNM_disparity', 0, 10, nothing)
cv2.createTrackbar("numDisparities", 'SGNM_disparity', 10, 40, nothing)
cv2.createTrackbar("blockSize", 'SGNM_disparity', 1, 20, nothing)
cv2.createTrackbar("disp12MaxDiff", 'SGNM_disparity', -1, 500, nothing)
cv2.createTrackbar("uniquenessRatio", 'SGNM_disparity', 3, 20, nothing)
cv2.createTrackbar("speckleWindowSize", 'SGNM_disparity', 0, 300, nothing)
cv2.createTrackbar("speckleRange", 'SGNM_disparity', 2, 4, nothing)
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
# minDisparity = 0  # 最小视差值:最小可能的视差值,通常为0，但有时校正算法可以移动图像，因此需要相应调整此参数
# numDisparities = 16 * 4  # 视差范围:最大视差减去最小视差,该值始终大于零,在当前的实现中，这个参数必须能被 16 整除
# blockSize = 1  # 匹配块大小(SADWindowSize): 它必须是一个奇数 >=1，通常应该在 3-11 范围内
# P1 = 8 * 3 * blockSize * blockSize  # 第一个参数控制视差平滑度，对相邻像素之间正负 1 的视差变化惩罚
# P2 = 4 * P1  # 第二个参数控制视差平滑度，值越大视差越平滑，相邻像素之间视差变化超过 1 的惩罚
# disp12MaxDiff = -1  # 左右视差检查中允许的最大差异（以整数像素为单位）。 将其设置为 -1 以禁用检查。
# preFilterCap = None  # 预过滤图像像素的截断值:该算法首先计算每个像素的 x 导数，并按 [-preFilterCap, preFilterCap] 间隔裁剪其值。将结果值传递给 Birchfield-Tomasi 像素成本函数
# uniquenessRatio = 9  # 视差唯一性百分比， 视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15
# speckleWindowSize = 20  # 平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50-200的范围内
# speckleRange = 2  # 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
# mode = None  # 将其设置为 StereoSGBM::MODE_HH 以运行完整的两遍动态编程算法。 它将消耗 O(W*H*numDisparities) 字节，这对于 640x480 立体声来说很大，对于 HD 尺寸的图片来说很大。 默认情况下，它设置为 false




def dis_co(frame1, frame2):


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
    img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)  # 图像校准
    img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
    disp = SGBM_stereo.compute(imgL, imgR).astype(np.float32) / 16.0  # disparity返回视差
    threeD = cv2.reprojectImageTo3D(disp, Q)  # 返回_3dImage 3D图像
    return threeD, disp

cap = cv2.VideoCapture(0)
cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
while cap.isOpened():
    t1 = time_sync()

    minDisparity = cv2.getTrackbarPos('minDisparity', 'SGNM_disparity')
    numDisparities = 16 * cv2.getTrackbarPos('numDisparities', 'SGNM_disparity')
    blockSize = cv2.getTrackbarPos('blockSize', 'SGNM_disparity')
    P1 = 8 * 3 * blockSize * blockSize
    P2 = 32 * 3 * blockSize * blockSize
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'SGNM_disparity')
    preFilterCap = cv2.getTrackbarPos('preFilterCap', 'SGNM_disparity')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'SGNM_disparity')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'SGNM_disparity')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'SGNM_disparity')
    mode = None

    ref, frame = cap.read()
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    left_frame = frame[0:frame_h, 0:int(frame_w / 2)]
    right_frame = frame[0:frame_h, int(frame_w / 2):frame_w]
    left_frame = cv2.resize(left_frame, dsize=(w, h))
    right_frame = cv2.resize(right_frame, dsize=(w, h))

    dislist, disp = dis_co(left_frame, right_frame)
    t2 = time_sync()
    print(f'{t2 - t1:.3f}s Done')
    cv2.imshow('SGNM_disparity', (disp - minDisparity) / numDisparities)
    cv2.waitKey(1)  # 1 millisecond