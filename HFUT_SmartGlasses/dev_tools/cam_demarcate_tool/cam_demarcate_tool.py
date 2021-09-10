import cv2
import numpy as np
import glob
"""
张氏标定相关具体讲解和相机采样过程，respect to 张正友：
（上、中、下）
https://blog.csdn.net/qq_37059483/article/details/79481014
https://blog.csdn.net/qq_37059483/article/details/79482541
https://blog.csdn.net/qq_37059483/article/details/79836411
"""
# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob("l_imgs/*.jpg")
for fname in images:
    img = cv2.imread(fname)
    cv2.imshow('img', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)
    print(ret)

    if ret:

        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        # print(corners2)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (8, 6), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
        cv2.imshow('img', img)
        cv2.waitKey(2000)

print(len(img_points))
cv2.destroyAllWindows()

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx:\n", mtx)  # 内参数矩阵
print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
print("tvecs:\n", tvecs)  # 平移向量  # 外参数
print("-----------------------------------------------------")


import cv2
import glob  # python内建文件操作模块

l_imgs_path_list = glob.glob("l_imgs/*.jpg")
r_imgs_path_list = glob.glob("r_imgs/*.jpg")
for l_img_path in l_imgs_path_list:
    r_img_path = l_img_path.replace('l', 'r')
    l_img = cv2.imread(l_img_path)
    r_img = cv2.imread(r_img_path)
    cv2.imshow('l_img', l_img)
    cv2.imshow('r_img', r_img)



