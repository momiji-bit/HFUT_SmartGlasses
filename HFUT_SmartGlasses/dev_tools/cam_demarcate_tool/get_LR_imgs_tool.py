# 双目同步摄像机左右画面分割
# 获取左右相机图片
# 请拍摄棋盘格图片
import cv2

cap = cv2.VideoCapture(0)
num = 0
print("按 esc 退出get_LR_imgs_tool")
print("按 空格 拍摄左右相机照片")
while cap.isOpened():
    ref, frame = cap.read()
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    left_frame = frame[0:frame_h, 0:int(frame_w / 2)]
    right_frame = frame[0:frame_h, int(frame_w / 2):frame_w]
    # cv2.imshow("left", left_frame)
    # cv2.imshow("right", right_frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == 32:
        cv2.imwrite("l_imgs/l_cam_img{}.jpg".format(num), left_frame)
        cv2.imwrite("r_imgs/r_cam_img{}.jpg".format(num), right_frame)
        print('l_cam_img{} r_cam_img{} Done!'.format(num, num))
        num += 1
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
