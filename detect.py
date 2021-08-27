import argparse
import sys
import time
import multiprocessing
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)   模型路径
        source=['0'],  # ['0']单目摄像头 ['0', '1']双目 ...
        imgsz=640,  # inference size (pixels)   图片像素大小
        conf_thres=0.25,  # confidence threshold    置信度阀值
        iou_thres=0.45,  # NMS IOU threshold    极大值抑制IOU阀值
        max_det=1000,  # maximum detections per image   每张图片最大预测框个数
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu     CPU/GPU加速
        view_img=False,  # show results     显示结果
        save_txt=False,  # save results to *.txt    保存结果到*.txt
        save_conf=False,  # save confidences in --save-txt labels   将置信度保存在 --save-txt 标签中
        save_crop=True,  # save cropped prediction boxes   保存裁剪好的预测框
        save_img=False,  # do not save images/videos  是否保存结果
        classes=None,  # filter by class: --class 0, or --class 0 2 3   筛选器只保留指定目标 person=0,bicycle=1...
        agnostic_nms=False,  # class-agnostic NMS   进行nms是否也去除不同类别之间的框，默认False
        augment=False,  # augmented inference   增强推理，推理的时候进行多尺度，翻转等操作(TTA)推理
        visualize=False,  # visualize features  可视化功能
        update=False,  # update all models  对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息
        project='runs/detect',  # save results to project/name  保存地址
        name='exp',  # save results to project/name 保存目录名称
        exist_ok=True,  # existing project/name ok, do not increment   是否覆盖已有项目
        line_thickness=2,  # bounding box thickness (pixels)    预测框边框粗细
        hide_labels=False,  # hide labels   隐藏标签
        hide_conf=False,  # hide confidences    隐藏置信度
        half=False,  # use FP16 half-precision inference    使用FP16半精度推理
        ):

    # Directories   目录
    # increment_path() 增加文件或目录路径
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run 增加运行
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir 创建目录

    # Initialize 初始化
    set_logging()
    device = select_device(device)  # 返回使用的设备 cuda:* 或 cpu
    half &= device.type != 'cpu'  # half precision only supported on CUDA 半精度仅支持CUDA

    # Load model 加载模型
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride 模型步幅
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names (列表: coco 80个不同类型)
    if half:
        model.half()  # to FP16

    # Second-stage classifier   二级分类器
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Dataloader    数据加载器
    view_img = check_imshow()  # bool类型
    cudnn.benchmark = True  # set True to speed up constant image size inference  将True设置为加速常量图像大小推断
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    bs = len(dataset)  # batch_size

    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference 运行推理
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference  推理
        t1 = time_sync()
        pred = model(img,
                     augment=augment,
                     visualize=increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Apply Classifier 应用分类器
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections 过程检测
        for i, det in enumerate(pred):  # detections per image 检测每一个图像
            p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count  # batch_size >= 1

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class (c对应每个种类)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.3f}')

                        # 使用OpenCV在图像 'im0'上绘制一个边界框
                        # xyxy 左上与右下顶点坐标
                        cc = plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        cv2.circle(img=im0, center=cc, radius=2, color=colors(c, True), thickness=5, lineType=cv2.LINE_AA)
                        print("{}[{}]".format(names[c], cc))

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == "__main__":
    run(source=['0', '1'])  # source为列表，内部传入摄像头编号
