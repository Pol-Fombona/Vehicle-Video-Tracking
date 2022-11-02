import argparse
import sys
import torch
import cv2
from pathlib import Path
import pandas as pd
from tabulate import tabulate
import time
import os


ROOT = Path(__file__).resolve().parents[0]

sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'yolov5'))
sys.path.append(str(ROOT / 'trackers/ocsort'))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import (check_img_size, non_max_suppression, 
            scale_coords, check_requirements, xyxy2xywh)
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors
from ocsort.ocsort import OCSort

in_count = set()
out_count = set()
in_cls_count = {}
out_cls_count = {}

roi_in = [(90, 750), (240, 750), (90, 900), (240, 900)]
roi_out = [(275, 750), (425, 750), (275, 900), (425, 900)]


def main(classes=None):

    conf_thres=0.5 
    iou_thres=0.5   

    source = 'C:/Users/Admin/Desktop/4t/PSIV/Vehicle-Video-Tracking/yolo/videos/output7_4x.mp4'
    video = source[source.rindex("/")+1:]

    yolo_weights = ROOT / "weights/yolov5s.pt"

    # Load model
    device = select_device('') # 0,1,2... o "cpu"
    model = DetectMultiBackend(yolo_weights, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size((480, 480), stride) 

    # Dataloader & Tracker
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    tracker = OCSort(det_thresh=0.45, iou_threshold=0.2)

    model.warmup(imgsz=(1, 3, *imgsz))  

    classes_names = []

    for item in classes:
        in_cls_count[names[item]] = 0
        out_cls_count[names[item]] = 0
        classes_names.append(names[item])

    start_time = time.time()

    for item in dataset:

        frame, frame_original = item[1], item[2]
        

        frame = torch.from_numpy(frame).to(device)
        frame = frame.float() / 255.0

        if len(frame.shape) == 3:
            frame = frame[None]


        pred = model.forward(frame)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=100)

        # Process detections
        for coords in pred:
            annotator = Annotator(frame_original, line_width=2, pil=not ascii)

            if coords is not None and len(coords):

                coords[:, :4] = scale_coords(frame.shape[2:], coords[:, :4], frame_original.shape).round()
                bb_coords = xyxy2xywh(coords[:, 0:4])
                confs = coords[:, 4]
                clss = coords[:, 5]

                outputs = tracker.update(bb_coords.cpu(), confs.cpu(), clss.cpu(), frame_original)

                if len(outputs) > 0:
                    for output in outputs:
    
                        bboxes = output[0:4]
                        x1, x2 = int(bboxes[0]), int(bboxes[2])
                        y1, y2 = int(bboxes[1]), int(bboxes[3])
                        x = (x1 + x2) // 2
                        y = (y1 + y2) // 2

                        if (x1 > 70 and x2 < 450 and y1 > 320):

                            object_id = int(output[4])
                            cls_id = int(output[5])
                            cls = names[cls_id]

                            if cls in classes_names: 
                                track_car(object_id, cls, x, y)

                            label = str(object_id) + " " + cls
                            annotator.box_label(bboxes, label, color=colors(cls_id, True))
                            frame_original = cv2.circle(frame_original, (x, y), radius=3, color=(255,0,0), thickness=-1)

            frame_original = annotator.result()
            
            cv2.putText(frame_original, "IN: " + str(len(in_count)), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_original, "OUT: " + str(len(out_count)), (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            frame_original = draw_roi(frame_original, 0.2)

            cv2.imshow(str(source), frame_original)
            cv2.waitKey(1) 

    time_final = time.time()
    print("\nSource:", video)
    print("Time Elapsed:", time_final - start_time, "seconds.")
    show_results()



def show_results():

    in_result = pd.DataFrame.from_dict(in_cls_count, orient='index', columns=['Total'])
    out_result = pd.DataFrame.from_dict(out_cls_count, orient='index', columns=['Total'])

    print("\nIN DATA:")
    print(tabulate(in_result, headers="keys", tablefmt="psql"))
    print("\nOUT DATA:")
    print(tabulate(out_result, headers="keys", tablefmt="psql"))



def draw_roi(frame_original, alpha):

    overlay = frame_original.copy()
    cv2.rectangle(overlay, roi_in[0], roi_in[3], (0, 255, 0), -1)
    cv2.rectangle(overlay, roi_out[0], roi_out[3], (0, 0, 255), -1)
    frame_original = cv2.addWeighted(overlay, alpha, frame_original, 1-alpha, 0)

    return frame_original


def track_car(object_id, cls, x, y):

    if (roi_in[0][0] < x < roi_in[1][0]) and (roi_in[0][1] < y < roi_in[2][1]) and (object_id not in in_count):
        in_count.add(object_id)
        in_cls_count[cls] += 1

    elif (roi_out[0][0] < x < roi_out[1][0]) and (roi_out[0][1] < y < roi_out[2][1]) and (object_id not in out_count):
        out_count.add(object_id)
        out_cls_count[cls] += 1
    
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', nargs='+', type=int, default=[1, 2, 3, 5, 7])

    opt = parser.parse_args()

    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    main(**vars(opt))


