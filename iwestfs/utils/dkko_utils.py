#WRITER : DKKO
#EMAILE : dkko@snuailab.ai
#DATE   : 20230517
import cv2
import numpy as np

def convert_crop_xyxy2nxywh(bbox,crop,frame_size):
    frame_w,frame_h=frame_size;
    bbox_x1,bbox_y1,bbox_x2,bbox_y2 = bbox;
    crop_x1,crop_y1,_,_ = crop;

    screen_x1 = bbox_x1+crop_x1;
    screen_y1 = bbox_y1+crop_y1;
    screen_x2 = bbox_x2+crop_x1;
    screen_y2 = bbox_y2+crop_y1;


    screen_cx = (screen_x2+screen_x1)/2;
    screen_cy = (screen_y2+screen_y1)/2;
    screen_w = (screen_x2-screen_x1);
    screen_h = (screen_y2-screen_y1);

    screen_ncx = screen_cx/frame_w;
    screen_ncy = screen_cy/frame_h;
    screen_nw  = screen_w/frame_w;
    screen_nh  = screen_h/frame_h;
    return [screen_ncx,screen_ncy,screen_nw,screen_nh];



def get_bound_box(img,model_result,score_threshold=0.9, area_threshold=0.35):#TODO
    boxs,conf=model_result
    conf=np.array(conf)
    boxs=np.array(boxs)
    box_list=boxs[conf>score_threshold]
    conf_list=conf[conf>score_threshold]
    img_h,img_w  = img.shape[:2];
    
    boxs_list = [];
    scores_list = [];
    for box_,conf_ in zip(box_list,conf_list):
        x1, y1, x2, y2 = box_
        box_area = (x2-x1)*(y2-y1)/(img_w*img_h);
        if(box_area < area_threshold):
            boxs_list.append([x1, y1, x2, y2]);
            scores_list.append(conf_);
    return boxs_list,scores_list

def draw_bound_box_score(img_,boxs,scores,color=(0,0,255)):#TODO
    img = img_.copy();
    box_list = [];
    for box_,conf_ in zip(boxs,scores):
        x1, y1, x2, y2 = box_
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        cv2.rectangle(img,(x1,y1),(x2,y2),color,3);
        text="score: "+str(conf_);
        cv2.putText(img,text,(x1, y1),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
    return img

def draw_bound_box(img_,boxs,color=(0,255,0)):#TODO
    img = img_.copy();
    box_list = [];
    for box_ in boxs:
        x1, y1, x2, y2 = box_
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        cv2.rectangle(img,(x1,y1),(x2,y2),color,3);
    return img

def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

