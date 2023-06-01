
import torch
import time

import numpy as np
import cv2

from iwestfs.models.model import Model

from iwestfs.utils import dkko_matplotshow
from iwestfs.utils import send_msg_smoke, send_msg_flame
from iwestfs.utils import AsyncCamera, FireSmokeEvent
from iwestfs.utils import convert_crop_xyxy2nxywh, get_bound_box, draw_bound_box_score

class Predict:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config.device)
        self.cams = []
        self.eventer = []
        self.bs = self.config.batch_size
        self.show = self.config.image_show
        self.cam_load()
        self.model = Model(self.config.model_name, self.config.pretrained_model).get_model()
        self.model.model.to(self.device)
        self.head_count = 0
        self.restapi = self.config.restapi_url
    
    def cam_load(self):
        for cam in self.config.cam:
            url = cam["url"]
            color = cam["color"]
            ROIs = cam["rois"]
            camera_infos_list = []
            for ROI in ROIs:
                x1 = ROI["x1"]
                y1 = ROI["y1"]
                x2 = ROI["x2"]
                y2 = ROI["y2"]
                name = ROI["name"]
                caption = ROI["caption"]
                event = ROI["event"]
                
                camera_infos_list.append([name, x1, y1, x2, y2, caption,event])
            
            
            roi_num = len(ROIs);
            if(not roi_num == 0):
                camera = AsyncCamera(url = url,
                                              camera_infos = camera_infos_list,
                                              COLOR = color)
                for i in range(roi_num):
                    self.cams.append([camera, i]);
                    self.eventer.append(FireSmokeEvent())

    def run(self):
        device_list_size = len(self.cams)
        show_n = int(np.sqrt(len(self.cams)));
        show_m = len(self.cams) - show_n;
        # my_show = dkko_matplotshow(show_num=(show_n+1,show_m));
        try:
            while True:
                frame_list = [];
                captions_list = [];
                event_list = [];
                diff_box_list = [];
                name_list = [];
                originframe_list = [];
                crop_box_list = [];
                gettime_list = [];
                result = True;

                for i in range(self.bs):
                    cam, cam_idx = self.cams[int(self.head_count%device_list_size)];
                    
                    ret, (image, gettime, caption, name, event, diff_box, crop_box, originframe_size) = cam.get_data(cam_idx);

                    gettime_list.append(gettime);
                    originframe_list.append(originframe_size);
                    frame_list.append(image);
                    captions_list.append(caption);
                    event_list.append(event)
                    diff_box_list.append(diff_box)
                    name_list.append(name);
                    crop_box_list.append(crop_box);
                    self.head_count += 1;
                    result = ret

                if(result):
                    model_results_list = self.model.infer_raw_image_multi(frame_list, captions=captions_list, device=self.config.device);
                    
                    for i,model_results in enumerate(model_results_list):
                        model_box,model_score=get_bound_box(frame_list[i],model_results,0.9);
                        #show
                        model_box_img=draw_bound_box_score(frame_list[i],model_box,model_score);

                        if(self.show):
                            cv2.imshow(event_list[i],model_box_img[:,:,::-1]);
                            # my_show.show(i,model_box_img)
                            #cv2.imshow(event_list[i]+"_diffbox",draw_bound_box(frame_list[i],diff_box_list[i])[:,:,::-1]);
                            event_active=self.eventer[i].check(model_box,diff_box_list[i],IOU_threshold=0.01);
            

                            if(event_active):
                                print(event_list[i]+", caption : "+captions_list[i],", mode :"+self.eventer[i].mode);
                                # if("flame" == event_list[i]):
                                nxywh=convert_crop_xyxy2nxywh(model_box[0],crop_box_list[i],originframe_list[i]);
                                if("flame" in event_list[i]):
                                    send_msg_flame(self.restapi_url, nxywh,model_score[0],gettime_list[i],name_list[i]);
                                elif("smoke" in event_list[i]):
                                    send_msg_smoke(self.restapi_url, nxywh,model_score[0],gettime_list[i],name_list[i]);
                    if(self.show):
                        key=cv2.waitKey(1);
                        if(key&0xFF==ord('q')):
                            break;
                else:
                    time.sleep(0.001);
        except KeyboardInterrupt:
            print('Ctrl + C 중지')
        if(self.show):
            cv2.destroyAllWindows();
        [camera.close() for camera, _ in self.cams];
        time.sleep(1);


