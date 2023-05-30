#WRITER : DKKO
#EMAILE : dkko@snuailab.ai
#DATE   : 20230517
import sys
sys.path.append("./mvits_for_class_agnostic_od")
from mvits_for_class_agnostic_od.models.model import Model
import torch
import time
from dkko_utils import *
import iwest_SUNAI_restapi
import yaml
import dkko_show

global mainrunflage

model_name = "mdef_detr"
checkpoints_path = "MDef_DETR_r101_epoch20_ore.pth"
model_batch_size = 1;
is_show = True;
video_src = "/home/dkko/Desktop/python_ws/Smoke_230412_iwest_All.mp4"
RESTAPI_URL = "http://0.0.0.0:8080/dkserver";
CAMERAINFO_PATH = "./config";
camera_info_file_list = [];
DEVICE = 'cuda:1'


if __name__ == "__main__":
    
    yaml_file = None;
    with open("iwest_config.yaml") as f:
        yaml_file = yaml.load(f, Loader=yaml.FullLoader)
    
    if(not yaml_file is None):
        model_name = yaml_file["MODEL_NAME"];
        checkpoints_path = yaml_file["PRETRAINED_MODEL"];
        model_batch_size = yaml_file["BATCH_SIZE"];
        is_show = yaml_file["IMAGE_SHOW"];
        RESTAPI_URL = yaml_file["RESTAPI_URL"];
        CAMERAINFO_PATH = yaml_file["CAMERAINFO_PATH"];
        DEVICE = yaml_file["DEVICE"];
    
    device = torch.device(DEVICE)
    camera_list,device_list,eventgen_list = Convert_JsonToCamera(CAMERAINFO_PATH)
    model = Model(model_name, checkpoints_path).get_model()
    model.model.to(device);
    device_list_size = len(device_list)
    head_count = 0;
    sum_time = 0.0;
    time_count = 0;
    s = time.time();

    mainrunflage = True; 
    show_n = int(np.sqrt(len(device_list)));
    show_m = len(device_list) - show_n;
    my_show = dkko_show.dkko_matplotshow(show_num=(show_n+1,show_m));
    
    try:
        while(mainrunflage & my_show.isrun()):
            frame_list = [];
            captions_list = [];
            event_list = [];
            diff_box_list = [];
            name_list = [];
            originframe_list = [];
            crop_box_list = [];
            gettime_list = [];
            result = True;

            for i in range(model_batch_size):
                device,device_idx= device_list[int(head_count%device_list_size)];
                 
                ret,(image,gettime,caption,name,event,diff_box,crop_box,originframe_size) = device.get_data(device_idx);

                gettime_list.append(gettime);
                originframe_list.append(originframe_size);
                frame_list.append(image);
                captions_list.append(caption);
                event_list.append(event)
                diff_box_list.append(diff_box)
                name_list.append(name);
                crop_box_list.append(crop_box);
                head_count += 1;
                result = ret

            if(result):
                model_results_list = model.infer_raw_image_multi(frame_list, captions=captions_list,device=DEVICE);
                
                for i,model_results in enumerate(model_results_list):
                    model_box,model_score=get_bound_box(frame_list[i],model_results,0.9);
                    #show
                    model_box_img=draw_bound_box_score(frame_list[i],model_box,model_score);

                    if(is_show):
                        #cv2.imshow(event_list[i],model_box_img[:,:,::-1]);
                        my_show.show(i,model_box_img)
                        #cv2.imshow(event_list[i]+"_diffbox",draw_bound_box(frame_list[i],diff_box_list[i])[:,:,::-1]);
                        event_active=eventgen_list[i].check(model_box,diff_box_list[i],IOU_threshold=0.01);
           

                        if(event_active):
                            print(event_list[i]+", caption : "+captions_list[i],", mode :"+eventgen_list[i].mode);
                            #if("flame" == event_list[i]):
                            nxywh=convert_crop_xyxy2nxywh(model_box[0],crop_box_list[i],originframe_list[i]);
                            if("flame" in event_list[i]):
                                iwest_SUNAI_restapi.send_msg_flame(RESTAPI_URL,nxywh,model_score[0],gettime_list[i],name_list[i]);
                            elif("smoke" in event_list[i]):
                                iwest_SUNAI_restapi.send_msg_smoke(RESTAPI_URL,nxywh,model_score[0],gettime_list[i],name_list[i]);
                if(is_show):
                    key=cv2.waitKey(1);
                    if(key&0xFF==ord('q')):
                        break;
            else:
                time.sleep(0.001);
    except KeyboardInterrupt:
        print('Ctrl + C 중지')
    if(is_show):
        cv2.destroyAllWindows();
    [camera.close() for camera in camera_list];
    time.sleep(1);

