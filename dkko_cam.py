#WRITER : DKKO
#EMAILE : dkko@snuailab.ai
#DATE   : 20230517
import multiprocessing as mp
from multiprocessing import Lock
from multiprocessing import shared_memory
import numpy as np
import cv2
import time

#multiprocessing camera
class AsyncCamera:
    #cameara_infos :[   [x1,y1,x2,y2,caption,event], 
    #                   [x1,y1,x2,y2,caption,event]..........,
    #                   [x1,y1,x2,y2,caption,event]];

    def __init__(self,url,name=[],cameara_infos:list=[],COLOR = "BGR"):
        self.precessing_lock = Lock();

        self.main_shm = [];#it not using in "__init__" funtion
        self.image_list = [];
        self.gettime_list = [];
        self.diff_box = [];
        self.flag_list = [];

        self.caption_list = [ caption for x1,y1,x2,y2,caption,event in cameara_infos];
        self.cropbox_list = [ [x1,y1,x2,y2] for x1,y1,x2,y2,caption,event in cameara_infos];
        self.event_list   = [ event for x1,y1,x2,y2,caption,event in cameara_infos];
        self.name_list    = [ name_ for name_ in name];

        self.data_num = len(cameara_infos);
        
        image_info_list = [];#다른 프로세서에서 공유메모리를 접근할수 있는 정보들
        diffbox_info_list = [];#다른 프로세서에서 공유메모리를 접근할수 있는 정보들
        gettime_info_list = [];#다른 프로세서에서 공유메모리를 접근할수 있는 정보들

        shared_flag_info_list = [];#다른 프로세서에서 공유메모리를 접근할수 있는 정보들

        full_screen_check_list = [];
        for x1,y1,x2,y2 in self.cropbox_list:
            if((x1==0) and (y1==0) and (x2==0) and (y2==0)):
                full_screen_check_list.append(True);
            else:
                full_screen_check_list.append(False);

        for i,is_fullscreen in enumerate(full_screen_check_list):
            x1,y1,x2,y2 = self.cropbox_list[i]
            #이미지 변수 생성(공유메모리 이용)
            if(is_fullscreen):
                shared_image,shared_image_info = self.generate_shm_data(data_shape=(2000,2000,3),data_type = np.uint8);
                self.image_list.append(shared_image);
                image_info_list.append(shared_image_info);
            else:
                shared_image,shared_image_info = self.generate_shm_data(data_shape=(y2-y1,x2-x1,3),data_type = np.uint8);
                self.image_list.append(shared_image);
                image_info_list.append(shared_image_info);
                
            #gettime 변수 생성
            shared_gettime,shared_gettime_info = self.generate_shm_data(data_shape=(1),data_type = np.uint64);
            self.gettime_list.append(shared_gettime);
            gettime_info_list.append(shared_gettime_info);

            #diff box 변수 생성
            shared_diffbox,shared_diffbox_info = self.generate_shm_data(data_shape=(20,5),data_type = np.int64);
            self.diff_box.append(shared_diffbox);
            diffbox_info_list.append(shared_diffbox_info);

        
            shared_flag, shared_flag_info = self.generate_shm_data(data_shape=(1),data_type = np.int64);
            self.flag_list.append(shared_flag);
            shared_flag_info_list.append(shared_flag_info);
        
        self.shared_killflag, shared_killflag_info = self.generate_shm_data(data_shape=(1),data_type = np.int64);
        self.shared_framesize, shared_framesize_info = self.generate_shm_data(data_shape=(2),data_type = np.int64);
        

        args = {};
        args["precessing_lock"] = self.precessing_lock;
        args["url"] = url;
        args["image_info_list"] = image_info_list;
        args["diffbox_info_list"] = diffbox_info_list;
        args["gettime_info_list"] = gettime_info_list;
        args["shared_flag_info_list"] = shared_flag_info_list;
        args["shared_killflag_info"] = shared_killflag_info;
        args["cropbox_list"] = self.cropbox_list;
        args["shared_framesize_info"] = shared_framesize_info;
        args["full_screen_check_list"] = full_screen_check_list;
        self.full_screen_check_list=full_screen_check_list;
        args["COLOR"] = COLOR;
   
        self.shared_killflag[0]=1;#1 == True, 0 == False;
        self.p = mp.Process(target=self.update, args=(args,));
        #start process
        self.p.daemon = True
        self.p.start()
    
    # 이 "update"는 main과 다른 별도의 프로세스로 할당되어 동작됨.
    def update(self,*args):
        args = args[0];
        sub_shm = [];
        precessing_lock = args["precessing_lock"];
        
        url = args["url"];
        image_info_list = args["image_info_list"];
        diffbox_info_list = args["diffbox_info_list"];
        gettime_info_list = args["gettime_info_list"];
        shared_flag_info_list = args["shared_flag_info_list"];
        shared_killflag_info = args["shared_killflag_info"];
        cropbox_list = args["cropbox_list"];
        shared_framesize_info = args["shared_framesize_info"];
        COLOR = args["COLOR"];
        full_screen_check_list = args["full_screen_check_list"];


        shared_killflag, sharedmemory_buff_killflag = self.get_shm_data(shared_killflag_info)
        sub_shm.append(sharedmemory_buff_killflag);

        shared_framesize, sharedmemory_buff_framesize = self.get_shm_data(shared_framesize_info)
        sub_shm.append(sharedmemory_buff_framesize);

        image_list = [];
        diffbox_list = [];
        gettime_list = [];
        sharedflag_list = [];
        diffdetector_list = [];

        info_num = len(image_info_list);
        for i in range(info_num):
            shm_data_image, sharedmemory_buff_image = self.get_shm_data(image_info_list[i]);
            sub_shm.append(sharedmemory_buff_image);
            image_list.append(shm_data_image);
        
            shm_data_diffbox, sharedmemory_buff_diffbox = self.get_shm_data(diffbox_info_list[i]);
            sub_shm.append(sharedmemory_buff_diffbox);
            diffbox_list.append(shm_data_diffbox);
        
            shm_data_gettime, sharedmemory_buff_gettime = self.get_shm_data(gettime_info_list[i]);
            sub_shm.append(sharedmemory_buff_gettime);
            gettime_list.append(shm_data_gettime);
        
            shared_flag, sharedmemory_buff_flag = self.get_shm_data(shared_flag_info_list[i]);
            sub_shm.append(sharedmemory_buff_flag);
            sharedflag_list.append(shared_flag)

            diffdetector_list.append(diff_dectector());
        
        
        cap = cv2.VideoCapture(url);
        if(cap.isOpened()):
            screen_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));
            screen_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));
            screen_fps = int(cap.get(cv2.CAP_PROP_FPS));
            precessing_lock.acquire();
            shared_framesize[0] = screen_w;
            shared_framesize[1] = screen_h;
            precessing_lock.release();
            print("open camera :"+ str(url), ",  COLOR : ",COLOR)

        

        #이미지 읽기 타임아웃 1초 설정
        #cap.set(cv2.CAP_PROP_XI_TIMEOUT, 1000);
        
        #카메라 오픈 대기
        # while((not cap.isOpened()) & shared_killflag[0].copy()==1):
        #     print("not opened "+url);
        #     time.sleep(2);
        #     cap = cv2.VideoCapture(url);
        diffbox_length = diffbox_list[i].shape[0];
        try:
            while(True):
                killflag = shared_killflag.copy()[0];

                if(killflag == 0):
                    break;
                if((type(url) is int) or (not "rtsp://" in url)):
                    time.sleep(1.0/screen_fps);
                
                ret,frame = cap.read();
                gattime = int(time.time());

                if(ret):
                    if(COLOR == "RGB"):
                        frame = frame[:,:,::-1];
                    
                    for i,(x1,y1,x2,y2) in enumerate(cropbox_list):    
                        if(not full_screen_check_list[i]):
                            frame=frame[y1:y2,x1:x2];
                        
                        
                        diffdetector_list[i].prediction(frame);
                        diffdetector_boxs = diffdetector_list[i].get_box();
                        diffbox = np.zeros_like(diffbox_list[i]);

                        for idx_,(x1,y1,x2,y2) in enumerate(diffdetector_boxs):
                            if(idx_>=diffbox_length):
                                break;
                            diffbox[idx_] = [1,x1,y1,x2,y2];
                            
                        precessing_lock.acquire();
                        np.copyto(diffbox_list[i],diffbox);
                        if(full_screen_check_list[i]):
                            np.copyto(image_list[i][:screen_h,:screen_w],frame);
                        else:
                            np.copyto(image_list[i],frame);
                        gettime_list[i][0] = gattime;
                        sharedflag_list[i][0] = 1;
                        precessing_lock.release();
        except KeyboardInterrupt:
            pass
        cap.release();
        print("close camera :"+ str(url))

                    


    # numpy를 이용한 공유메모리 할당 방법
    def generate_shm_data(self,data_shape = (),data_type = np.uint8):
        #임시 데이터 정의
        temp_data = np.zeros(data_shape,dtype=data_type);
        
        #정의된 데이터 크기만큼 공유메모리 할당 
        sharedmemory_buff = shared_memory.SharedMemory(create=True, size=temp_data.nbytes);
        
        #공유 데이터 생성 -> 공유 메모리와 해당 데이터의 포인트 주소를 링크함.(buffer 변수의 역할)
        shm_data = np.ndarray(temp_data.shape, dtype=temp_data.dtype, buffer = sharedmemory_buff.buf);

        #로컬함수에서 할당받은 공유메모리가 없어지지 않도록 "main_shm" 리스트에 저장
        self.main_shm.append(sharedmemory_buff)

        #생성 정보 리턴
        data_info = [sharedmemory_buff.name,temp_data.dtype,temp_data.shape];
        return shm_data, data_info;

    # numpy를 이용한 공유메모리 데이터 얻기
    def get_shm_data(self,shm_data_info):
        
        sharedmemory_data_name,sharedmemory_data_type,sharedmemory_data_shape = shm_data_info;

        #정의된 데이터 크기만큼 공유메모리 할당(주의 : 반드시 얻은 해당 변수가 가비지 컬렉터에 처리되지 않도록 유의!!!!)
        sharedmemory_buff = shared_memory.SharedMemory(name = sharedmemory_data_name);
        
        #공유 데이터 생성 -> 공유 메모리와 해당 데이터의 포인트 주소를 링크함.(buffer 변수의 역할)
        shm_data = np.ndarray(sharedmemory_data_shape, dtype=sharedmemory_data_type, buffer = sharedmemory_buff.buf);

        return shm_data, sharedmemory_buff;



    def get_data(self, index = 0):
        #return Boolean,[image,gettime,caption,name,event,diff_box,crop_box,originframe_size]
        # shared_flag == 1 is update video
        # shared_flag == 2 isn't update video
        shared_flag = self.flag_list[index].copy();
        caption = self.caption_list[index];
        name = self.name_list[index];
        event = self.event_list[index];
        crop_box = self.cropbox_list[index];

        originframe_w,originframe_h=self.shared_framesize.copy();

        if(shared_flag[0] == 1):
            diff_box = self.verify_diffboxs(self.diff_box[index].copy());#shm
            if(self.full_screen_check_list[index]):
                image = self.image_list[index][:originframe_h,:originframe_w].copy();#shm
            else:    
                image = self.image_list[index].copy();#shm
            
            gettime = self.gettime_list[index].copy();#shm
            
            
            self.precessing_lock.acquire();
            self.flag_list[index][0] = 2;
            self.precessing_lock.release();

            return True,[image,gettime[0],caption,name,event,diff_box,crop_box,[originframe_w,originframe_h]];
    
        return False,[np.zeros_like(self.image_list[index]),0,caption,name,event,[],crop_box,[originframe_w,originframe_h]];
    
    def verify_diffboxs(self,diffboxs:np.array):
        verified_diffboxs = diffboxs[np.where(diffboxs[:,0]==1)];
        return verified_diffboxs[:,1:].tolist();
    

    def sum_crop_boxs(self,box_xyxy,crop_xyxy):
        c_x1,c_y1,c_x2,c_y2=crop_xyxy;
        temp_box = [];
        for b_x1,b_y1,b_x2,b_y2 in box_xyxy:
            temp_box.append([c_x1+b_x1,c_y1+b_y1,c_x2+b_x2,c_y2+b_y2]);
        return temp_box;


    
    def close(self):
        self.precessing_lock.acquire()
        self.shared_killflag[0]=0;#1 == True, 0 == False;
        [shm.unlink()for shm in self.main_shm];
        self.precessing_lock.release();


class diff_dectector:
    def __init__(self,threshold_0_255=60,consecutive=10,update_timing=60):
        self.frame_count = 0;
        self.consecutive_frame = consecutive
        self.background = None;
        self.threshold_0_255 = threshold_0_255;
        self.frame_diff_list = [];
        self.threshold_area = 500;
        self.pre_frame=None;
        self.pre_pre_frame=None;
        self.update_count = 0;
        self.update_timing = update_timing;
        self.save_boxs_list = [];
    
    
    def get_box(self):
        return self.save_boxs_list;


    def prediction(self,img):
        box_list = [];
        frame = img.copy();
        self.frame_count += 1;
        self.update_count +=1;
        #orig_frame = frame.copy()
        orig_frame2 = img.copy()
        # IMPORTANT STEP: convert the frame to grayscale first
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if(self.background is None):
            self.background = gray;
            self.pre_frame = orig_frame2;
        
        if (self.update_count % self.update_timing == 0 or self.update_count == 1):
            self.pre_pre_frame = self.pre_frame
            self.pre_frame = orig_frame2;
            
        if (self.frame_count % self.consecutive_frame == 0) or (self.frame_count == 1):
            self.frame_diff_list = []
            if(not self.pre_pre_frame is None):
                self.background = cv2.cvtColor(self.pre_pre_frame, cv2.COLOR_BGR2GRAY);
                #cv2.imshow("background", self.background);
        
        # find the difference between current frame and base frame
        frame_diff = cv2.absdiff(gray, self.background)
        # thresholding to convert the frame to binary
        ret, thres = cv2.threshold(frame_diff, self.threshold_0_255, 255, cv2.THRESH_BINARY)
        # dilate the frame a bit to get some more white area...
        # ... makes the detection of contours a bit easier
        dilate_frame = cv2.dilate(thres, None, iterations=10)
        # append the final result into the `frame_diff_list`
        self.frame_diff_list.append(dilate_frame)

        
        # if we have reached `consecutive_frame` number of frames
        if len(self.frame_diff_list) == self.consecutive_frame:
            # add all the frames in the `frame_diff_list`
            sum_frames = sum(self.frame_diff_list)

            # find the contours around the white segmented areas
            contours, hierarchy = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # draw the contours, not strictly necessary
            # for i, cnt in enumerate(contours):
            #     cv2.drawContours(frame, contours, i, (0, 0, 255), 3)

            
            for contour in contours:
                # continue through the loop if contour area is less than 500...
                # ... helps in removing noise detection
                if cv2.contourArea(contour) < self.threshold_area:
                    continue

                # get the xmin, ymin, width, and height coordinates from the contours
                (x, y, w, h) = cv2.boundingRect(contour)
                box_list.append([x, y,x+w,y+h]);
                # draw the bounding boxes
                #cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # self.update_count+=1;
            # if(self.update_count % int(self.consecutive_frame*3) == 1):
            #     self.pre_frame = orig_frame2;
            
            self.save_boxs_list = box_list;
            return None;
            #return True,box_list

        return None;
        #return False,box_list;

