import multiprocessing as mp
from multiprocessing import Lock
from multiprocessing import shared_memory
import numpy as np
import cv2
import time


#multiprocessing camera
class NomalAsyncCamera:
    def __init__(self,url,name=[],screen_size=(1000,1000),COLOR = "BGR"):

        self.precessing_lock = Lock();
        self.main_shm = [];#it not using in "__init__" funtion
        self.image_list = [];
        self.gettime_list = [];
        self.flag_list = [];

        self.name_list    = [ name ];
        
        image_info_list = [];#다른 프로세서에서 공유메모리를 접근할수 있는 정보들
        gettime_info_list = [];#다른 프로세서에서 공유메모리를 접근할수 있는 정보들
        shared_flag_info_list = [];#다른 프로세서에서 공유메모리를 접근할수 있는 정보들
        self.screen_size = screen_size;

      
        #이미지 변수 생성(공유메모리 이용)
        shared_image,shared_image_info = self.generate_shm_data(data_shape=(screen_size[1],screen_size[0],3),data_type = np.uint8);
        self.image_list.append(shared_image);
        image_info_list.append(shared_image_info);

        #gettime 변수 생성
        shared_gettime,shared_gettime_info = self.generate_shm_data(data_shape=(1),data_type = np.uint64);
        self.gettime_list.append(shared_gettime);
        gettime_info_list.append(shared_gettime_info);
    
        shared_flag, shared_flag_info = self.generate_shm_data(data_shape=(1),data_type = np.int64);
        self.flag_list.append(shared_flag);
        shared_flag_info_list.append(shared_flag_info);
        
        self.shared_killflag, shared_killflag_info = self.generate_shm_data(data_shape=(1),data_type = np.int64);
        
        args = {};
        args["precessing_lock"] = self.precessing_lock;
        args["url"] = url;
        args["image_info_list"] = image_info_list;
        args["gettime_info_list"] = gettime_info_list;
        args["shared_flag_info_list"] = shared_flag_info_list;
        args["shared_killflag_info"] = shared_killflag_info;
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
        gettime_info_list = args["gettime_info_list"];
        shared_flag_info_list = args["shared_flag_info_list"];
        shared_killflag_info = args["shared_killflag_info"];
        COLOR = args["COLOR"];


        shared_killflag, sharedmemory_buff_killflag = self.get_shm_data(shared_killflag_info)
        sub_shm.append(sharedmemory_buff_killflag);


        image_list = [];
        gettime_list = [];
        sharedflag_list = [];

        info_num = len(image_info_list);
        for i in range(info_num):
            shm_data_image, sharedmemory_buff_image = self.get_shm_data(image_info_list[i]);
            sub_shm.append(sharedmemory_buff_image);
            image_list.append(shm_data_image);
        
            shm_data_gettime, sharedmemory_buff_gettime = self.get_shm_data(gettime_info_list[i]);
            sub_shm.append(sharedmemory_buff_gettime);
            gettime_list.append(shm_data_gettime);
        
            shared_flag, sharedmemory_buff_flag = self.get_shm_data(shared_flag_info_list[i]);
            sub_shm.append(sharedmemory_buff_flag);
            sharedflag_list.append(shared_flag)
        
        cap = cv2.VideoCapture(url);

        if(cap.isOpened()):
            screen_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));
            screen_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));
            print("open camera :"+ str(url));
        
        #이미지 읽기 타임아웃 1초 설정
        #cap.set(cv2.CAP_PROP_XI_TIMEOUT, 1000);
        
        #카메라 오픈 대기
        # while((not cap.isOpened()) & shared_killflag[0].copy()==1):
        #     print("not opened "+url);
        #     time.sleep(2);
        #     cap = cv2.VideoCapture(url);

        while(True):
            killflag = shared_killflag[0].copy();
            if(killflag == 0):
                break;
            ret,frame = cap.read();
            gattime = int(time.time());


            if(ret):
                if(COLOR == "RGB"):
                    frame = frame[:,:,::-1];
                precessing_lock.acquire();
                np.copyto(image_list[i],frame);
                gettime_list[i][0] = gattime;
                sharedflag_list[i][0] = 1;
                precessing_lock.release();
        
        print("close camera :"+ str(url))
        cap.release();

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
        #return Boolean,[image,gettime,name,originframe_size]
        # shared_flag == 1 is update video
        # shared_flag == 2 isn't update video
        shared_flag = self.flag_list[index].copy();
        name = self.name_list[0];

        if(shared_flag[0] == 1):
            image = self.image_list[index].copy();#shm
            gettime = self.gettime_list[index].copy();#shm
            
            
            self.precessing_lock.acquire();
            self.flag_list[index][0] = 2;
            self.precessing_lock.release();

            return True,[image,gettime[0],name,self.screen_size];
    
        return False,[np.zeros_like(self.image_list[index]),0,name,self.screen_size];
    
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
        self.shared_killflag[0] = 0;#1 == True, 0 == False;
        self.precessing_lock.release();
        time.sleep(0.1);
        [shm.unlink()for shm in self.main_shm];
        


cap=NomalAsyncCamera(0,screen_size=(640,480));
while(True):
    ret,data_info=cap.get_data(0)
    if(ret):
        frame=data_info[0]
        cv2.imshow("frame",frame);
        key=cv2.waitKey(33);
        if(key&0xFF==ord('q')):
            break
    else:
        time.sleep(0.001);

cap.close();
time.sleep(1);