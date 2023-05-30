import dkko_utils



class FireSmokeEvent:
    def __init__(self,mode="rise",steady_threshold=3):
        # select mode between "fall" and "rise" and "steady"
        self.mode = mode;
        self.steady_threshold = steady_threshold;

        self.activate = False;
        self.steady_count = 0;
        self.pre_signal = 0;
    
    def check(self,model_box,diff_boxs=None,IOU_threshold=0.05):
        signal = 0;
        if(not diff_boxs is None):
            for diff_box_ in diff_boxs:
                for model_box_ in model_box:
                    if(dkko_utils.IoU(diff_box_,model_box_) > IOU_threshold):
                        signal = 1;
                        break;
        else:
            for model_box_ in model_box:
                 signal = 1;
        
        if(self.mode in "fall"):
            if((self.pre_signal==1) and (signal == 0)):
                self.activate = True;
            else:
                self.activate = False;
        
        elif(self.mode in "rise"):
            if((self.pre_signal==0) and (signal == 1)):
                self.activate = True;
            else:
                self.activate = False;
        
        elif(self.mode in "steady"):
            if(signal == 1):
                self.steady_count += 1;
            else:
                self.steady_count = 0;
                self.activate = False;

            if(self.steady_count>=self.steady_threshold):
                self.activate = True;
        else:
            self.activate = False;
        

        self.pre_signal = signal;
        return self.activate