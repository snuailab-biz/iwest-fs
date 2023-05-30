from matplotlib import pyplot as plt


class dkko_matplotshow:
    

    def __init__(self,show_num = (1,1)):
        
        self.total_num = int(show_num[0])*int(show_num[1]);



        self.fig, self.ax = plt.subplots(int(show_num[0]),int(show_num[1]));
        self.RunFlag = True
        self.is_fistimage=[ True for _ in range(self.total_num)];
        self.my_windows_list = [ None for _ in range(self.total_num)];
        self.ax = self.ax.flatten()
        pass

    def isrun(self):
        return self.RunFlag

    def press(self,event):
        print('press', event.key);
        if event.key == 'q':
            self.RunFlag = False;
    
    def show(self,i,raw_img):
        if(self.is_fistimage[i]):
            self.is_fistimage[i] = False;
            self.my_windows_list[i] = self.ax[i].imshow(raw_img);
            self.fig.canvas.mpl_connect('key_press_event', self.press)
            self.ax[i].axis("off")
            plt.tight_layout(pad=0)
            plt.margins(0,0)
            
        else:
            self.my_windows_list[i].set_data(raw_img);
            plt.show(block = False);
            plt.pause(0.0001);