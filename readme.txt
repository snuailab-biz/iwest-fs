# Writer : DKKO
# EMALE  : dkko@snuailab.ai


# 토치 버전은 시스템마다 다르니 참고
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers==4.11.3
pip install opencv-python

# git에서 다운받거나 압축파일을 풀어 사용할것
git clone https://github.com/mmaaz60/mvits_for_class_agnostic_od.git

# 시스템에 맞추어 빌드할 것
cd mvits_for_class_agnostic_od
cd models/ops
sh make.sh


# ModulatedDetection클래스에 아래 함수 내용을 추가할 것(위치 : mvits_for_class_agnostic_od/inference/modulated_detection.py) -> cv이미지 기반으로 해당 모델의 기능 동작시키기 위해 변경함.




def infer_raw_image(self, raw_image, **kwargs):
    caption = kwargs["caption"]
    # Read the image
    im = Image.fromarray(raw_image);
    
    # import cv2
    # im = cv2.imread(image_path)[:,:,::-1]
    imq = np.array(im)
    if len(imq.shape) != 3:
        im = im.convert('RGB')

    #print(imq.shape)

    img = self.transform(im).unsqueeze(0).cuda()
    # propagate through the models
    memory_cache = self.model(img, [caption], encode_and_save=True)
    outputs = self.model(img, [caption], encode_and_save=False, memory_cache=memory_cache)
    # keep only predictions with self.conf_thresh+ confidence
    probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
    keep = (probas > self.conf_thresh).cpu()
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)
    kept_probs = probas[keep]
    # Convert outputs to the required format
    bboxes = list(bboxes_scaled.numpy())
    probs = list(kept_probs.numpy())
    boxes, scores = [], []
    for b, conf in zip(bboxes, probs):
        boxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
        scores.append(conf)
    # Read image, perform inference, parse results, append the predicted boxes to detections
    return boxes, scores

def infer_raw_image_multi(self, image_list:list, **kwargs):
        #caption = kwargs["caption"]
        captions = kwargs["captions"]
        device = kwargs["device"]
        # Read the image
        image_num = len(image_list)
        crops = [ Image.fromarray(image) for image in image_list]
        if(device in 'cpu'):
            imgs = [self.transform(crop).unsqueeze(0).cpu() for crop in crops]
            imgs = torch.cat(imgs)
            outputs = self.model(imgs, captions)
        else:
            imgs = [self.transform(crop).unsqueeze(0).cuda() for crop in crops]
            imgs = torch.cat(imgs)
            memory_cache = self.model(imgs, [captions[i] for i in range(imgs.shape[0])], encode_and_save=True)
            outputs = self.model(imgs, captions, encode_and_save=False, memory_cache=memory_cache)
        
        # propagate through the models
        #memory_cache = self.model(imgs, [caption for i in range(imgs.shape[0])], encode_and_save=True)
        #outputs = self.model(imgs, [caption], encode_and_save=False, memory_cache=memory_cache)
    
        
        all_result = []
        
        for i in range(len(crops)):
            # keep only predictions with self.conf_thresh+ confidence
            probas = 1 - outputs['pred_logits'].softmax(-1)[i, :, -1].cpu()
            keep = (probas > self.conf_thresh).cpu()
            # convert boxes from [0; 1] to image scales
            bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'].cpu()[i, keep], crops[i].size)
            kept_probs = probas[keep]
            # Convert outputs to the required format
            bboxes = list(bboxes_scaled.numpy())
            probs = list(kept_probs.numpy())
            boxes, scores = [], []
            for b, conf in zip(bboxes, probs):
                boxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
                scores.append(conf)
            # Read image, perform inference, parse results, append the predicted boxes to detections
            
            all_result.append([boxes,scores])


        return all_result

# ModulatedDetection클래스에 아래 내용을 수정할 것(위치 : mvits_for_class_agnostic_od/inference/modulated_detection.py)
class ModulatedDetection(Inference):
    """
    The class supports the inference using both MDETR & MDef-DETR models.
    """
    def __init__(self, model, confidence_thresh=0.0):
        Inference.__init__(self, model)
        self.conf_thresh = confidence_thresh
        self.transform = T.Compose([
            T.Resize((400,400)),#Todo origin =[ T.Resize(800) ]
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])