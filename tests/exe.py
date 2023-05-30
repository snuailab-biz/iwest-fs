from iwestfs.vision import Predict
from iwestfs.utils import get_cfg

CFG = get_cfg('/home/ljj/workspace/iwest-fs/dockerin/camera.yaml')
# run()
a = Predict(CFG)
a.run()
