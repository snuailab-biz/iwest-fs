from iwestfs.vision import Predict
from iwestfs.utils import get_cfg



CFG = get_cfg('config/camera.yaml')
# run()
a = Predict(CFG)
a.run()
