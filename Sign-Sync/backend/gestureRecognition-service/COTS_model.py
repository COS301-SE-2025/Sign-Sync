import sys
import torchvision.transforms.functional as F
# make “import torchvision.transforms.functional_tensor” point at torchvision.transforms.functional
sys.modules['torchvision.transforms.functional_tensor'] = F
import omegaconf
from openhands.apis.inference import InferenceModel

cfg = omegaconf.OmegaConf.load("config/wlasl_st_gcn.yaml")
model = InferenceModel(cfg=cfg)
model.init_from_checkpoint_if_available()
if cfg.data.test_pipeline.dataset.inference_mode:
    model.test_inference()
else:
    model.compute_test_accuracy()