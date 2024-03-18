# my_project/config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.DATAMODULE = CN()
_C.DATAMODULE.DATASETS_NAME = "chv_dataset"
_C.DATAMODULE.IMAGES_PATH = "data/chv_dataset/images"
_C.DATAMODULE.TRAIN_ANNOTATIONS = "data/chv_dataset/annotations/train.json"
_C.DATAMODULE.VAL_ANNOTATIONS = "data/chv_dataset/annotations/val.json"
_C.DATAMODULE.TEST_ANNOTATIONS = "data/chv_dataset/annotations/test.json"

_C.MODEL_FACTORY = CN()
_C.MODEL_FACTORY.CFG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
_C.MODEL_FACTORY.HYPERPARAMS = "configs/experiment/faster_rcnn_base.yaml"
_C.MODEL_FACTORY.TOTAL_EPOCHS = 20


def get_params_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
