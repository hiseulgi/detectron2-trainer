import rootutils

root = rootutils.autosetup()

import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor, hooks


class FasterRCNN:

    # note: default values according to chv datasets
    def __init__(
        self,
        model_config: str = "default",
        hyperparameters_config: str = "configs/hyperparameters.yaml",
        total_epochs: int = 20,
        total_number_images: int = 1330,
        datasets_name: str = "chv_dataset",
        is_use_epoch: bool = False,
    ):

        self.model_config = model_config
        self.hyperparameters_config = hyperparameters_config
        self.total_epochs = total_epochs
        self.total_number_images = total_number_images
        self.datasets_name = datasets_name
        self.is_use_epoch = is_use_epoch

        if model_config == "default":
            self.model_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        else:
            self.model_config = model_config

        self.cfg = None

        self._setup()

    def _setup(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.model_config))
        cfg.merge_from_file(self.hyperparameters_config)
        cfg.DATASETS.TRAIN = (f"{self.datasets_name}_train",)
        cfg.DATASETS.TEST = (f"{self.datasets_name}_val",)

        # calculate the number of iterations
        if self.is_use_epoch:
            cfg.SOLVER.MAX_ITER = (
                self.total_epochs * self.total_number_images
            ) // cfg.SOLVER.IMS_PER_BATCH

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        self.cfg = cfg

    def get_cfg(self):
        return self.cfg
