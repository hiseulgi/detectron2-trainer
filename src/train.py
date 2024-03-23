import rootutils

root = rootutils.autosetup()

import argparse
import os

import comet_ml
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor, hooks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from dotenv import load_dotenv
from yacs.config import CfgNode

from src.data.data_module import CocoDataModule
from src.models.faster_rcnn import FasterRCNN
from src.schema.trainer_schema import TrainerSchema
from src.trainer.comet_trainer import CometDefaultTrainer, log_image_predictions
from src.utils.config import get_params_cfg_defaults

# Comet ML setup from .env
load_dotenv()
comet_ml.init(
    api_key=os.getenv("COMET_API_KEY"), project_name=os.getenv("COMET_PROJECT_NAME")
)


def main(params_cfg: CfgNode):
    # get comet_ml experiment
    experiment = comet_ml.Experiment()

    # data module setup
    datamodule = CocoDataModule(
        datasets_name=params_cfg.DATAMODULE.DATASETS_NAME,
        images_path=params_cfg.DATAMODULE.IMAGES_PATH,
        train_annotations_path=params_cfg.DATAMODULE.TRAIN_ANNOTATIONS,
        val_annotations_path=params_cfg.DATAMODULE.VAL_ANNOTATIONS,
        test_annotations_path=params_cfg.DATAMODULE.TEST_ANNOTATIONS,
    )

    # get total number of train images
    params_cfg.DATAMODULE.TOTAL_NUM_TRAIN_IMAGES = datamodule.get_total_images(
        split="train"
    )

    # model config setup
    cfg = FasterRCNN(
        datasets_name=params_cfg.DATAMODULE.DATASETS_NAME,
        total_number_images=params_cfg.DATAMODULE.TOTAL_NUM_TRAIN_IMAGES,
        model_config=params_cfg.MODEL_FACTORY.CFG,
        hyperparameters_config=params_cfg.MODEL_FACTORY.HYPERPARAMS,
        total_epochs=params_cfg.MODEL_FACTORY.TOTAL_EPOCHS,
        is_use_epoch=params_cfg.MODEL_FACTORY.IS_USE_EPOCH,
    ).get_cfg()

    # trainer setup
    trainer = CometDefaultTrainer(cfg, experiment)

    # Register Hook to compute metrics using an Evaluator Object
    trainer.register_hooks(
        [
            hooks.EvalHook(
                cfg.TEST.EVAL_PERIOD,
                lambda: trainer.evaluate_metrics(cfg, trainer.model),
            )
        ]
    )

    # Register Hook to compute eval loss
    trainer.register_hooks(
        [
            hooks.EvalHook(
                cfg.TEST.EVAL_PERIOD, lambda: trainer.evaluate_loss(cfg, trainer.model)
            )
        ]
    )

    # print all hooks
    print(
        f"\n======================\nTrainer Hooks: {trainer._hooks}\n======================\n"
    )

    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluation Test Set
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(
        cfg.DATASETS.TEST[0], cfg, False, output_dir=cfg.OUTPUT_DIR
    )
    test_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    test_results = inference_on_dataset(predictor.model, test_loader, evaluator)

    # log metrics and add prefix
    for k, v in test_results.items():
        experiment.log_metrics(v, prefix=f"test-{k}")

    # log image predictions
    log_image_predictions(
        predictor=predictor,
        experiment=experiment,
        metadata=datamodule.get_metadata(split="test"),
        num_images=10,
    )

    # log model
    experiment.log_model("faster_rcnn", cfg.MODEL.WEIGHTS)


if __name__ == "__main__":
    params_cfg = get_params_cfg_defaults()
    params_cfg.merge_from_file("configs/data/default.yaml")
    params_cfg.merge_from_file("configs/model/faster_rcnn.yaml")
    params_cfg.merge_from_file("configs/train.yaml")

    main(params_cfg)
