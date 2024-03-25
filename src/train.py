import rootutils

root = rootutils.autosetup()

import argparse
import os

import comet_ml
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor, hooks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from dotenv import load_dotenv
from yacs.config import CfgNode

from src.data.data_module import CocoDataModule
from src.models.faster_rcnn import FasterRCNN
from src.trainer.comet_trainer import CometDefaultTrainer, log_image_predictions
from src.utils.config import get_params_cfg_defaults

load_dotenv()


def main(params_cfg: CfgNode):
    # ====================
    # COMET ML SETUP
    # get comet_ml experiment
    comet_ml.init(
        api_key=os.getenv("COMET_API_KEY"), project_name=os.getenv("COMET_PROJECT_NAME")
    )
    experiment = comet_ml.Experiment()

    # ====================
    # DATA SETUP
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

    # ====================
    # MODEL SETUP
    # model config setup
    cfg = FasterRCNN(
        datasets_name=params_cfg.DATAMODULE.DATASETS_NAME,
        total_number_images=params_cfg.DATAMODULE.TOTAL_NUM_TRAIN_IMAGES,
        model_config=params_cfg.MODEL_FACTORY.CFG,
        hyperparameters_config=params_cfg.MODEL_FACTORY.HYPERPARAMS,
        total_epochs=params_cfg.MODEL_FACTORY.TOTAL_EPOCHS,
        is_use_epoch=params_cfg.MODEL_FACTORY.IS_USE_EPOCH,
    ).get_cfg()

    # ====================
    # TRAINER SETUP
    # trainer setup
    trainer = CometDefaultTrainer(cfg, experiment)

    # calculate eval period based on total epochs
    if params_cfg.MODEL_FACTORY.IS_USE_EPOCH:
        cfg.TEST.EVAL_PERIOD = (
            cfg.SOLVER.MAX_ITER // params_cfg.MODEL_FACTORY.TOTAL_EPOCHS
        )

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

    # ====================
    # TRAINING
    # Train Model
    print("\n\n====================")
    print("Start Training")

    trainer.resume_or_load(resume=False)
    trainer.train()

    print("\n\nEnd Training")
    print("====================\n\n")

    # ====================
    # EVALUATION
    # Evaluation Test Set
    print("\n\n====================")
    print("Start Evaluation")

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.DATASETS.TEST = params_cfg.DATAMODULE.DATASETS_NAME + "_test"

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(cfg.DATASETS.TEST, cfg, False, output_dir=cfg.OUTPUT_DIR)

    test_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST)
    test_results = inference_on_dataset(trainer.model, test_loader, evaluator)

    # log metrics and add prefix
    for k, v in test_results.items():
        print(f"test/{k}: {v}")
        experiment.log_metrics(v, prefix=f"test/{k}")

    # log image predictions
    log_image_predictions(
        predictor=predictor,
        experiment=experiment,
        metadata=datamodule.get_metadata(split="test"),
        num_images=10,
    )

    print("\n\nEnd Evaluation")
    print("====================\n\n")

    # log model
    print("\n\n====================")
    print("Log Model")

    experiment.log_model("faster_rcnn", cfg.MODEL.WEIGHTS)

    print("\n\nEnd Log Model")
    print("====================\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program description")
    parser.add_argument(
        "--data_config",
        default="configs/data/default.yaml",
        help="Path to data configuration file",
    )
    parser.add_argument(
        "--model_config",
        default="configs/model/faster_rcnn.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--train_config",
        default="configs/train.yaml",
        help="Path to training configuration file",
    )

    args = parser.parse_args()

    params_cfg = get_params_cfg_defaults()
    params_cfg.merge_from_file(args.data_config)
    params_cfg.merge_from_file(args.model_config)
    params_cfg.merge_from_file(args.train_config)

    main(params_cfg)
