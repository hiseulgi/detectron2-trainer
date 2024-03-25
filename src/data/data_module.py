import rootutils

root = rootutils.autosetup()

import os
from typing import Optional

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances


class CocoDataModule:

    def __init__(
        self,
        datasets_name: str,
        images_path: str,
        train_annotations_path: str,
        val_annotations_path: Optional[str] = None,
        test_annotations_path: Optional[str] = None,
    ) -> None:
        self.datasets_name = datasets_name
        self.images_path = images_path
        self.train_annotations_path = train_annotations_path
        self.val_annotations_path = val_annotations_path
        self.test_annotations_path = test_annotations_path

        self._register_datasets()

    def _register_datasets(self) -> None:
        register_coco_instances(
            f"{self.datasets_name}_train",
            {},
            self.train_annotations_path,
            self.images_path,
        )

        if self.val_annotations_path:
            register_coco_instances(
                f"{self.datasets_name}_val",
                {},
                self.val_annotations_path,
                self.images_path,
            )

        if self.test_annotations_path:
            register_coco_instances(
                f"{self.datasets_name}_test",
                {},
                self.test_annotations_path,
                self.images_path,
            )

    def get_metadata(self, split: str = "train") -> any:
        try:
            if split in ["train", "val", "test"]:
                return MetadataCatalog.get(f"{self.datasets_name}_{split}")
        except KeyError:
            raise KeyError(f"Metadata for {self.datasets_name}_{split} not found")

    def get_total_images(self, split: str = "train") -> int:
        images_dir = self.get_metadata(split).image_root
        return len(os.listdir(images_dir))


# debug
if __name__ == "__main__":
    coco_data_module = CocoDataModule(
        datasets_name="chv_dataset",
        images_path="../datasets/chv_dataset_coco/images",
        train_annotations_path="../datasets/chv_dataset_coco/annotations/instances_default.json",
        val_annotations_path="../datasets/chv_dataset_coco/annotations/instances_default.json",
        test_annotations_path="../datasets/chv_dataset_coco/annotations/instances_default.json",
    )

    print(coco_data_module.get_metadata("train"))
    print(coco_data_module.get_total_images("train"))
    print(coco_data_module.get_metadata("val"))
    print(coco_data_module.get_total_images("val"))
    print(coco_data_module.get_metadata("test"))
    print(coco_data_module.get_total_images("test"))
