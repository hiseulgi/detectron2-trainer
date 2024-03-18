from pydantic import BaseModel


class DataModuleSchema(BaseModel):
    datasets_name: str
    images_path: str
    train_annotations_path: str
    val_annotations_path: str
    test_annotations_path: str


class FasterRCNNSchema(BaseModel):
    model_config: str
    hyperparameters_config: str
    total_number_images: int
    total_epochs: int


class TrainerSchema(BaseModel):

    datamodule: DataModuleSchema
    model_config: FasterRCNNSchema
