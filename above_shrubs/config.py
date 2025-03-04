from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    CNN data configuration class (embedded with OmegaConf).
    """

    # where is this coming from
    experiment_name: Optional[str] = None

    # dinov2_rs, resnet, custom_unet
    backbone_name: Optional[str] = 'dinov2_rs'

    # dinov2_rs_rpt, fcn, unet
    decoder_name: Optional[str] = 'dinov2_rs_rpt'

    version: Optional[str] = '3.0.0'
    main_dir: Optional[str] = 'output'
    metadata_filename: Optional[str] = None

    # training resources
    gpu_devices: Optional[str] = None
    mixed_precision: Optional[str] = None

    # Working directory
    work_dir: Optional[str] = None
    project_name: Optional[str] = None

    # directory that store train data
    train_tiles_dir: Optional[str] = None
    train_data_dir: Optional[str] = None
    train_label_dir: Optional[str] = None

    # directory that stores test data
    test_tiles_dir: Optional[str] = None
    test_data_dir: Optional[str] = None
    test_label_dir: Optional[str] = None

    # number of training images to use
    # sometimes we just want to train with a fixed
    # number of images, -1 implies all images
    # will be used
    num_train_images: Optional[int] = -1

    # seed value
    seed: int = 42

    # do we need to z-score tiles
    standardization: Optional[str] = 'global'

    # tile size - training/testing chip size
    # we get tile_size automatically from first tile - this is
    # relevant for preprocessing
    tile_size: Optional[int] = 256

    # batch size: number of training chips for each iteration within an epoch
    batch_size: Optional[int] = 64

    # Max num epochs for training
    num_epochs: Optional[int] = 6000

    # Only for ViT models (dinov2, etc)
    num_epochs_warmup: Optional[int] = 1

    # Decoder for any torchgeo supported geospatial foundation model
    # https://torchgeo.readthedocs.io/en/stable/api/models.html
    decoder: Optional[str] = 'fcn'

    # The amount you apply to the optimizer to adjust weights;
    # how the step size will affect model performance ; put high
    # then go lower as you fine-tune
    # the lower, the slower the training
    learning_rate: Optional[float] = 0.1

    # Scheduler for lowering learning rate
    lr_scheduler: Optional[str] = None

    # Model improvement epochs - if model doesnt improve after
    #  this value, then stop training
    patience: Optional[int] = 10

    # resnet50 pre-trained weights
    weights: Optional[str] = 'ResNet50_Weights.LANDSAT_TM_TOA_SIMCLR'

    # Callbacks function expression, expects list of metrics
    callbacks: List[str] = field(
        default_factory=lambda: ["pl.pytorch.callbacks.ModelCheckpoint"]
    )

    # Logger function expression, expects a string with the function
    loggers: List[str] = field(
        default_factory=lambda: [
            "pl.pytorch.loggers.TensorBoardLogger" +
            "(save_dir='output', name='tensorboard_logs')"
        ]
    )

    # fast_dev_run if we want a quick run
    fast_dev_run: Optional[bool] = False

    # These are hardcoded for now based on our 4-band VHR SR merged tiles
    # set from ifsar + lidar training used for 20231014 in mjm dir
    # TODO: get this on the fly or read in directly somehow
    preprocessing_mean_vector: list = field(
        default_factory=lambda: [
            368.7239, 547.4674, 528.48615, 2144.9368])

    preprocessing_std_vector: list = field(
        default_factory=lambda: [
            115.4657, 157.63426, 231.98418, 1246.9503])

    input_bands: list = field(
        default_factory=lambda: [
            "Blue", "Green", "Red", "NIR1", "HOM1", "HOM2"])
    output_bands: list = field(
        default_factory=lambda: [
            "Blue", "Green", "Red", "NIR1", "HOM1", "HOM2"])

    # filenames storing DTM and DSM
    dtm_path: Optional[str] = None
    dsm_path: Optional[str] = None

    inference_regex_list: Optional[List[str]] = field(
        default_factory=lambda: [])
    inference_save_dir: str = "results"
    experiment_type: Optional[str] = 'output'
    model_dir: Optional[str] = 'output'

    # model filename in case you want to specify one
    model_filename: Optional[str] = None

    # filenames storing cloud mask
    cloudmask_path: Optional[str] = None

    # Specify dtype of prediction
    prediction_dtype: Optional[str] = 'float32'

    # Specify no-data value for prediction
    prediction_nodata: Optional[int] = 255

    # Specify compression for prediction
    prediction_compress: Optional[str] = 'LZW'

    # Specify driver for prediction (COG and ZARR included)
    prediction_driver: Optional[str] = 'GTiff'


@dataclass
class CHMConfig(Config):

    # filenames storing DTM and DSM
    dtm_path: Optional[str] = None
    dsm_path: Optional[str] = None

    # filenames storing cloud mask
    cloudmask_path: Optional[str] = None

    input_bands: list = field(
        default_factory=lambda: [
            "Blue", "Green", "Red", "NIR1", "HOM1", "HOM2"])
    output_bands: list = field(
        default_factory=lambda: [
            "Blue", "Green", "Red", "NIR1", "HOM1", "HOM2"])

    # loss function
    loss_func: Optional[str] = 'mse'  # 'mae'


@dataclass
class LandCoverConfig(Config):

    # directory that store train data
    train_data_dir: Optional[str] = None
    train_label_dir: Optional[str] = None

    # directory that stores test data
    test_dir: Optional[str] = None
    test_data_dir: Optional[str] = None
    test_label_dir: Optional[str] = None

    # filenames storing DTM and DSM
    dtm_path: Optional[str] = None
    dsm_path: Optional[str] = None

    # filenames storing cloud mask
    cloudmask_path: Optional[str] = None

    input_bands: list = field(
        default_factory=lambda: [
            "Blue", "Green", "Red", "NIR1", "HOM1", "HOM2"])
    output_bands: list = field(
        default_factory=lambda: [
            "Blue", "Green", "Red", "NIR1", "HOM1", "HOM2"])

    # loss function
    loss_func: Optional[str] = 'dice'
