from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    CNN data configuration class (embedded with OmegaConf).
    """

    # Directory to store model artifacts
    # model_dir: Optional[str]

    # String with model function (e.g. tensorflow object)
    # model: Optional[str]

    # Working directory
    work_dir: Optional[str] = None

    # directory that store train data
    train_tiles_dir: Optional[str] = None
    train_data_dir: Optional[str] = None
    train_label_dir: Optional[str] = None

    # directory that stores test data
    test_tiles_dir: Optional[str] = None
    test_data_dir: Optional[str] = None
    test_label_dir: Optional[str] = None

    # seed value
    seed: int = 42

    # do we need to z-score tiles
    standardization: Optional[bool] = False

    # tile size
    tile_size: int = 256

    # batch size
    batch_size: int = 64

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
