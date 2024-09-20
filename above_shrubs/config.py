from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    """
    CNN data configuration class (embedded with OmegaConf).
    """

    # Directory to store model artifacts
    #model_dir: Optional[str]

    # String with model function (e.g. tensorflow object)
    #model: Optional[str]

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


@dataclass
class CHMConfig(Config):

    # filenames storing DTM and DSM
    dtm_path: Optional[str] = None
    dsm_path: Optional[str] = None

    # filenames storing cloud mask
    cloudmask_path: Optional[str] = None


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
