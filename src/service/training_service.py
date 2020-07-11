"""This file is the pipeline for data etl"""

# import relation package.
import pickle
import pandas as pd

# import project package.
from config.config_setting import ConfigSetting


class TrainingService:
    def __init__(self):
        config_setting = ConfigSetting()
        self.config = config_setting.yaml_parser()
        self.log = config_setting.set_logger(["training_service"])
