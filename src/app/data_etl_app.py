"""This file is the pipeline for data etl"""

# import relation package.

# import project package.
from config.config_setting import ConfigSetting
from src.service.data_etl_service import DataEtlService

class DataEtlApp:
    def __init__(self):
        config_setting = ConfigSetting()
        self.config = config_setting.yaml_parser()
        self.log = config_setting.set_logger(["data_etl_app"])
        self.data_etl_service = DataEtlService()
    
    def start(self):
        self.extract()
        self.transform()
        self.load_to()
        self.finish()

    def extract(self):
        self.data_etl_service.load_data()

    def transform(self):
        self.data_etl_service.remove_feature()
        self.data_etl_service.feature_selection()

    def load_to(self):
        self.data_etl_service.save_dataframe()
        self.data_etl_service.save_label()

    def finish(self):
        pass
