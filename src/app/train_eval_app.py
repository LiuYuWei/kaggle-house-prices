"""This file is the pipeline for data etl"""

# import relation package.

# import project package.
from config.config_setting import ConfigSetting
from src.service.train_eval_service import TrainEvalService

class TrainEvalApp:
    def __init__(self):
        config_setting = ConfigSetting()
        self.config = config_setting.yaml_parser()
        self.log = config_setting.set_logger(["data_etl_app"])
        self.train_eval_service = TrainEvalService()
    
    def start(self, model='lr'):
        self.read_data()
        self.train_test_split()
        self.training_model(model)
        self.evaluate()
        self.finish(model)

    def read_data(self):
        self.train_eval_service.read_preprocess_data()
        self.train_eval_service.read_preprocess_label()
    
    def train_test_split(self):
        self.train_eval_service.train_test_split()
    
    def training_model(self, model):
        if model == 'lr':
            self.train_eval_service.linear_regression_training()
        elif model == 'gbr':
            self.train_eval_service.gbr_training()

    def evaluate(self):
        self.train_eval_service.evaluate()

    def finish(self, model):
        self.train_eval_service.model_save(model)
