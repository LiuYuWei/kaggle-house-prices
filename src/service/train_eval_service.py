"""This file is the pipeline for data etl"""

# import relation package.
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR


# import project package.
from config.config_setting import ConfigSetting


class TrainEvalService:
    def __init__(self):
        config_setting = ConfigSetting()
        self.config = config_setting.yaml_parser()
        self.log = config_setting.set_logger(["training_service"])
        self.df = {}
        self.label = []
        self.data = {}
        self.model = None

    def read_preprocess_data(self, save_file_path=None):
        if save_file_path is None:
            save_file_path = self.config['load_to']['save_file_path']
        self.df['train'] = pd.read_csv("{}/{}".format(save_file_path, 'train.csv'))
        self.df['test'] = pd.read_csv("{}/{}".format(save_file_path, 'test.csv'))
        self.log.info('Successfully read training and testing data.')

    def read_preprocess_label(self, save_file_path=None):
        if save_file_path is None:
            save_file_path = self.config['load_to']['save_file_path']
        with open("{}/{}".format(save_file_path, 'training_label.pkl'), 'rb') as file:
            self.label = pickle.load(file)
        self.log.info('Successfully read label file data.')

    def train_test_split(self):
        self.data['x_train'], self.data['x_valid'], self.data['y_train'], self.data['y_valid'] = train_test_split(
            self.df['train'], self.label, test_size=0.05, random_state=42)
        self.log.info('Successfully split the training data and validation data.')
        self.log.info('shape of training data: {}'.format(len(self.data['x_train'])))
        self.log.info('shape of validation data: {}'.format(len(self.data['x_valid'])))
    
    def linear_regression_training(self):
        self.model = LinearRegression()
        self.model.fit(self.data['x_train'], self.data['y_train'])
        validation_score = self.model.score(self.data['x_valid'], self.data['y_valid'])
        self.log.info("Linear regression score: {}".format(validation_score))
    
    def gbr_training(self):
        self.model = GradientBoostingRegressor(
            n_estimators=400, max_depth=5, min_samples_split=2, learning_rate=0.1, loss='ls')
        self.model.fit(self.data['x_train'], self.data['y_train'])
    
    def svr_training(self):
        self.model = SVR(kernel='rbf')
        self.model.fit(self.data['x_train'], self.data['y_train'])

    def evaluate(self, model):
        if model == 'svr':
            validation_score = self.model.predict(self.data['x_valid'])
        else:
            validation_score = self.model.score(self.data['x_valid'], self.data['y_valid'])
        self.log.info("Score: {}".format(validation_score))
    
    def model_save(self, model, save_file_path=None):
        if save_file_path is None:
            save_file_path = self.config['load_to']['save_file_path']
        file = open("{}/model/training_model_{}.pkl".format(save_file_path, model), 'wb')
        pickle.dump(self.model, file)
        file.close()
