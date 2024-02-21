"""This file is the pipeline for data etl"""

# import relation package.
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# import project package.
from config.config_setting import ConfigSetting


class DataEtlService:
    def __init__(self):
        config_setting = ConfigSetting()
        self.config = config_setting.yaml_parser()
        self.log = config_setting.set_logger(["data_etl_service"])
        self.df = {}
        self.train_label = []
        self.one_hot_encoder = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        # read train data
        self.df['train'] = pd.read_csv(self.config['extract']['train_file'])
        self.train_label = list(self.df['train']['SalePrice'])
        # read test data
        self.df['test'] = pd.read_csv(self.config['extract']['test_file'])
        self.log.info("Finish load training data and testing data")
        self.log.info("Length of training data: {}".format(
            len(self.df['train'])))
        self.log.info("Length of testing data: {}".format(
            len(self.df['test'])))

    def remove_feature(self):
        for table in self.df.values():
            for column in self.config['transform']['drop_columns']:
                if column in list(table.columns):
                    table.drop(column, axis=1, inplace=True)
                    self.log.info('Remove columns: {}'.format(column))

    def feature_selection(self):
        total_missing = self.df['train'].isnull().sum()
        to_delete = total_missing[total_missing >
                                  (self.df['train'].shape[0]/3.)]
        for table in self.df.values():
            table.drop(list(to_delete.index), axis=1, inplace=True)

        numerical_features = self.df['test'].select_dtypes(
            include=["float", "int", "bool"]).columns.values
        categorical_features = self.df['train'].select_dtypes(
            include=["object"]).columns.values
        self.log.info("Finish select data feature.")
        self.log.info('Delete feature: {}'.format(list(to_delete.index)))
        self.log.info('numerical_features: {}'.format(
            list(numerical_features)))
        self.log.info('categorical_features: {}'.format(
            list(categorical_features)))
        return numerical_features, categorical_features
    
    def fill_na(self):
        numeric_columns = self.df['train'].select_dtypes(include=[np.number]).columns

        # 對每個數值型列使用其均值進行缺失值填充
        for column in numeric_columns:
            self.df['train'][column] = self.df['train'][column].fillna(self.df['train'][column].mean())
            self.df['test'][column] = self.df['test'][column].fillna(self.df['test'][column].mean())

    def get_dummies(self, categorical_features):
        self.df['train'] = self.df['train'].where(
            pd.notnull(self.df['train']), None)
        self.df['train'] = pd.get_dummies(self.df['train'])
        self.df['test'] = self.df['test'].where(
            pd.notnull(self.df['test']), None)
        self.df['test'] = pd.get_dummies(self.df['test'])
        self.log.info("Finish get dummy.")

    def save_dataframe(self, save_file_path=None):
        if save_file_path is None:
            save_file_path = self.config['load_to']['save_file_path']
        self.df['train'].to_csv(
            "{}/{}".format(save_file_path, 'train.csv'), index=False)
        self.df['test'].to_csv(
            "{}/{}".format(save_file_path, 'test.csv'), index=False)
        self.log.info('Successfully save the dataframe file.')

    def save_label(self, save_file_path=None):
        if save_file_path is None:
            save_file_path = self.config['load_to']['save_file_path']
        file = open("{}/{}".format(save_file_path, 'training_label.pkl'), 'wb')
        pickle.dump(self.train_label, file)
        file.close()
        self.log.info('Successfully save the label.')
