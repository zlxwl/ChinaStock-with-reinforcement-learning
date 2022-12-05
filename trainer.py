# -*- coding: utf-8 -*-
# @Time    : 2022/12/5 上午11:31
# @Author  : Zhong Lei
# @FileName: trainer.py
import os
from env import StockLearningEnv
from models import DRLAgent
import config
from data import Data
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from argparse import ArgumentParser


class Trainer:
    def __init__(self,
                 model_name='a2c',
                 total_timesteps=200000,
                 train_dir='train_file',
                 data_dir='data_dir') -> None:
        self.model_name = model_name
        self.total_timesteps = total_timesteps
        self.train_dir = train_dir
        self.data_dir = data_dir

    def create_train_dir(self):
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
            print('create file {} succeeded!'.format(self.train_dir))
        else:
            print('file {} already exist!'.format(self.train_dir))

    def train(self):
        train_data, trade_data = self.get_data()
        train_env, trade_env = self.get_env(train_data, trade_data)
        agent = DRLAgent(env=train_env)
        model = agent.get_model(self.model_name,
                                model_kwargs=config.__dict__['{}_PARAMS'.format(self.model_name.upper())],
                                verbose=0)
        model.learn(total_timesteps=self.total_timesteps,
                    eval_env=train_env,
                    eval_freq=500,
                    log_interval=1,
                    tb_log_name='env_cashpenalty_highlr',
                    n_eval_episodes=1)
        self.save_model()

    def get_data(self):
        train_data_path = os.path.join(self.data_dir, 'train.csv')
        trade_data_path = os.path.join(self.data_dir, 'trade.csv')
        if not (os.path.exists(train_data_path) or
                os.path.exists(trade_data_path)):
            print('data not exist, start downloading')
            Data().pull_data()
        print('download finished, start reading')
        train_data = pd.read_csv(train_data_path)
        trade_data = pd.read_csv(trade_data_path)
        print('reading success!')
        return train_data, trade_data

    def get_env(self,
                train_data: pd.DataFrame,
                trade_data: pd.DataFrame) -> DummyVecEnv:
        env_train, _ = StockLearningEnv(df=train_data,
                                        random_start=True,
                                        **config.ENV_PARAMS).get_sb_env()
        env_trade, _ = StockLearningEnv(df=trade_data,
                                        random_start=True,
                                        **config.ENV_PARAMS).get_sb_env()
        return env_train, env_trade

    def save_model(self, model):
        model_path = os.path.join(self.train_dir, '{}.model'.format(self.model_name))
        model.save(model_path)


def start_train():
    parser = ArgumentParser(description='set parameters for train mode')
    parser.add_argument('--model', '-m',
                        dest='model',
                        default='a2c',
                        help='choose the model type',
                        type=str)
    parser.add_argument('--total_timesteps', '-tts',
                        dest='total_timesteps',
                        default=2000000,
                        help='set the total_timesteps when you train the model',
                        type=int)
    options = parser.parse_args()
    Trainer(model_name=options.model,
            total_timesteps=options.total_timesteps).train()


if __name__ == '__main__':
    start_train()
