# -*- coding: utf-8 -*-
# @Time    : 2022/12/5 上午11:31
# @Author  : Zhong Lei
# @FileName: trader.py
import os
import pandas as pd
import sys
from stable_baselines3.common.vec_env import DummyVecEnv
from models import DRLAgent
from argparse import ArgumentParser
from env import StockLearningEnv
import config


class Trader:
    def __init__(self,
                 model_name,
                 train_dir='train_file',
                 data_dir='data_file',
                 trade_dir='trade_file'):
        self.model_name = model_name
        self.train_dir = train_dir
        self.data_dir = data_dir
        self.train_dir = trade_dir
        self.create_trade_dir()

    def create_trade_dir(self):
        if not os.path.exists(self.trade_dir):
            os.makedirs(self.trade_dir)
            print('{} file created succeed!'.format(self.trade_dir))
        else:
            print('{} file exit'.format(self.trade_dir))

    def trade(self):
        trade_data = self.get_trade_data()
        env_trade = self.get_env(trade_data)
        agent = DRLAgent(env=env_trade)
        model = self.get_model(agent)
        if model is not None:
            account_value, actions = DRLAgent.prediction(model=model,
                                                         env=env_trade)
            self.save_trade_result(account_value, actions)
            self.print_trade_result(account_value, actions)

    def get_trade_data(self) -> pd.DataFrame:
        trade_data_path = os.path.join(self.data_dir, 'trade.csv')
        if not os.path.exists(trade_data_path):
            print('data not exists, start downloading')

        trade_data = pd.read_csv(trade_data_path)
        return trade_data

    def get_env(self, trade_data: pd.DataFrame) -> DummyVecEnv:
        return StockLearningEnv(df=trade_data,
                                random_start=True,
                                **config.ENV_PARAMS)

    def get_model(self, agent: DRLAgent):
        model = agent.get_model(self.model_name,
                                model_kwargs=config.__dict__['{}_PARAMS'.format(self.model_name.upper())],
                                verbose=0)
        model_dir = os.path.join(self.train_dir, '{}.model'.format(self.model_name))
        if os.path.exists(model_dir):
            model.load(model_dir)
        else:
            return None

    def save_trade_result(self,
                          account_value_df: pd.DataFrame,
                          actions_df: pd.DataFrame) -> None:
        account_value_path = os.path.join(self.train_dir,
                                          'account_value_{}.csv'.format(self.model_name))
        account_value_df.to_csv(account_value_path, index=False)
        actions_path = os.path.join(self.trade_dir, 'actions_{}.csv'.format(self.model_name))
        actions_df.to_csv(actions_path, index=False)

    def print_trade_result(self,
                           account_value_df: pd.DataFrame,
                           actions_df: pd.DataFrame) -> None:
        print('回测的时间窗口: {} to {}'.format(config.END_TRADE_DATE, config.END_TEST_DATE))
        print('查看日账户净值')
        print('开始:')
        print(account_value_df.head())
        print('结束:')
        print(account_value_df.tail())
        print('查看每日所做交易')
        print(actions_df.tail())


def start_trade():
    parser = ArgumentParser(description='set parameters for train mode')
    parser.add_argument('--model',
                        '-m',
                        dest='model',
                        default='a2c',
                        help='choose the model type')
    options = parser.parse_args()
    Trader(model_name=options.model).trade()


if __name__ == '__main__':
    start_trade()