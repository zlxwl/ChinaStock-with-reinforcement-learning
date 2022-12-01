# -*- coding: utf-8 -*-
# @Time    : 2022/12/1 下午4:59
# @Author  : Zhong Lei
# @FileName: data.py
import os
from typing import List
import pandas as pd
import config


class Data:
    def __init__(self,
                 ticker_list: List=config.SSE_50,
                 data_dir: str='data_dir'):
        self.ticker_list = ticker_list
        self.data_dir = data_dir
        self.create_data_dir()

    def create_data_dir(self) -> None:
        if not os.path.exists(self.data_dir):
            os.makedirs(path=self.data_dir)
            print("{} create file success !".format(self.data_dir))
        else:
            print("file {} already exist ".format(self.data_dir))

    def pull_data(self) -> pd.DataFrame:
        pass

    def split_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def display_data_information(self,
                                 train_data: pd.DataFrame,
                                 trade_data: pd.DataFrame) -> None:
        pass

    def save_data(self,
                  train_data: pd.DataFrame,
                  trade_data: pd.DataFrame) -> None:
        pass


if __name__ == '__main__':
    # =============== test pull_data() ====================
    Data().pull_data()
