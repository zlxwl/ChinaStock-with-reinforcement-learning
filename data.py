# -*- coding: utf-8 -*-
# @Time    : 2022/12/1 下午4:59
# @Author  : Zhong Lei
# @FileName: data.py
import os
from typing import List
import pandas as pd
import config
from data_downloader import TuDataDownLoader
from stockstats import StockDataFrame as Sdf
import itertools


class Data:
    def __init__(self,
                 ticker_list: List = config.SSE_50,
                 start_date: str = config.START_DATE,
                 end_date: str = config.END_DATE,
                 tushare_tocken: str = config.TUSHARE_TOCKEN,
                 pull_index: bool = False,
                 data_dir: str = 'data_file'):
        self.ticker_list = ticker_list
        self.data_dir = data_dir
        self.create_data_dir()

        self.start_date = start_date
        self.end_date = end_date
        self.tuchare_token = tushare_tocken
        self.pull_index = pull_index

    def create_data_dir(self) -> None:
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print("{} create file success !".format(self.data_dir))
        else:
            print("file {} already exist ".format(self.data_dir))

    def pull_data(self) -> pd.DataFrame:
        data = TuDataDownLoader(ticker_list=self.ticker_list,
                                start_date=self.start_date,
                                endt_date=self.end_date,
                                tushare_token=self.tuchare_token,
                                pull_index=self.pull_index).pull_data()
        data.sort_values(['date', 'tic'], ignore_index=True).head()
        print('download data from {} to {}'.format(config.START_DATE, config.END_DATE))
        print('download ticker list: ')
        print(self.ticker_list)

        processed_df = FeatureEngineer(use_technical_indicator=True).preprocess_data(data)
        processed_df['amount'] = processed_df.volume * processed_df.close
        processed_df['change'] = (processed_df.close - processed_df.open) / processed_df.close
        processed_df['daily_variance'] = (processed_df.high - processed_df.low) / processed_df.close
        processed_df = processed_df.fillna(0)

        print('technical indicator list:')
        print(config.TECHNICAL_INDICATOR_LIST)
        print('technical {} '.format(len(config.TECHNICAL_INDICATOR_LIST)))
        print(processed_df.head())
        processed_df.to_csv(os.path.join(self.data_dir, 'data.csv'), index=False)
        self.split_data(processed_df)

    def split_data(self, data: pd.DataFrame) -> pd.DataFrame:
        def split_data_by_data(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
            data = df[(df.date >= start) & (df.date < end)]
            data = data.sort_values(['date', 'tic'], ignore_index=True)
            data.index = data.date.factorize()[0]
            return data

        # do split data by train_data, trade_data;
        train_data = split_data_by_data(data, config.STATE_TRADE_DATE, config.END_TRADE_DATE)
        trade_data = split_data_by_data(data, config.END_TRADE_DATE, config.END_TEST_DATE)

        self.display_data_information(train_data, trade_data)
        self.save_data(train_data, trade_data)

    def display_data_information(self,
                                 train_data: pd.DataFrame,
                                 trade_data: pd.DataFrame) -> None:
        print("train data: from {} to {}".format(config.STATE_TRADE_DATE,
                                                 config.END_TRADE_DATE))
        print("test data: from {} to {}".format(config.END_TRADE_DATE,
                                                config.END_TEST_DATE))
        print("train data length: {}, test data length {}".format(len(train_data), len(trade_data)))
        print("train data : test data: {}: {}".format(round(len(train_data) / len(trade_data), 1), 1))
        print("train_data.head:")
        print(train_data.head())
        print("trade_data.head:")
        print(trade_data.head())

    def save_data(self,
                  train_data: pd.DataFrame,
                  trade_data: pd.DataFrame) -> None:
        train_data.to_csv(os.path.join(self.data_dir, "train.csv"), index=False)
        trade_data.to_csv(os.path.join(self.data_dir, "trade.csv"), index=False)


class FeatureEngineer:
    def __init__(self,
                 return_full_table: bool = True,
                 use_technical_indicator: bool = True,
                 tech_indicator_list: List = config.TECHNICAL_INDICATOR_LIST):
        self.return_full_table = return_full_table
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.use_technical_indicator:
            df = self.add_technical_indicator(df)
            print('add technical indicator success!')
        if self.return_full_table:
            df = self.full_table(df)
            print('use 0 to fillna unlisted company')
        return df

    def add_technical_indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(by=['tic', 'date'])

        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for ticker in unique_ticker:
                tmp_df = pd.DataFrame(stock[stock.tic == ticker][indicator])
                tmp_df['tic'] = ticker
                tmp_df['date'] = df[df.tic == ticker]['date'].to_list()
                indicator_df = indicator_df.append(tmp_df, ignore_index=True)
            df = df.merge(indicator_df[['tic', 'date', indicator]], on=['tic', 'date'], how='left')
        df = df.sort_values(by=['date', 'tic'])
        return df

    def full_table(self, df: pd.DataFrame) -> pd.DataFrame:
        ticker_list = df['tic'].unique().tolist()
        date_list = list(pd.date_range(df['date'].min(), df['date'].max()).astype(str))
        combination = list(itertools.product(date_list, ticker_list))

        df_full = pd.DataFrame(combination, columns=['date', 'tic']).merge(df, on=['date', 'tic'])
        df_full = df_full[df_full['date'].isin(df['date'])].fillna(0)
        df_full = df_full.sort_values(['date', 'tic'], ignore_index=True)
        return df_full


if __name__ == '__main__':
    # =============== test pull_data() ====================
    # Data().pull_data()

    # =============== test FeatureEngineer() ==============
    # df = TuDataDownLoader(ticker_list=config.SSE_50[:2],
    #                  start_date=config.START_DATE,
    #                  endt_date=config.END_DATE,
    #                  tushare_token=config.TUSHARE_TOCKEN,
    #                  pull_index=False).pull_data()
    Data().pull_data()