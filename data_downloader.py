# -*- coding: utf-8 -*-
# @Time    : 2022/12/1 下午5:29
# @Author  : Zhong Lei
# @FileName: TushareDataDownLoader.py
from typing import List
import tushare as ts
import pandas as pd
from datetime import datetime
import time
import config


class TuDataDownLoader:
    def __init__(self,
                 ticker_list: List,
                 start_date: str,
                 endt_date: str,
                 tushare_token: str,
                 pull_index: bool):
        self.ticker_list = ticker_list
        self.start_date = start_date
        self.end_date = endt_date
        self.tushare_token = tushare_token
        self.pull_index = pull_index

        self.ticker_len = len(self.ticker_list)
        self.data_time = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        self.pro = self.init_token()

    def init_token(self):
        ts.set_token(self.tushare_token)
        return ts.pro_api()

    def pull_data(self) -> pd.DataFrame:
        df = pd.DataFrame()
        counter = 0
        print('start downloading')
        for ticker in self.ticker_list:
            counter += 1
            if counter % 10 == 0:
                print('downloading process: {}%'
                      .format(counter / len(self.ticker_list) * 100))
            try:
                if not self.pull_index:
                    tmp_df = ts.pro_bar(ts_code=ticker,
                                        asset='E',
                                        adj='qfq',
                                        start_date=self.start_date,
                                        end_date=self.end_date)
                else:
                    tmp_df = self.pro.index_daily(ts_code=ticker,
                                                  adj='qfq',
                                                  start_date=self.start_date,
                                                  end_date=self.end_date)
                tmd_df = tmp_df.set_index('trade_date', drop=True)
                df = df.append(tmd_df)
            except:
                print(" sleep for 3 seconds")
                time.sleep(3)
        print('download finished')

        df = df.reset_index()
        df = df.drop(columns=['pre_close', 'change', 'pct_chg', 'amount'])
        df.columns = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']

        df['date'] = df.date.apply(lambda x: datetime.strptime(x[:4] + '-' + x[4:6] + '-' + x[6:], "%Y-%m-%d"))
        df['day'] = df['date'].dt.dayofweek
        df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
        df['date'] = df.date.apply(lambda x: x.strftime("%Y-%m-%d"))

        df = df.dropna()
        df = df.reset_index(drop=True)
        df = df.sort_values(by=['date', 'tic']).reset_index(drop=True)
        print('dateframe size {}'.format(df.shape))
        return df


if __name__ == '__main__':
    # ========== test pull_data ==================
    TuDataDownLoader(ticker_list=config.CSI_300[:2],
                     start_date=config.START_DATE,
                     endt_date=config.END_DATE,
                     tushare_token=config.TUSHARE_TOCKEN,
                     pull_index=False).pull_data()
