# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : env.py
# Time       ：2022/11/5 21:01
# Author     ：Zhong Lei
"""
from copy import deepcopy
from typing import List
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common import logger, utils
import time
import random


class StockLearningEnv(gym.Env):
    metadata = {
        "render.modes": ['huamn']
    }

    def __init__(
            self,
            df: pd.DataFrame,
            buy_cost_pct: float = 3e-3,
            sell_cost_pct: float = 3e-3,
            data_col_name: str = 'date',
            hmax: int = 10,
            print_verbosity: int = 10,
            initial_amount: float = 1e6,
            daily_information_cols: List = ['open', 'close', 'high', 'low', 'volume'],
            cache_indicator_data: bool = True,
            random_start: bool = True,
            patient: bool = True,
            currency: str = '￥'
    ):
        self.df = df
        self.stock_col = 'tic'
        self.assets = df[self.stock_col].unique()
        self.dates = df[data_col_name].sort_values().unique()
        self.random_start = random_start
        self.patient = patient
        self.currency = currency

        self.df = self.df.set_index(data_col_name)
        self.hmax = hmax
        self.initial_mount = initial_amount
        self.print_verbosity = print_verbosity
        self.bus_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.daily_information_cols = daily_information_cols
        self.cache_indicator_data = cache_indicator_data

        self.state_space = (1 + len(self.assets) + len(self.assets) * len(self.daily_information_cols))
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.assets),))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))

        self.turbulence = 0
        self.episode = -1
        self.episode_history = []
        self.print_header = False
        self.cache_data = None
        self.max_total_assets = 0
        if self.cache_indicator_data:
            """
            [[data1], [data2], ...] date * [stock * col]
            data1 = [stock1 * cols, stock2 * cols, ...]  stock
            """
            print('加载缓存数据')
            self.cache_data = [self.get_date_vector(i) for i, _ in enumerate(self.dates)]
            # self.cache_data = [self.stock_order_by_day(date) for date in self.dates]
            print('数据缓存成功')

    def seed(self, seed=None):
        if seed is None:
            seed = int(round(time.time()) * 1000)
        random.seed(seed)

    @property
    def current_step(self):
        return self.date_index - self.starting_point

    @property
    def holdings(self):
        return self.state_memory[-1][1: len(self.assets) + 1]

    @property
    def cash_on_hand(self):
        return self.state_memory[-1][0]

    @property
    def closings(self):
        return np.array(self.get_date_vector(self.date_index, cols=['close']))

    def get_date_vector(self, date: int, cols: List = None):
        if (cols is None) and (self.cache_data is not None):
            return self.cache_data[date]
        else:
            date = self.dates[date]
            if cols is None:
                cols = self.daily_information_cols
            trunc_df = self.df.loc[[date]]
            res = []
            for asset in self.assets:
                tmp_res = trunc_df[trunc_df[self.stock_col] == asset]
                res += tmp_res.loc[date, cols].to_list()
            assert len(res) == len(self.assets) * len(cols)
            return res

    def stock_order_by_day(self, date, cols=None):
        # list  [date_1, date_2, ... date_n]  data_1 = [tic * col]
        res = []
        trunc_df = self.df.loc[date]
        for asset in self.assets:
            tmp_res = trunc_df[trunc_df[self.stock_col] == asset]

            if tmp_res.shape[0] == 0:
                tmp_res = [asset] + (tmp_res.shape[1] - 1) * [0]
                res += tmp_res
            else:
                res += tmp_res.loc[date].to_list()
        # assert len(res) == len(self.assets) * len(cols)
        return res

    def reset(self):
        self.seed()
        self.sum_trades = 0
        self.max_total_assets = self.initial_mount
        if self.random_start:
            self.starting_point = random.choice(range(int(len(self.dates) * 0.5)))
        else:
            self.starting_point = 0
        self.date_index = self.starting_point
        self.turbulence = 0
        self.episode += 1

        self.actions_memory = []
        self.transaction_memory = []
        self.state_memory = []
        self.account_information = {
            'cash': [],
            'asset_value': [],
            'total_assets': [],
            'reward': []
        }

        init_state = np.array(
            [self.initial_mount]
            + [0] * len(self.assets)
            + self.get_date_vector(self.date_index)
        )
        self.state_memory.append(init_state)

        return init_state

    def return_terminal(self, reason: str = 'Last Date', reward: int = 0):
        state = self.state_memory[-1]
        self.log_step(reason=reason, terminal_reward=reward)
        self.logger = utils.configure_logger()
        gl_pct = self.account_information['total_assets'][-1] / self.initial_mount

        self.logger.record('environment/gain_loss_pct', (gl_pct - 1) * 100)
        self.logger.record('environment/total_assets', int(self.account_information['total_assets'][-1]))

        reward_pct = gl_pct
        self.logger.record('environment/total_reward_pct', (reward_pct - 1) * 100)
        self.logger.record('environment/total_trades', self.sum_trades)
        self.logger.record('environment/avg_daily_trades', self.sum_trades / self.current_step)
        self.logger.record('environment/avg_daily_trades_per_asset',
                           self.sum_trades / self.current_step / len(self.assets))
        self.logger.record('environment/completed_steps', self.current_step)
        self.logger.record('environment/sum_rewards', np.sum(self.account_information['reward']))
        self.logger.record('environment/retreat_proportion',
                           self.account_information['total_assets'][-1] / self.max_total_assets)
        return state, reward, True, {}

    def log_header(self):
        if not self.print_header:
            self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"
            print(self.template.format(
                "EPISODE",
                "STEPS",
                "TERMINAL_REASON",
                "CASH",
                "TOT_ASSETS",
                "TERMINAL_REWARD",
                "GAINLOSS_PCT",
                "RETREAT_PROPORTION"
            ))
            self.print_header = True

    def log_step(self, reason: str, terminal_reward: float = None):
        if terminal_reward is None:
            terminal_reward = self.account_information['reward'][-1]
        assets = self.account_information['total_assets'][-1]
        temp_retreat_pct = assets / self.max_total_assets - 1
        retreat_pct = temp_retreat_pct if assets < self.max_total_assets else 0
        gl_pct = self.account_information['total_assets'][-1] / self.initial_mount

        rec = [
            self.episode,
            self.date_index - self.starting_point,
            reason,
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['cash'][-1]))}",
            f"{self.currency}{'{:0,.0f}'.format(float(assets))}",
            f"{terminal_reward * 100:0.5f}%",
            f"{(gl_pct - 1) * 100:0.5f}%",
            f"{retreat_pct * 100:0.2f}%"
        ]
        self.episode_history.append(rec)
        print(self.template.format(*rec))

    def get_transactions(self, actions: np.array):
        self.actions_memory.append(actions)
        actions = actions * self.hmax

        actions = np.where(self.closings > 0, actions, 0)
        out = np.zeros_like(actions)
        zero_or_not = self.closings != 0

        actions = np.divide(actions, self.closings, out=out, where=zero_or_not)
        actions = np.maximum(actions, -np.array(self.holdings))
        actions[actions == -0] = 0
        return actions

    def get_reward(self):
        if self.current_step == 0:
            return 0
        else:
            assets = self.account_information['total_assets'][-1]
            retreat = 0
            if assets >= self.max_total_assets:
                self.max_total_assets = assets
            else:
                retreat = assets / self.max_total_assets - 1
            reward = assets / self.initial_mount - 1  # (收益率)
            reward += retreat  # reward = gc_pct + retreat
            return reward

    def step(self, actions: np.array):
        self.sum_trades += np.sum(np.abs(actions))
        self.log_header()
        if (self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason='update')
        if self.date_index == len(self.dates) - 1:
            return self.return_terminal(reward=self.get_reward())
        else:
            begin_cash = self.cash_on_hand
            assert min(self.holdings) >= 0
            assert_value = np.dot(self.holdings, self.closings)
            self.account_information['asset_value'].append(assert_value)
            self.account_information['cash'].append(begin_cash)
            reward = self.get_reward()
            self.account_information['reward'].append(reward)
            self.account_information['total_assets'].append(assert_value + begin_cash)

            # first sell
            transactions = self.get_transactions(actions)
            sells = -np.clip(transactions, -np.inf, 0)
            proceeds = np.dot(sells, self.closings)
            costs = proceeds * self.sell_cost_pct
            coh = begin_cash + proceeds

            # then buy
            buys = np.clip(transactions, np.inf, 0)
            spend = np.dot(buys, self.closings)
            costs += spend * self.bus_cost_pct

            if (spend + costs) > coh:
                if self.patient:
                    transactions = np.where(transactions > 0, 0, transactions)
                    spend = 0
                    costs = 0
                else:
                    return self.return_terminal(reason='CASH SHORTAGE', reward=self.get_reward())

            self.transaction_memory.append(transactions)
            assert (spend + costs) <= coh
            coh = coh - spend - costs
            holdings_updates = self.holdings + transactions

            self.date_index += 1
            state = (
                    [coh] + list(holdings_updates) + self.get_date_vector(self.date_index)
            )
            self.state_memory.append(state)
            return state, reward, False, {}

    def get_sb_env(self):
        def get_self():
            return deepcopy(self)

        e = DummyVecEnv([get_self])
        obs = e.reset()
        return e, obs

    def get_multiporc_env(self, n):
        def get_self():
            return deepcopy(self)

        e = SubprocVecEnv([get_self for _ in range(n)], start_method='fork')
        obs = e.reset()
        return e, obs

    def save_asset_memory(self):
        if self.current_step == 0:
            return None
        else:
            self.account_information['date'] = self.dates[-len(self.account_information['cash']):]
            return pd.DateFrame(self.account_information)

    def save_action_memory(self):
        if self.current_step == 0:
            return None
        else:
            return pd.DataFrame(
                {
                    'date': self.account_information[-len(self.account_information['cash']):],
                    'actions': self.actions_memory,
                    'transactions': self.transaction_memory
                }
            )
