# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : models.py
# Time       ：2022/11/20 21:21
# Author     ：Zhong Lei
"""
from typing import Any
from stable_baselines3 import DDPG, A2C, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import config
from env import StockLearningEnv
import numpy as np
import pandas as pd

MODELS = {
    'a2c': A2C,
    'ddpg': DDPG,
    'td3': TD3,
    'ppo': PPO,
    'sac': SAC
}
MODEL_KWARGS = {x: config.__dict__['{}_PARAMS'.format(x.upper())] for x in MODELS.keys()}
NOISE = {
    'normal': NormalActionNoise,
    'ornstein_unlenbeck': OrnsteinUhlenbeckActionNoise
}


class DRLAgent:
    def __init__(self, env: StockLearningEnv):
        self.env = env

    def train(self, model: Any, tb_log_name: str, total_timesteps: int = 5000):
        model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)
        return model

    @staticmethod
    def prediction(model: Any, env: StockLearningEnv) -> pd.DataFrame:
        test_env, test_obs = env.get_sb_env()
        account_memory = []
        actions_memory = []
        test_env.reset()

        len_env = len(env.df.index.unique())
        for i in range(len_env):
            action, _states = model.predict(test_obs)
            test_obs, _, dones, _ = test_env.step(action)
            if i == len_env - 2:
                account_memory = test_env.env_method(method_name='save_asset_memory')
                actions_memory = test_env.env_method(method_name='save_action_memory')
            if dones[0]:
                account_memory = test_env.env_method(method_name='save_asset_memory')
                actions_memory = test_env.env_method(method_name='save_action_memory')
                print('回测完成')
                break
        return account_memory[0], actions_memory[0]

    def get_model(self, model_name: str,
                  policy: str = 'MlpPolicy',
                  policy_kwargs: dict = None,
                  model_kwargs: dict = None,
                  verbose: int = 1
                  ) -> Any:
        if model_name not in MODELS:
            raise NotImplementedError('model not exist')

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs['action_noise'] = NOISE[model_kwargs['action_noise']](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )

        print(model_kwargs)

        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log='{}/{}'.format(config.TENSORBOARD_LOG_DIR, model_name),
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            **model_kwargs
        )
        return model


if __name__ == '__main__':
    print()
