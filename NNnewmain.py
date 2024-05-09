import sys
import yaml
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from job_generator import JobGenerator
# from myagent_gym import  WTPAgent, CostPAgent, RandomAgent, MyAgent
from multiAgent import  MyAgent
from mypolicy import EpsGreedyQPolicy_with_action_filter_and_decay
from mylogger import EpisodeLogger, EpisodeLogger_4_dqn, EpisodeLogger_4_dqn_test
from myenv_reschedule import SchedulingEnv
from util.datetime import get_timestamp
import gym
import random

debug = True
random.seed(0)


with open('config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

# 環境のパラメータを設定
max_step = np.inf
n_window = config['param_env']['n_window']
n_on_premise_node = config['param_env']['n_on_premise_node']
n_cloud_node = config['param_env']['n_cloud_node']
n_job_queue_obs = config['param_env']['n_job_queue_obs']
n_job_queue_bck = config['param_env']['n_job_queue_bck']
penalty_not_allocate = config['param_env']['penalty_not_allocate']  # 割り当てない(一時キューに格納する)という行動を選択した際のペナルティー
penalty_invalid_action = config['param_env']['penalty_invalid_action']  # actionが無効だった場合のペナルティー


# 基本パラメータによって決まるパラメータ
# モデルの入力層のノード数(observationの要素数)
n_observation = n_on_premise_node*n_window + n_cloud_node*n_window + 4*10
# モデルの出力層のノード数(actionの要素数)
n_action = 4


# ジョブと学習環境の設定

# 報酬の各目的関数の重みを設定
weight_wt = config['param_agent']['weight_wt']
weight_cost = config['param_agent']['weight_cost']

nb_steps = config['param_simulation']['nb_steps']
nb_episodes = config['param_simulation']['nb_episodes']
nb_max_episode_steps = config['param_simulation']['nb_max_episode_steps'] # 1エピソードあたりの最大ステップ数(-1:最大ステップ無し)
if nb_max_episode_steps == -1:
    nb_max_episode_steps = np.inf

multi_algorithm = config['param_simulation']['multi_algorithm'] # 0:single algorithm(後でジョブを決める) 1:multi algorithm

if multi_algorithm:
    # 環境に入力するジョブを生成
    job_generator = JobGenerator(nb_steps, n_window, n_on_premise_node, n_cloud_node, config)
    jobs_set = job_generator.generate_jobs_set()

    env=SchedulingEnv(
        max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
        weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action, multi_algorithm=True,
        jobs_set=jobs_set)

else:
    env = SchedulingEnv(
        max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,
        weight_wt, weight_cost, penalty_not_allocate, penalty_invalid_action
    )


# 画像を保存するフォルダを作成
# folder_suffix = config['param_simulation']['folder_suffix']
# folder_path = 'results/' + get_timestamp() + '-' + folder_suffix if folder_suffix else get_timestamp() # 実験結果を保存するフォルダの名前
# if not os.path.exists(folder_path):  # ディレクトリがなかったら
#     os.mkdir(folder_path)  # 作成したいフォルダ名を作成
# else:  # 既にあれば
#     print('その名前のフォルダは既に存在します')
#     sys.exit()

# f_prm = open(folder_path + '/parameter.csv', 'w')
prm_env = [env.weight_wt, env.weight_cost, env.n_window, env.n_on_premise_node, env.n_cloud_node, env.n_job_queue_obs, env.n_job_queue_bck, env.penalty_invalid_action]
df_prme = pd.DataFrame(
    [prm_env],
    columns=['weight_wt', 'weight_cost', 'n_window', 'n_on_premise_node', 'n_cloud_node', 'n_job_queue_obs', 'n_job_queue_bck',
           'penalty_invalid_action'],
    index=['idx']
)
# df_prme.to_csv(folder_path + '/parameter_env.csv')

# ジョブをjsonファイルとして出力
# job_generator.save_jobs_set(folder_path + '/jobs_set.json')

# if job_generator.job_type == 3 or job_generator.job_type == 4:
# if job_generator.df_prmj is not None:
#     job_generator.df_prmj.to_csv(folder_path + '/parameter_job.csv')


# NN，エージェントの定義
# ニューラルネットワークの構造を定義
# 入力層の定義
model1 = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model1.add(Flatten(input_shape=(1,) + (n_observation,)))
model1.add(Dense(256))
model1.add(Activation('relu'))
model1.add(Dense(64))
model1.add(Activation('relu'))
model1.add(Dense(n_action))
model1.add(Activation('linear'))
# print(model.summary())  # モデルの定義をコンソールに出力

model2 = Sequential()
model2.add(Flatten(input_shape=(1,) + (n_observation,)))
model2.add(Dense(256))
model2.add(Activation('relu'))
model2.add(Dense(64))
model2.add(Activation('relu'))
model2.add(Dense(n_action))
model2.add(Activation('linear'))

model3 = Sequential()
model3.add(Flatten(input_shape=(1,) + (n_observation,)))
model3.add(Dense(256))
model3.add(Activation('relu'))
model3.add(Dense(64))
model3.add(Activation('relu'))
model3.add(Dense(n_action))
model3.add(Activation('linear'))


# モデルのコンパイル
memory1 = SequentialMemory(limit=50000, window_length=1)

# policyの宣言
# 0.99996^100000==0.01
EGQpolicy = EpsGreedyQPolicy(eps=0.01)
EGQpolicy_d = EpsGreedyQPolicy_with_action_filter_and_decay(eps=1, min_eps=.01, eps_decay=.99998)


# dqn = DQNAgent(model=model, nb_actions=n_action, memory=memory, nb_steps_warmup=50, target_model_update=1e-2, policy=policy)
# dqn = DQNAgent_without_action_filter(model=model, nb_actions=n_action, memory=memory, nb_steps_warmup=50, target_model_update=1e-2, policy=EGQpolicy_d)
dqn = MyAgent(
    model1 = model1, model2 = model2, model3 = model3, nb_actions=n_action, memory = memory1, nb_episodes_warmup=30,
    target_model_update=1e-2, policy=EGQpolicy_d, n_window=n_window,
    n_on_premise_node=n_on_premise_node, n_cloud_node=n_cloud_node,
    n_job_queue_obs=n_job_queue_obs, n_job_queue_bck=n_job_queue_bck,object_mode=sys.argv[1]
)
dqn.compile(optimizer=Adam(learning_rate=1e-3),metrics=['mse','mse'])



# 全アルゴリズム同士同じジョブで一気に実行

# loggerの宣言
episode_logger_dqn = EpisodeLogger_4_dqn()
episode_logger_wtp = EpisodeLogger()
episode_logger_cp = EpisodeLogger()
episode_logger_rnd = EpisodeLogger()

# dqnの学習
history = dqn.fit(env, nb_episodes=nb_episodes, visualize=False, verbose=2, nb_max_episode_steps=nb_max_episode_steps, callbacks=[episode_logger_dqn])

# # 待ち時間優先のエージェント
# wtp = WTPAgent(nb_actions=n_action, n_window=n_window, n_on_premise_node=n_on_premise_node, n_cloud_node=n_cloud_node, n_job_queue_obs=n_job_queue_obs, n_job_queue_bck=n_job_queue_bck)
# # 訓練(?)
# history = wtp.fit(env, nb_steps=nb_steps, visualize=False, verbose=2, nb_max_episode_steps=nb_max_episode_steps, callbacks=[episode_logger_wtp])

# # コスト優先のエージェント
# cp = CostPAgent(nb_actions=n_action, n_window=n_window, n_on_premise_node=n_on_premise_node, n_cloud_node=n_cloud_node, n_job_queue_obs=n_job_queue_obs, n_job_queue_bck=n_job_queue_bck)
# # 訓練(?)
# history = cp.fit(env, nb_steps=nb_steps, visualize=False, verbose=2, nb_max_episode_steps=nb_max_episode_steps, callbacks=[episode_logger_cp])

# # ランダム行動のエージェント
# rnd = RandomAgent(nb_actions=n_action, n_window=n_window, n_on_premise_node=n_on_premise_node, n_cloud_node=n_cloud_node, n_job_queue_obs=n_job_queue_obs, n_job_queue_bck=n_job_queue_bck)
# # 訓練(?)
# history = rnd.fit(env, nb_steps=nb_steps, visualize=False, verbose=2, nb_max_episode_steps=nb_max_episode_steps, callbacks=[episode_logger_rnd])


# モデルを保存
# model_path = folder_path + '/model'
# try:
#     dqn.model.save(model_path)
#     print('saved model to {}'.format(model_path))
# except:
#     print('failed to save model')

# 各エピソードで割り当てないという行動を取った回数
count_not_allocate = np.zeros(len(episode_logger_dqn.actions.keys()))
for episode in episode_logger_dqn.actions.keys():
    actions = episode_logger_dqn.actions[episode]
    for a in actions:
        if a == n_action - 1:
            count_not_allocate[episode] += 1

x = list(range(len(count_not_allocate)))
y_not_allocate = count_not_allocate
fig = plt.figure()
# plt.title('count not allocate')
plt.xlabel('episode')
plt.ylabel('count not allocate')
plt.plot(x, y_not_allocate, label='dqn')
# plt.show()
# file_path = folder_path + '/count_not_allocate.png'
# fig.savefig(file_path)

# df_dqn_train['not_allocate'] = y_not_allocate


# 各エピソードで無効な行動を取った回数
count_invalid_action = np.zeros(len(episode_logger_dqn.actions.keys()))
for episode in episode_logger_dqn.actions.keys():
    actions = episode_logger_dqn.actions[episode]
    rewards = episode_logger_dqn.rewards[episode]
    # print(actions)
    # for a, r in zip(actions, [item[0] for item in rewards]):
    for a, r in zip(actions, rewards):
        if a != n_action - 1 and r == -penalty_invalid_action:
            count_invalid_action[episode] += 1

x = list(range(len(count_invalid_action)))
y_invalid_action = count_invalid_action
fig = plt.figure()
# plt.title('count invalid action')
plt.xlabel('episode')
plt.ylabel('count invalid action')
plt.plot(x, y_invalid_action, label='dqn')
# plt.show()
# file_path = folder_path + '/count_invalid_action.png'
# fig.savefig(file_path)


# # テスト

# model = load_model(model_path)

# dqn = MyAgent(
#     model=model, nb_actions=n_action, memory=memory, nb_steps_warmup=50,
#     target_model_update=1e-2, policy=EGQpolicy, n_window=n_window,
#     n_on_premise_node=n_on_premise_node, n_cloud_node=n_cloud_node,
#     n_job_queue_obs=n_job_queue_obs, n_job_queue_bck=n_job_queue_bck
# )
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])


# # 記録用
# episode_logger_dqn_test = EpisodeLogger_4_dqn_test()
# episode_logger_wtp_test = EpisodeLogger()
# episode_logger_cp_test = EpisodeLogger()
# episode_logger_rnd_test = EpisodeLogger()

# テスト
# history = dqn.test(env, nb_episodes=10, visualize=False, callbacks=[episode_logger_dqn_test])

# history = wtp.test(env, nb_episodes=10, visualize=False, callbacks=[episode_logger_wtp_test])
# history = cp.test(env, nb_episodes=10, visualize=False, callbacks=[episode_logger_cp_test])
# history = rnd.test(env, nb_episodes=10, visualize=False, callbacks=[episode_logger_rnd_test])


# matplotlibの日本語設定
# from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']



