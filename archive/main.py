import sys
import yaml
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from job_generator import JobGenerator
from myagent_gym import  WTPAgent, CostPAgent, RandomAgent, MyAgent
from mypolicy import MultiObjectiveEpsGreedyQPolicy
from mylogger import EpisodeLogger, EpisodeLogger_4_dqn, EpisodeLogger_4_dqn_test
from myenv import SchedulingEnv
from util.datetime import get_timestamp
import gym


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
n_observation = n_on_premise_node*n_window + n_cloud_node*n_window + 4*n_job_queue_obs + 1
# モデルの出力層のノード数(actionの要素数)
n_action = n_window*2 + 1


# ジョブと学習環境の設定

# 報酬の各目的関数の重みを設定
weight_wt = config['param_agent']['weight_wt']
weight_cost = config['param_agent']['weight_cost']

nb_steps = config['param_simulation']['nb_steps']
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
folder_suffix = config['param_simulation']['folder_suffix']
folder_path = 'results/' + get_timestamp() + '-' + folder_suffix if folder_suffix else get_timestamp() # 実験結果を保存するフォルダの名前
if not os.path.exists(folder_path):  # ディレクトリがなかったら
    os.mkdir(folder_path)  # 作成したいフォルダ名を作成
else:  # 既にあれば
    print('その名前のフォルダは既に存在します')
    sys.exit()

# f_prm = open(folder_path + '/parameter.csv', 'w')
prm_env = [env.weight_wt, env.weight_cost, env.n_window, env.n_on_premise_node, env.n_cloud_node, env.n_job_queue_obs, env.n_job_queue_bck, env.penalty_invalid_action]
df_prme = pd.DataFrame(
    [prm_env],
    columns=['weight_wt', 'weight_cost', 'n_window', 'n_on_premise_node', 'n_cloud_node', 'n_job_queue_obs', 'n_job_queue_bck',
           'penalty_invalid_action'],
    index=['idx']
)
df_prme.to_csv(folder_path + '/parameter_env.csv')

# ジョブをjsonファイルとして出力
job_generator.save_jobs_set(folder_path + '/jobs_set.json')

# if job_generator.job_type == 3 or job_generator.job_type == 4:
if job_generator.df_prmj is not None:
    job_generator.df_prmj.to_csv(folder_path + '/parameter_job.csv')


# NN，エージェントの定義
# ニューラルネットワークの構造を定義
model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Flatten(input_shape=(1,) + (n_observation,)))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(n_action))
model.add(Activation('linear'))
print(model.summary())  # モデルの定義をコンソールに出力

# モデルのコンパイル
memory = SequentialMemory(limit=50000, window_length=1)

# policyの宣言
# 0.99996^100000==0.01
EGQpolicy = EpsGreedyQPolicy(eps=0.01)
EGQpolicy_d = MultiObjectiveEpsGreedyQPolicy(eps=1, min_eps=.01, eps_decay=.99999)

# dqn = DQNAgent(model=model, nb_actions=n_action, memory=memory, nb_steps_warmup=50, target_model_update=1e-2, policy=policy)
# dqn = DQNAgent_without_action_filter(model=model, nb_actions=n_action, memory=memory, nb_steps_warmup=50, target_model_update=1e-2, policy=EGQpolicy_d)
dqn = MyAgent(
    model = model, nb_actions=n_action, memory=memory, nb_steps_warmup=50,
    target_model_update=1e-2, policy=EGQpolicy, n_window=n_window,
    n_on_premise_node=n_on_premise_node, n_cloud_node=n_cloud_node,
    n_job_queue_obs=n_job_queue_obs, n_job_queue_bck=n_job_queue_bck
)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])


# 全アルゴリズム同士同じジョブで一気に実行

# loggerの宣言
episode_logger_dqn = EpisodeLogger_4_dqn()
episode_logger_wtp = EpisodeLogger()
episode_logger_cp = EpisodeLogger()
episode_logger_rnd = EpisodeLogger()

# dqnの学習
history = dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=2, nb_max_episode_steps=nb_max_episode_steps, callbacks=[episode_logger_dqn])

# 待ち時間優先のエージェント
wtp = WTPAgent(nb_actions=n_action, n_window=n_window, n_on_premise_node=n_on_premise_node, n_cloud_node=n_cloud_node, n_job_queue_obs=n_job_queue_obs, n_job_queue_bck=n_job_queue_bck)
# 訓練(?)
history = wtp.fit(env, nb_steps=nb_steps, visualize=False, verbose=2, nb_max_episode_steps=nb_max_episode_steps, callbacks=[episode_logger_wtp])

# コスト優先のエージェント
cp = CostPAgent(nb_actions=n_action, n_window=n_window, n_on_premise_node=n_on_premise_node, n_cloud_node=n_cloud_node, n_job_queue_obs=n_job_queue_obs, n_job_queue_bck=n_job_queue_bck)
# 訓練(?)
history = cp.fit(env, nb_steps=nb_steps, visualize=False, verbose=2, nb_max_episode_steps=nb_max_episode_steps, callbacks=[episode_logger_cp])

# ランダム行動のエージェント
rnd = RandomAgent(nb_actions=n_action, n_window=n_window, n_on_premise_node=n_on_premise_node, n_cloud_node=n_cloud_node, n_job_queue_obs=n_job_queue_obs, n_job_queue_bck=n_job_queue_bck)
# 訓練(?)
history = rnd.fit(env, nb_steps=nb_steps, visualize=False, verbose=2, nb_max_episode_steps=nb_max_episode_steps, callbacks=[episode_logger_rnd])

# 学習の様子をdataframeで記録
# df_dqn_train = pd.DataFrame({'loss': episode_logger_dqn.loss.values()})
# df_wtp_train = pd.DataFrame()
# df_cp_train = pd.DataFrame()
# df_rnd_train = pd.DataFrame()

# 学習結果描画
# 画像の画質を設定
plt.rcParams['figure.dpi'] = 300

# loss
# なぜか1エピソード目がnanになるので書き換える
episode_logger_dqn.loss[0] = episode_logger_dqn.loss.get(1, 1)

x_loss = episode_logger_dqn.loss.keys()
y_loss = episode_logger_dqn.loss.values()

fig = plt.figure()
plt.xlabel('episode')
plt.ylabel('mean loss')
plt.plot(x_loss, y_loss, label='dqn')
# plt.show()
file_path = folder_path + '/loss.png'
fig.savefig(file_path)

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
file_path = folder_path + '/count_not_allocate.png'
fig.savefig(file_path)

# df_dqn_train['not_allocate'] = y_not_allocate


# 各エピソードで無効な行動を取った回数
count_invalid_action = np.zeros(len(episode_logger_dqn.actions.keys()))
for episode in episode_logger_dqn.actions.keys():
    actions = episode_logger_dqn.actions[episode]
    rewards = episode_logger_dqn.rewards[episode]
    print(actions)
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
file_path = folder_path + '/count_invalid_action.png'
fig.savefig(file_path)

# df_dqn_train['invalid_action'] = y_invalid_action


# dqnのreward(移動平均あり)
mean_rewards_dqn = {}
for k, v in episode_logger_dqn.rewards.items():
    mean_rewards_dqn[k] = np.mean(v)

x_dqn = mean_rewards_dqn.keys()
y_dqn = mean_rewards_dqn.values()
num = 5  # 移動平均の個数
b = np.ones(num)/num
y_dqn_ma = np.convolve(list(y_dqn), b, mode='same')  # 移動平均

fig = plt.figure()
plt.xlabel('episode')
plt.ylabel('mean reward')
plt.plot(x_dqn, y_dqn, label='raw', alpha=0.3)
plt.plot(x_dqn, y_dqn_ma, label='smoothed', color='#1f77b4')
# plt.title('mean reward per episode')
plt.legend()
# plt.show()

# df_dqn_train['reward'] = y_dqn


# dqnの実質のreward(無効なactionの時の報酬を除外)
mean_rewards_dqn_substantial = {}
for episode in episode_logger_dqn.rewards.keys():
    actions = episode_logger_dqn.actions[episode]
    rewards = episode_logger_dqn.rewards[episode]
    substantial_rewards = []
    for a, e in zip(actions, rewards):
        if e != -penalty_invalid_action:
            # print(a,e)
            substantial_rewards.append(e)
    mean_rewards_dqn_substantial[episode] = np.mean(substantial_rewards)

x_dqn_substantial = mean_rewards_dqn_substantial.keys()
y_dqn_substantial = mean_rewards_dqn_substantial.values()

fig = plt.figure()
plt.xlabel('episode')
plt.ylabel('substantial mean reward')
plt.plot(x_dqn_substantial, y_dqn_substantial, label='dqn')
# plt.title('substantial mean reward per episode')
# plt.legend()
# plt.show()
file_path = folder_path + '/mean_reward_substantial_dqn_train.png'
fig.savefig(file_path)

# df_dqn_train['substantial_reward'] = y_dqn_substantial


# 各エピソードの待ち時間
mean_waiting_times_dqn = {}
for k, v in episode_logger_dqn.waiting_times.items():
    if v:  # 最後のエピソードは途中で打ち切られ，waiting_timeを記録できていないことがあるからあるものだけで
        mean_waiting_times_dqn[k] = np.mean(v)
x_dqn = mean_waiting_times_dqn.keys()
y_dqn = mean_waiting_times_dqn.values()
y_dqn_ma = np.convolve(list(y_dqn), b, mode='same')  # 移動平均

mean_waiting_times_wtp = {}
for k, v in episode_logger_wtp.waiting_times.items():
    if v:  # 最後のエピソードは途中で打ち切られ，waiting_timeを記録できていないことがあるからあるものだけで
        mean_waiting_times_wtp[k] = np.mean(v)
    if len(mean_waiting_times_wtp) == len(mean_waiting_times_dqn):
        break

x_wtp = x_dqn
y_wtp = [np.mean(list(mean_waiting_times_wtp.values()))] * len(x_wtp)

mean_waiting_times_cp = {}
for k, v in episode_logger_cp.waiting_times.items():
    if v:  # 最後のエピソードは途中で打ち切られ，waiting_timeを記録できていないことがあるからあるものだけで
        mean_waiting_times_cp[k] = np.mean(v)
    if len(mean_waiting_times_cp) == len(mean_waiting_times_dqn):
        break
x_cp = x_dqn
y_cp = [np.mean(list(mean_waiting_times_cp.values()))] * len(x_cp)

mean_waiting_times_rnd = {}
for k, v in episode_logger_rnd.waiting_times.items():
    if v:  # 最後のエピソードは途中で打ち切られ，waiting_timeを記録できていないことがあるからあるものだけで
        mean_waiting_times_rnd[k] = np.mean(v)
    if len(mean_waiting_times_rnd) == len(mean_waiting_times_dqn):
        break
x_rnd = x_dqn
y_rnd = [np.mean(list(mean_waiting_times_rnd.values()))] * len(x_rnd)

fig = plt.figure()
plt.xlabel('episode')
plt.ylabel('mean waiting time')
plt.plot(x_dqn, y_dqn, label='dqn(raw)', color='#1f77b4', alpha=0.3)
plt.plot(x_dqn, y_dqn_ma, label='dqn(smoothed)', color='#1f77b4')
plt.plot(x_wtp, y_wtp, label='wtp(average)', color='#ff7f0e', linestyle='dashed')
plt.plot(x_cp, y_cp, label='cp(average)', color='#2ca02c', linestyle='dashdot')
plt.plot(x_rnd, y_rnd, label='random(average)', color='#d62728', linestyle='dotted')
# plt.title('mean waiting time per episode')
plt.legend()
# plt.show()
file_path = folder_path + '/waiting_time_train_ma.png'
fig.savefig(file_path)

# 待ち時間をdfで記録し，csvに出力
df_wt_train = pd.DataFrame()
df_wt_train['dqn'] = y_dqn
df_wt_train['wtp'] = list(mean_waiting_times_wtp.values())
df_wt_train['cp'] = list(mean_waiting_times_cp.values())
df_wt_train['rnd'] = list(mean_waiting_times_rnd.values())

df_wt_train.to_csv(folder_path + '/wt_train.csv')

# df_dqn_train['waiting_time'] = y_dqn
# df_wtp_train['waitng_time'] = mean_waiting_times_wtp
# df_cp_train['waiting_time'] = mean_waiting_times_cp
# df_rnd_train['waiting_time'] = mean_waiting_times_rnd


# 各エピソードのコスト

num = 5  # 移動平均の個数
b = np.ones(num) / num

mean_costs_dqn = {}
for k, v in episode_logger_dqn.costs.items():
    mean_costs_dqn[k] = np.mean(v)

x_dqn = mean_costs_dqn.keys()
y_dqn = mean_costs_dqn.values()
y_dqn_ma = np.convolve(list(y_dqn), b, mode='same')  # 移動平均

mean_costs_wtp = {}
for k, v in episode_logger_wtp.costs.items():
    mean_costs_wtp[k] = np.mean(v)
    if len(mean_costs_dqn) == len(mean_costs_wtp):
        break

x_wtp = x_dqn
y_wtp = [np.mean(list(mean_costs_wtp.values()))] * len(x_wtp)

mean_costs_cp = {}
for k, v in episode_logger_cp.costs.items():
    mean_costs_cp[k] = np.mean(v)
    if len(mean_costs_dqn) == len(mean_costs_cp):
        break

x_cp = x_dqn
y_cp = [np.mean(list(mean_costs_cp.values()))] * len(x_cp)

mean_costs_rnd = {}
for k, v in episode_logger_rnd.costs.items():
    mean_costs_rnd[k] = np.mean(v)
    if len(mean_costs_dqn) == len(mean_costs_rnd):
        break

x_rnd = x_dqn
y_rnd = [np.mean(list(mean_costs_rnd.values()))] * len(x_rnd)

fig = plt.figure()
plt.xlabel('episode')
plt.ylabel('mean cost')
plt.plot(x_dqn, y_dqn, label='dqn(raw)', color='#1f77b4', alpha=0.3)
plt.plot(x_dqn, y_dqn_ma, label='dqn(smoothed)', color='#1f77b4')
plt.plot(x_wtp, y_wtp, label='wtp(average)', color='#ff7f0e', linestyle='dashed')
plt.plot(x_cp, y_cp, label='cp(average)', color='#2ca02c', linestyle='dashdot')
plt.plot(x_rnd, y_rnd, label='random(average)', color='#d62728', linestyle='dotted')

# plt.title('mean cost per episode')
plt.legend()
# plt.show()
file_path = folder_path + '/cost_train_ma.png'
fig.savefig(file_path)

# コストをdfで記録し，csvに出力
df_cost_train = pd.DataFrame()
df_cost_train['dqn'] = y_dqn
df_cost_train['wtp'] = list(mean_costs_wtp.values())
df_cost_train['cp'] = list(mean_costs_cp.values())
df_cost_train['rnd'] = list(mean_costs_rnd.values())

df_cost_train.to_csv(folder_path + '/cost_train.csv')

# モデルを保存
model_path = folder_path + '/model'
try:
    dqn.model.save(model_path)
    print('saved model to {}'.format(model_path))
except:
    print('failed to save model')


# テスト

model = load_model(model_path)

dqn = MyAgent(
    model=model, nb_actions=n_action, memory=memory, nb_steps_warmup=50,
    target_model_update=1e-2, policy=EGQpolicy, n_window=n_window,
    n_on_premise_node=n_on_premise_node, n_cloud_node=n_cloud_node,
    n_job_queue_obs=n_job_queue_obs, n_job_queue_bck=n_job_queue_bck
)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# 記録用
episode_logger_dqn_test = EpisodeLogger_4_dqn_test()
episode_logger_wtp_test = EpisodeLogger()
episode_logger_cp_test = EpisodeLogger()
episode_logger_rnd_test = EpisodeLogger()

# テスト
history = dqn.test(env, nb_episodes=10, visualize=False, callbacks=[episode_logger_dqn_test])
history = wtp.test(env, nb_episodes=10, visualize=False, callbacks=[episode_logger_wtp_test])
history = cp.test(env, nb_episodes=10, visualize=False, callbacks=[episode_logger_cp_test])
history = rnd.test(env, nb_episodes=10, visualize=False, callbacks=[episode_logger_rnd_test])


# matplotlibの日本語設定
# from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

df_res_test = pd.DataFrame(index=['dqn', 'wtp', 'cp'])

# 各エピソードの平均報酬

mean_rewards_dqn = {}
for k, v in episode_logger_dqn_test.rewards.items():
    mean_rewards_dqn[k] = np.mean(v)

x_dqn = mean_rewards_dqn.keys()
y_dqn = mean_rewards_dqn.values()

# dqnの実質のreward(無効なactionの時の報酬を除外)
mean_rewards_dqn_substantial = {}
for episode in episode_logger_dqn_test.rewards.keys():
    actions = episode_logger_dqn_test.actions[episode]
    rewards = episode_logger_dqn_test.rewards[episode]
    substantial_rewards = []
    for a, e in zip(actions, rewards):
        if e != -penalty_invalid_action:
            # print(a,e)
            substantial_rewards.append(e)
    mean_rewards_dqn_substantial[episode] = np.mean(substantial_rewards)

x_dqn_substantial = mean_rewards_dqn_substantial.keys()
y_dqn_substantial = mean_rewards_dqn_substantial.values()

mean_rewards_wtp = {}
for k, v in episode_logger_wtp_test.rewards.items():
    mean_rewards_wtp[k] = np.mean(v)

x_wtp = x_dqn
y_wtp = mean_rewards_wtp.values()

mean_rewards_cp = {}
for k, v in episode_logger_cp_test.rewards.items():
    mean_rewards_cp[k] = np.mean(v)

x_cp = x_dqn
y_cp = mean_rewards_cp.values()

mean_rewards_rnd = {}
for k, v in episode_logger_rnd_test.rewards.items():
    mean_rewards_rnd[k] = np.mean(v)

x_rnd = x_dqn
y_rnd = mean_rewards_rnd.values()

# 全エピソードでの平均報酬
x = np.arange(3) + 1
# mean_dqn_substantial = np.mean(list(y_dqn_substantial))
mean_dqn = np.mean(list(y_dqn))
mean_wtp = np.mean(list(y_wtp))
mean_cp = np.mean(list(y_cp))
# mean_rnd = np.mean(list(y_rnd))
y = [mean_dqn, mean_wtp, mean_cp]
label = ['dqn', 'wtp', 'cp']
fig = plt.figure()
plt.ylabel('mean reward')
plt.bar(x, y, width=0.3, tick_label=label, align='center')
# plt.title('mean reward')
# plt.show()
file_path = folder_path + '/mean_reward_test.png'
fig.savefig(file_path, dpi=300)
# print(mean_dqn, mean_wtp, mean_cp)
df_res_test['reward'] = y


# 各エピソードの待ち時間
mean_waiting_times_dqn = {}
for k, v in episode_logger_dqn_test.waiting_times.items():
    if v:  # 最後のエピソードは途中で打ち切られ，waiting_timeを記録できていないことがあるからあるものだけで
        mean_waiting_times_dqn[k] = np.mean(v)
x_dqn = mean_waiting_times_dqn.keys()
y_dqn = mean_waiting_times_dqn.values()

mean_waiting_times_wtp = {}
for k, v in episode_logger_wtp_test.waiting_times.items():
    if v:  # 最後のエピソードは途中で打ち切られ，waiting_timeを記録できていないことがあるからあるものだけで
        mean_waiting_times_wtp[k] = np.mean(v)

x_wtp = x_dqn
y_wtp = mean_waiting_times_wtp.values()

mean_waiting_times_cp = {}
for k, v in episode_logger_cp_test.waiting_times.items():
    if v:  # 最後のエピソードは途中で打ち切られ，waiting_timeを記録できていないことがあるからあるものだけで
        mean_waiting_times_cp[k] = np.mean(v)

x_cp = x_dqn
y_cp = mean_waiting_times_cp.values()

mean_waiting_times_rnd = {}
for k, v in episode_logger_rnd_test.waiting_times.items():
    if v:  # 最後のエピソードは途中で打ち切られ，waiting_timeを記録できていないことがあるからあるものだけで
        mean_waiting_times_rnd[k] = np.mean(v)

x_rnd = x_dqn
y_rnd = mean_waiting_times_rnd.values()

# 全エピソードでの平均待ち時間
x = np.arange(3) + 1
mean_dqn = np.mean(list(y_dqn))
mean_wtp = np.mean(list(y_wtp))
mean_cp = np.mean(list(y_cp))
# mean_rnd = np.mean(list(y_rnd))
y = [mean_dqn, mean_wtp, mean_cp]
label = ['dqn', 'wtp', 'cp']
fig = plt.figure()
plt.ylabel('mean waiting time')
plt.bar(x, y, width=0.3, tick_label=label, align='center')
# plt.title('mean waiting time')
# plt.show()
file_path = folder_path + '/mean_waiting_time_test.png'
fig.savefig(file_path)
# print(mean_dqn, mean_wtp, mean_cp)
df_res_test['waiting time'] = y


# 各エピソードのコスト
mean_costs_dqn = {}
for k, v in episode_logger_dqn_test.costs.items():
    mean_costs_dqn[k] = np.mean(v)

x_dqn = mean_costs_dqn.keys()
y_dqn = mean_costs_dqn.values()

mean_costs_wtp = {}
for k, v in episode_logger_wtp_test.costs.items():
    mean_costs_wtp[k] = np.mean(v)

x_wtp = x_dqn
y_wtp = mean_costs_wtp.values()

mean_costs_cp = {}
for k, v in episode_logger_cp_test.costs.items():
    mean_costs_cp[k] = np.mean(v)

x_cp = x_dqn
y_cp = mean_costs_cp.values()

mean_costs_rnd = {}
for k, v in episode_logger_rnd_test.costs.items():
    mean_costs_rnd[k] = np.mean(v)

x_rnd = x_dqn
y_rnd = mean_costs_rnd.values()

# 全エピソードでの平均コスト
x = np.arange(3) + 1
mean_dqn = np.mean(list(y_dqn))
mean_wtp = np.mean(list(y_wtp))
mean_cp = np.mean(list(y_cp))
# mean_rnd = np.mean(list(y_rnd))
y = [mean_dqn, mean_wtp, mean_cp]
label = ['dqn', 'wtp', 'cp']
fig = plt.figure()
plt.ylabel('mean cost')
plt.bar(x, y, width=0.3, tick_label=label, align='center')
# plt.title('mean cost')
# plt.show()
file_path = folder_path + '/mean_cost_test.png'
fig.savefig(file_path)
# print(mean_dqn, mean_wtp, mean_cp)
df_res_test['cost'] = y

df_res_test.to_csv(folder_path + '/result_test.csv')
