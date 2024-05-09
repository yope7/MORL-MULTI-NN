import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 300
num = 5  # 移動平均の個数
b = np.ones(num)/num

folder_name = 'njob1-2-3_6msteps'
df = pd.read_csv(folder_name + '/wt_train.csv')
df_prm_job = pd.read_csv(folder_name + '/parameter_job.csv')
episode_switch_job_env = [df_prm_job['point_switch_job_type1'][0], df_prm_job['point_switch_job_type2'][0]]

# df = df[start:]
x = list(range(len(df)))
y_dqn_ma = np.convolve(df['dqn'], b, mode='same')  # 移動平均

# wtp
y_wtp = []
start = 0
for i in range(3):
    if i == 2:
        end = len(x)
        y_wtp = y_wtp + [np.mean(list(df['wtp'][start:end]))] * (end - start)
    else:
        end = episode_switch_job_env[i]
        y_wtp = y_wtp + [np.mean(list(df['wtp'][start:end]))] * (end - start)
        start = end
# cp
y_cp = []
start = 0
for i in range(3):
    if i == 2:
        end = len(x)
        y_cp = y_cp + [np.mean(list(df['cp'][start:end]))] * (end - start)
    else:
        end = episode_switch_job_env[i]
        y_cp = y_cp + [np.mean(list(df['cp'][start:end]))] * (end - start)
        start = end
# rnd
y_rnd = []
start = 0
for i in range(3):
    if i == 2:
        end = len(x)
        y_rnd = y_rnd + [np.mean(list(df['rnd'][start:end]))] * (end - start)
    else:
        end = episode_switch_job_env[i]
        y_rnd = y_rnd + [np.mean(list(df['rnd'][start:end]))] * (end - start)
        start = end

fig = plt.figure()
plt.xlabel('episode')
plt.ylabel('mean waiting time')
plt.plot(x, df['dqn'], label='dqn(raw)', color='#1f77b4', alpha=0.3)
plt.plot(x, y_dqn_ma, label='dqn(smoothed)', color='#1f77b4')
plt.plot(x, y_wtp, label='wtp(average)', color='#ff7f0e', linestyle='dashed')
plt.plot(x, y_cp, label='cp(average)', color='#2ca02c', linestyle='dashdot')
plt.plot(x, y_rnd, label='random(average)', color='#d62728', linestyle='dotted')
# plt.title('mean waiting time per episode')
plt.legend()
plt.show()
fig.savefig(folder_name + '/adaptation_wt.png')
