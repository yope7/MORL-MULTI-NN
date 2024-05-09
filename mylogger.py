import numpy as np
import rl.callbacks


# dqnの学習の記録用
class EpisodeLogger_4_dqn(rl.callbacks.Callback):
    def __init__(self):
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.costs = {}
        self.times = {}
        self.waiting_times = {}
        self.end_times = {}
        self.jobs = {}
        self.q_values = {}
        self.loss = {}

    def on_episode_begin(self, episode, logs):
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.costs[episode] = []
        self.times[episode] = []
        self.end_times[episode] = []
        self.waiting_times[episode] = []
        self.q_values[episode] = []
        self.loss[episode] = []

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        if logs['cost'] != -1:  # そのステップでのactionが「割り当てない」でなかった，またはフィルター無しのdqnagentが無効なactionを取った場合
            self.costs[episode].append(logs['cost'])
        self.times[episode].append(logs['time'])
        self.q_values[episode].append(logs['q_values'])
        self.loss[episode].append(logs['metrics'][0])

    # 提出時間を迎えていたのにジョブキューに格納されなかったジョブ
    def on_episode_end(self, episode, logs):
        end_time = logs['end_time']
        self.end_times[episode] = end_time
        jobs = logs['jobs']
        self.jobs[episode] = jobs

        i = 0
        #         print('i: ' + str(i))
        #         print('logs[jobs][i][-1]: ' + str(logs['jobs'][i][-1]))
        #         print('time: ' + str(time))
        #         print('logs[jobs][i][0] <= time' + str(logs['jobs'][i][0] <= time))
        #         print('n_jobs: ' + str(len(logs['jobs'])))
        while True:
            #             print(i)
            if i < len(jobs):  # そのエピソードで生成されたジョブの中でまだ見ていないジョブがある場合
                submitted_time = jobs[i][0]
                if submitted_time <= end_time:  # ジョブが提出時刻を迎えていた場合
                    waiting_time = jobs[i][-1]
                    if waiting_time == -1:  # 割り当てられていなかった場合
                        waiting_time = end_time + 10 - submitted_time  # 待ち時間を大きめに設定
                    self.waiting_times[episode].append(waiting_time)  # 待ち時間を記録
                    i += 1
                else:  # ジョブが提出時刻を迎えていなかった場合
                    break
            else:  # そのエピソードで生成されたジョブを全て見た場合
                break
        #             print('i: ' + str(i))

        self.loss[episode] = np.mean(self.loss[episode])

    def on_train_end(self, logs=None):
        # 訓練の最後のエピソードではだいたいon_episode_endが呼ばれず平均化されていないのでここで平均化
        self.loss[max(self.loss.keys())] = np.mean(self.loss[max(self.loss.keys())])
        # waiting_timesの最後もここでやればいいかも


# dqnのテストの記録用
class EpisodeLogger_4_dqn_test(rl.callbacks.Callback):
    def __init__(self):
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.costs = {}
        self.times = {}
        self.waiting_times = {}
        self.end_times = {}
        self.jobs = {}
        self.q_values = {}

    def on_episode_begin(self, episode, logs):
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.costs[episode] = []
        self.times[episode] = []
        self.end_times[episode] = []
        self.waiting_times[episode] = []
        self.q_values[episode] = []

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        if logs['cost'] != -1:  # そのステップでのactionが「割り当てない」でなかった，またはフィルター無しのdqnagqntが無効なactionを取った場合
            self.costs[episode].append(logs['cost'])
        self.times[episode].append(logs['time'])
        self.q_values[episode].append(logs['q_values'])

    # 提出時間を迎えていたのにジョブキューに格納されなかったジョブ
    def on_episode_end(self, episode, logs):
        end_time = logs['end_time']
        self.end_times[episode] = end_time
        jobs = logs['jobs']
        self.jobs[episode] = jobs

        i = 0
        #         print('i: ' + str(i))
        #         print('logs[jobs][i][-1]: ' + str(logs['jobs'][i][-1]))
        #         print('time: ' + str(time))
        #         print('logs[jobs][i][0] <= time' + str(logs['jobs'][i][0] <= time))
        #         print('n_jobs: ' + str(len(logs['jobs'])))
        while True:
            #             print(i)
            if i < len(jobs):  # そのエピソードで生成されたジョブの中でまだ見ていないジョブがある場合
                submitted_time = jobs[i][0]
                if submitted_time <= end_time:  # ジョブが提出時刻を迎えていた場合
                    waiting_time = jobs[i][-1]
                    if waiting_time == -1:  # 割り当てられていなかった場合
                        waiting_time = end_time + 10 - submitted_time  # 待ち時間を大きめに設定
                    self.waiting_times[episode].append(waiting_time)  # 待ち時間を記録
                    i += 1
                else:  # ジョブが提出時刻を迎えていなかった場合
                    break
            else:  # そのエピソードで生成されたジョブを全て見た場合
                break
            # print('i: ' + str(i))


# ヒューリスティックの記録用

class EpisodeLogger(rl.callbacks.Callback):
    def __init__(self):
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.costs = {}
        self.times = {}
        self.waiting_times = {}
        self.end_times = {}
        self.jobs = {}

    def on_episode_begin(self, episode, logs):
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.costs[episode] = []
        self.times[episode] = []
        self.end_times[episode] = []
        self.waiting_times[episode] = []

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        if logs['cost'] != -1:  # そのステップでのactionが「割り当てない」でなかった，またはフィルター無しのdqnagqntが無効なactionを取った場合
            self.costs[episode].append(logs['cost'])
        self.times[episode].append(logs['time'])

    # 提出時間を迎えていたのにジョブキューに格納されなかったジョブ
    def on_episode_end(self, episode, logs):
        end_time = logs['end_time']
        self.end_times[episode] = end_time
        jobs = logs['jobs']
        self.jobs[episode] = jobs

        i = 0
        #         print('i: ' + str(i))
        #         print('logs[jobs][i][-1]: ' + str(logs['jobs'][i][-1]))
        #         print('time: ' + str(time))
        #         print('logs[jobs][i][0] <= time' + str(logs['jobs'][i][0] <= time))
        #         print('n_jobs: ' + str(len(logs['jobs'])))
        while True:
            #             print(i)
            if i < len(jobs):  # そのエピソードで生成されたジョブの中でまだ見ていないジョブがある場合
                submitted_time = jobs[i][0]
                if submitted_time <= end_time:  # ジョブが提出時刻を迎えていた場合
                    waiting_time = jobs[i][-1]
                    if waiting_time == -1:  # 割り当てられていなかった場合
                        waiting_time = end_time + 10 - submitted_time  # 待ち時間を大きめに設定
                    self.waiting_times[episode].append(waiting_time)  # 待ち時間を記録
                    i += 1
                else:  # ジョブが提出時刻を迎えていなかった場合
                    break
            else:  # そのエピソードで生成されたジョブを全て見た場合
                break
            # print('i: ' + str(i))
