import gym
import numpy as np
from collections import deque
import sys
from sklearn.preprocessing import MinMaxScaler
import csv

# 学習環境
class SchedulingEnv(gym.core.Env): 
    def __init__(self, max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck, weight_wt,
                 weight_cost, penalty_not_allocate, penalty_invalid_action, multi_algorithm, jobs_set,
                 job_type=0):
        self.step_count = 0  # 現在のステップ数(今何ステップ目かを示す)
        # self.n_job_per_time = self.config['param_job']['n_job_per_time']
        self.episode = 0  # 現在のエピソード(今何エピソード目かを示す); agentに教えてもらう
        self.time = 0  # 時刻(ジョブ到着の判定に使う)
        self.max_step = max_step  # ステップの最大数(1エピソードの終了時刻)
        self.index_next_job = 0  # 次に待っている新しいジョブのインデックス 新しいジョブをジョブキューに追加するときに使う
        # self.index_next_job_ideal = 0 # 理想的な状態(処理時間を迎えたのにジョブキューがいっぱいでジョブキューに格納されていないジョブがない)であれば次に待っている新しいジョブのインデックス
        self.n_window = n_window  # スライドウィンドウの横幅
        self.n_on_premise_node = n_on_premise_node  # オンプレミス計算資源のノード数
        self.on_premise_window_history = np.zeros((6,1)) # オンプレミスのスライドウィンドウの履歴
        self.on_premise_window_user_history = np.zeros((6,1)) # オンプレミスのスライドウィンドウの履歴
        self.cloud_window_user_history_show = np.zeros((6,1)) # クラウドのスライドウィンドウの履歴
        self.on_premise_window_user_history_show = np.zeros((6,1)) # オンプレミスのスライドウィンドウの履歴
        self.cloud_window_history = np.zeros((6,1)) # クラウドのスライドウィンドウの履歴
        self.cloud_window_user_history = np.zeros((6,1)) # クラウドのスライドウィンドウの履歴
        self.n_cloud_node = n_cloud_node  # クラウド計算資源のノード数
        self.n_job_queue_obs = n_job_queue_obs  # ジョブキューの観測部分の長さ
        self.n_job_queue_bck = n_job_queue_bck  # ジョブキューのバックログ部分の長さ
        self.rear_job_queue = 0  # ジョブキューの末尾 (== 0: ジョブキューが空)
        self.weight_wt = weight_wt  # 報酬における待ち時間の重み
        self.weight_cost = weight_cost  # 報酬におけるコストの重み
        self.penalty_not_allocate = penalty_not_allocate  # 割り当てない(一時キューに格納する)という行動を選択した際のペナルティー
        self.penalty_invalid_action = penalty_invalid_action  # actionが無効だった場合のペナルティー
        self.on_premise_window = np.zeros((self.n_on_premise_node, self.n_window))  # オンプレミスのスライドウィンドウ
        self.cloud_window = np.zeros((self.n_cloud_node, self.n_window))  # クラウドのスライドウィンドウ

        self.n_action = 3  # 行動数
        self.action_space = gym.spaces.Discrete(self.n_action)  # 行動空間
        self.observation_space = gym.spaces.Discrete(
            self.n_on_premise_node * self.n_window + self.n_cloud_node * self.n_window + 4 * self.n_job_queue_obs + 1)  # 観測(状態)空間
        self.tmp_queue = deque()  # 割り当てられなかったジョブを一時的に格納するキュー
        self.mm = MinMaxScaler()  # 観測データを標準化するやつ
        self.multi_algorithm = multi_algorithm  # 複数アルゴリズムで一気に実行しており，各アルゴリズムでジョブが同じかどうか; ジョブ生成に関わる
        if self.multi_algorithm:  # クラス宣言前に既にジョブセットを定義している場合
            self.job_type = job_type  # ジョブタイプは既に決まっている

        # アルゴリズムごとに同じジョブにする場合(複数アルゴリズムで一気に実行している場合)は環境定義の前にジョブをすでに生成してあるのでそれをエピソード最初に読み取るだけ

        if self.multi_algorithm:  # アルゴリズムごとに同じジョブにする場合
            self.jobs_set = jobs_set  # 事前に生成したジョブセットを受け取る
            #jobをcsvで保存．savetxt
            #配列をファイルにjsonで保存，delimiterで改行文字を指定
            # print("jobs_set_head:\n",self.jobs_set[0])
            

            
        else:  # アルゴリズムごとに同じジョブではない場合
            # ジョブ設定
            # ジョブをエピソードごとに変えるか，固定にするかを指定
            self.job_is_constant = int(input('Is job constant? \n 0:not constant 1:constant \n >'))
            # ジョブタイプを指定
            if self.job_is_constant:
                self.job_type = int(input('select job-type \n 1:default 2:input file 3:random \n >'))
            else:
                self.job_type = int(input(
                    'select job-type \n 1:default 2:input file 3:random 4:random(エピソードごとにジョブ数やジョブサイズの範囲を変える) \n >'))

            # ランダムの場合，必要な変数を指定
            if self.job_type == 3:
                # 最大時間を入力
                self.max_t = int(input('input max time \n >'))
                # 単位時間あたりのジョブ数を入力
                while True:
                    self.n_job_per_time = int(input('input number of jobs per time (-1:random) \n >'))
                    self.n_job_per_time_is_random = False
                    if 1 <= self.n_job_per_time <= 10:
                        break
                    elif self.n_job_per_time == -1:
                        self.n_job_per_time_is_random = True
                        break
                    else:
                        print('input again')

            # 途中で性質が切り替わる場合の設定
            if self.job_type == 4:
                self.random_job_type = int(input(
                    'select random job type \n 1:a few small jobs -> many large jobs 2:a few small jobs -> a few large jobs 3:a few small jobs -> many small jobs \n >'))
                self.point_switch_random_job_type = int(input('ジョブの性質が変わるエピソードを入力 \n >'))
                # 最大時間を入力
                self.max_t = int(input('input max time \n >'))

            # ジョブが固定の場合，ここでジョブを設定してしまう
            if self.job_is_constant:
                self.jobs = self.get_jobs()

        # デバッグ用
        self.count_failed_to_allocating = 0  # 割り当て失敗回数
        self.job_queue = np.zeros((len(self.jobs_set),8))

    # 各ステップで実行される操作
    def step(self, action_raw):
        time_reward_new = 0
        # print('start step')
        fairness = 0
        self.user_wt = [[i,-100] for i in range(100)]

        # print('job\n', self.job_queue[:5,:])


        # 時刻
        time = self.time

        allocated_job = self.job_queue[0]

        is_fifo =-1

        



        action = self.get_converted_action(action_raw)   
        # print('action: ' + str(action))

        is_valid = False  # actionが有効かどうか
        is_valid, wt_real = self.check_is_valid_action(action)

        # print("is_valid: ",is_valid)

        if is_valid == False:  # すきまがない場合or job_queueが空の場合
            # print("allocated_job: ",allocated_job)
            if np.all(allocated_job == 0):  # ジョブキューが空の場合
                job_none = True
            else:  # ジョブキューが空でない場合
                job_none = False
            # print('is_valid: ' + str(is_valid))
            self.time_transition()

            var_reward = 0
            var_after = 0
            wt_step = -1
            std_mean_before = 0
            std_mean_after = 0
            std_reward = 0
            
            
            # print('time shift',self.time)
            # print("force timeshift")

            # print('allocated_job is zero')

        else:
            job_none = False
            # print('valid action')

            #配列の各要素における0番目の値の分散を計算
            # print("self.job_allocated: ",self.job_allocated)
            if self.job_allocated == []:
                var_before = 0
                std_mean_before = 0
            else:
                # print("sums_user_wt: ",sums_user_wt)
                # print("self.user_wt_sum: ",self.user_wt_sum)
                if np.mean(self.user_wt_sum) == 0:
                    var_before = 0
                else:
                    var_before = np.std(self.user_wt_sum)
                # print("var_before: ",var_before)
                std_mean_before = self.get_user_wt_std(self.user_wt_log)

                if action[0] == 0:
                    if action[1] == 0:
                        hue,wt_parallel = self.check_is_valid_action([0,1])
                    if action[1] == 1:
                        hue,wt_parallel = self.check_is_valid_action([0,1])

                if action[0] == 1:
                    if action[1] == 0:
                        hue,wt_parallel = self.check_is_valid_action([1,1])
                    if action[1] == 1:
                        hue,wt_parallel = self.check_is_valid_action([1,0])
                
                if wt_parallel == -1:
                    time_reward_new = 1
                else:
                    # print("wt_real: ",wt_real,"wt_parallel: ",wt_parallel)
                    if wt_real < wt_parallel:
                        time_reward_new = 1
                    elif wt_real == wt_parallel:
                        time_reward_new = 0
                    else:
                        time_reward_new = -1
                    # print("time_reward_new: ",time_reward_new)





            # print("var_before: ",var_before)
            if action[0] == 0:  # FIFO
                # print('job_queue before fio\n',self.job_queue)
                wt_step = self.schedule(action)
                # print('self.priority: ',self.user_priority)
                # print('schedule fifo')
                            #ジョブキューをスライド
                self.job_queue = np.roll(self.job_queue, -1, axis=0)
                self.job_queue[-1] = 0
                self.rear_job_queue -= 1

                is_fifo = 1

                # print('job_queue after fio\n',self.job_queue)


            elif action[0] ==1:  # 借金
                # print("fairness")

                wt_step, target_job = self.schedule_for_fairness(action)
                # print("self.on_premise_window_user:\n",self.on_premise_window_user)
                # print('self.priority: ',self.user_priority)

                # print('job_queue before f\n',self.job_queue)
                # print("self.rear_job_queue: ",self.rear_job_queue)

                self.job_queue = np.delete(self.job_queue, target_job,axis = 0)
                self.job_queue = np.vstack((self.job_queue,np.zeros((1,len(self.jobs[0])))))
                self.rear_job_queue -= 1

                # print('job_queue after f',self.job_queue)
                # print("self.rear_job_queue: ",self.rear_job_queue)

                is_fifo = 0

                fairness = 1

            elif action[0] == 2:
                # print('schedule_min')
                wt_step, target_job = self.schedule_min(action)

                self.job_queue = np.delete(self.job_queue, target_job,axis = 0)
                self.job_queue = np.vstack((self.job_queue,np.zeros((1,len(self.jobs[0])))))

                is_fifo = 0 

            else:
                pass
                wt_step, target_job = self.schedule_max(action)

                self.job_queue = np.delete(self.job_queue, target_job,axis = 0)
                self.job_queue = np.vstack((self.job_queue,np.zeros((1,len(self.jobs[0])))))

                is_fifo = 0


                # wt_step = self.schedule(action)
            # print("wt_step: ",wt_step)
            # print("sums_user_wt: ",sums_user_wt)
            if np.mean(self.user_wt_sum) == 0:
                var_after = 0
            else:
                var_after = np.std(self.user_wt_sum)

            std_mean_after = self.get_user_wt_std(self.user_wt_log)


            # print('self.user_wt_sum: ',self.user_wt_sum)
            # print("var_after: ",var_after)

            #配列の各要素における0番目の値の分散を計算

            # print("var_after: ",var_after)

            if var_before > var_after:
                var_reward = 1
            elif var_before == var_after:
                var_reward = 0
            else:
                var_reward = 0

            if std_mean_before > std_mean_after:
                std_reward = 1
            elif std_mean_before == std_mean_after:
                std_reward = 0
            else:
                std_reward = 0
            # print('std_before: ',std_mean_before)
            # print('std_after: ',std_mean_after)

            # var_reward = var_before - var_after

            is_valid = True
            # print('schedule')


                

        # 観測データ(状態)を取得
        observation = self.get_observation()
        # 報酬を取得
        reward = self.get_reward(action, allocated_job, time, is_valid, job_none)
        # コストを取得
        cost = self.compute_cost(action, allocated_job, is_valid)

        # エピソードの終了時刻を満たしているかの判定
        done = self.check_is_done()
        # print("done: ",done)

        info = {}

        time = self.time

        # print("time: ",time)
        #     print('================================================')
        # print('self.user_wt: ',self.user_wt)

        user_wt = self.user_wt
        user_wt_sum = self.user_wt_sum
        user_wt_log = self.user_wt_log



        #         return observation, reward, cost, time, waiting_time, done, info
        return observation, reward, cost, time, done, info, var_reward,std_reward, var_after, wt_step, fairness, user_wt, user_wt_sum,is_fifo,user_wt_log,time_reward_new


    def get_user_wt_std(self,data):
        users = {i: [] for i in range(10)}
        users_mean = []

        # print(data)

        # データをイテレートして、適切なユーザーリストに追
        if np.all(data == 0):
            std_mean = 0
        
        else:
            for column in data:
                user_id = column[1]
                if user_id in users:
                    users[user_id].append(column[2])
            # print(users)
            for column in users:
                if column:
                    users_mean.append(np.mean(users[column]))
                # print(users_mean)
            std_mean = np.std(users_mean)

        return std_mean
    # スカラーのactionをリストに変換
    def get_converted_action(self, a):
        if a == 0:
            method = 0
            use_cloud = 0
        elif a == 1:
            method = 0
            use_cloud = 1
        
        elif a == 2:
            method = 1
            use_cloud = 0

        elif a == 3:
            method = 1
            use_cloud = 1

        elif a == 4:
            method = 2
            use_cloud = 0

        elif a == 5:
            method = 2
            use_cloud = 1

        elif a == 6:
            method = 3
            use_cloud = 0
        
        elif a == 7:
            method = 3
            use_cloud = 1

        else:
            print('a is invalid')
            exit()
        action = [method, use_cloud]

        return action

    # 初期化
    # 各エピソードの最初に呼び出される
    def reset(self):
        # エピソードを1進める
        #         self.episode += 1

        # 変数を初期化
        self.time = 0
        self.sums_user =[]
        self.job_allocated = []
        self.step_count = 0
        if self.multi_algorithm:  # アルゴリズムごとに同じジョブである場合
            self.jobs = self.jobs_set[self.episode]
        else:  # アルゴリズムごとに同じジョブではない場合
            # ジョブが固定でない(ジョブをエピソードごとに変える)場合，ジョブを再設定
            if not self.job_is_constant:
                self.jobs = self.get_jobs()  # ジョブを設定
        self.max_t = self.jobs[-1][0]  # 最大時間
        self.index_next_job = 0  # 新しいジョブをジョブキューに追加するときに使う
        self.on_premise_window = np.zeros((self.n_on_premise_node, self.n_window))  # オンプレミスのスライドウィンドウ
        self.on_premise_window_user = np.zeros((self.n_on_premise_node, self.n_window))  # オンプレミスのスライドウィンドウ
        self.cloud_window = np.zeros((self.n_cloud_node, self.n_window))  # クラウドのスライドウィンドウ
        self.cloud_window_user = np.zeros((self.n_cloud_node, self.n_window))  # クラウドのスライドウィンドウ
        self.job_queue = np.zeros((len(self.jobs),8)) # ジョブキュー
        self.rear_job_queue = 0  # ジョブキューの末尾 (== 0: ジョブキューが空)
        self.tmp_queue = deque()  # 割り当てられなかったジョブを一時的に格納するキュー
        self.user_wt_log = []


        # ジョブキューに新しいジョブを追加
        # print("self.job_queue: ",self.job_queue)
        self.append_new_job2job_queue()
        # print("add job to job queue")
        # print("self.job_queue: ",self.job_queue)
        # 観測データ(状態)を取得
        observation = self.get_observation()

        self.user_priority = [[i, -1] for i in range(100)]
        self.user_wt = [[i,0] for i in range(100)]
        self.user_wt_sum = [[i,0] for i in range(100)]

        # print('-------------reseted-------------')
        # print('job',self.jobs)
        # exit()

        return observation

    # ジョブを生成
    def get_jobs(self):
        if self.multi_algorithm:  # アルゴリズムごとに同じジョブにする場合
            self.jobs = self.jobs_set[self.episode]
        else:  # アルゴリズムごとに同じジョブではない場合
            # デフォルト
            if self.job_type == 1:
                jobs = np.array(
                    [  # [submit_time, processing_time, required_nodes_num, can_use_cloud, job_id, waiting_time(初期値=-1)]
                        [0, 2, 1, 1, 0, -1],
                        [0, 1, 5, 0, 1, -1],
                        [1, 6, 3, 0, 2, -1],
                        [2, 4, 2, 1, 3, -1],
                        [3, 8, 7, 1, 4, -1],
                        [3, 6, 5, 0, 5, -1],
                        [3, 4, 4, 1, 6, -1],
                        [4, 5, 9, 1, 7, -1],
                        [5, 10, 4, 1, 8, -1],
                        [5, 2, 3, 1, 9, -1],
                        [6, 3, 1, 1, 10, -1],
                        [7, 4, 6, 1, 11, -1],
                        [8, 3, 7, 1, 12, -1],
                        [9, 5, 7, 1, 13, -1]
                    ])

            elif self.job_type == 2:
                pass
                # jobs = pd.read_csv('jobs.csv')
                # jobs = jobs.values.tolist()

            # ランダム
            elif self.job_type == 3:
                jobs = []

                job_id = 0
                waiting_time = -1  # 待ち時間は初期値の-1で固定
                for i in range(self.max_t + 1):
                    if self.n_job_per_time_is_random:
                        self.n_job_per_time = np.random.randint(1, 6)
                    for _ in range(self.n_job_per_time):
                        submit_time = i

                        # 通常サイズのジョブ(デフォルト仕様)
                        processing_time = np.random.randint(1, (self.n_window + 1) // 2)
                        n_required_nodes = np.random.randint(1,
                                                             (max(self.n_on_premise_node, self.n_cloud_node) + 1) // 2)
                        # 大きめのジョブ
                        #                     processing_time = np.random.randint(1, (self.n_window+1)//4*3)
                        #                     n_required_nodes = np.random.randint(1, (max(self.n_on_premise_node, self.n_cloud_node)+1)//4*3)

                        can_use_cloud = np.random.randint(0, 2)
                        jobs.append(
                            [submit_time, processing_time, n_required_nodes, can_use_cloud, job_id, waiting_time])
                        job_id += 1

            elif self.job_type == 4:
                jobs = []

                job_id = 0
                waiting_time = -1  # 待ち時間は初期値の-1で固定

                # 小少->
                if self.random_job_type == 1:
                    if self.time < self.point_switch_random_job_type:
                        self.n_job_per_time = np.random.randint(1, (
                                (self.n_job_queue_obs + self.n_job_queue_bck) * 3) // 10)
                    else:
                        self.n_job_per_time = np.random.randint(1, (
                                (self.n_job_queue_obs + self.n_job_queue_bck) * 3) // 10)

                elif self.random_job_type == 2:
                    pass
                # 小少 -> 小多
                elif self.random_job_type == 3:
                    if self.episode < self.point_switch_random_job_type:
                        self.n_job_per_time = np.random.randint(1, (
                                (self.n_job_queue_obs + self.n_job_queue_bck) * 3) // 10)
                    else:
                        self.n_job_per_time = np.random.randint(
                            ((self.n_job_queue_obs + self.n_job_queue_bck) * 3) // 10,
                            self.n_job_queue_obs + self.n_job_queue_bck)

                    max_processing_time = (self.n_window // 2)
                    max_n_required_nodes = (max(self.n_on_premise_node, self.n_cloud_node) // 2)
                    # print('max_processing_time: ' + str(max_processing_time))
                    # print('max_n_required_nodes: ' + str(max_n_required_nodes))

                    for i in range(self.max_t + 1):
                        for _ in range(self.n_job_per_time):
                            submit_time = i

                            # 通常サイズのジョブ(デフォルト仕様)
                            processing_time = np.random.randint(1, max_processing_time + 1)
                            n_required_nodes = np.random.randint(1, max_n_required_nodes + 1)
                            can_use_cloud = np.random.randint(0, 2)
                            jobs.append(
                                [submit_time, processing_time, n_required_nodes, can_use_cloud, job_id, waiting_time])
                            job_id += 1
        # ジョブを出力
        #         print('jobs are below')
        #         print(jobs)
        return jobs

    def get_map(self, name):
            
            
            # print("self.cloud_window_user_history:\n",self.cloud_window_user_history)

            #配列をファイルに上書き保存，delimiterで改行文字を指定
            np.savetxt(name+"cloud.csv", self.cloud_window_user,fmt='%d')
            np.savetxt(name+"on_premise.csv", self.on_premise_window_user,fmt='%d')

            # print("self.on_premise_window_history:\n",self.on_premise_window_history,end="]")
            # print("self.cloud_window_history:\n",self.cloud_window_history,end="]")
            # exit()
            self.reset_window_history()

    def time_transition(self):
        # 時間を1進める
        self.time += 1
        # print('時間を1進めました')
        # オンプレミス, クラウドそれぞれのスライディングウィンドウを1タイムスライス分だけスライド

        self.get_window_history_onpre()
        self.get_window_history_cloud()
        self.on_premise_window = np.roll(self.on_premise_window, -1, axis=1)
        self.on_premise_window[:, -1] = 0
        self.cloud_window = np.roll(self.cloud_window, -1, axis=1)
        self.cloud_window[:, -1] = 0

        for c in range(len(self.user_priority)):
                self.user_priority[c][1] = +1
        # print("self.jobs:\n",self.jobs)

        

        # print("self.job_queue: ",self.job_queue)
        self.append_new_job2job_queue()
        # print("add job to job queue")
        # print("self.job_queue: ",self.job_queue)
    # ウィンドウの履歴を取得
    def get_window_history_onpre(self):
            # print(self.on_premise_window[i])
        self.on_premise_window_user= np.hstack((self.on_premise_window_user,np.zeros((self.n_on_premise_node,1))))
        # print("self.on_premise_window_user:\n",self.on_premise_window_user)
        #userについても同様に履歴を取得
 
    # ウィンドウの履歴を取得
    def get_window_history_cloud(self):
        #mapの右端に0を追加
        self.cloud_window_user= np.hstack((self.cloud_window_user , np.zeros((self.n_on_premise_node,1))))
        # print("self.cloud_window_user:\n",self.cloud_window_user)  

    # エピソード終了時にウィンドウの履歴をリセット
    def reset_window_history(self):
        #(6,1)のshapeで初期化
        self.on_premise_window_history = np.zeros((6,1))
        self.cloud_window_history = np.zeros((6,1))
        self.on_premise_window_user_history = np.zeros((6,1))
        self.cloud_window_user_history = np.zeros((6,1))
    # ジョブキューに新しいジョブを追加

    def append_new_job2job_queue(self):
        for i in range(len(self.jobs)):
            # print("self.jobs:\n",self.jobs)
            # print('index_next_job: ' + str(self.index_next_job))
            # print("len(self.jobs): "+ str(len(self.jobs)))
            if self.index_next_job == len(self.jobs):  # 最後のジョブまでジョブキューに格納した場合、脱出
                # print('job_queue'  + str(self.job_queue))
                # exit()
                break
            head_job = self.jobs[self.index_next_job]  # 先頭ジョブ


            # print('time',self.time)

            if head_job[0] <= self.time:  # 先頭のジョブが到着時刻を迎えていればジョブキューに追加
                # print('in')
                # ジョブキューに格納する前に提出時間が末尾に，処理時間が先頭になるようにインデックスをずらす

                # print(self.job_queue[i][3])

                if int(self.job_queue[i][2]) == 0.0:
                    # print('in2')
                    head_job = np.roll(head_job, -1)

                    #self.job_queue[i] = head_job[1:]

                    # print(self.job_queue[i])
                    self.job_queue[i] = head_job

                    # print("self.job_queue: "+ str(self.job_queue[i]))
                    self.rear_job_queue += 1
                    self.index_next_job += 1
                # print('job_queue',self.job_queue)
        # 理想的な状態であれば次に待っている新しいジョブのインデックスを更新

    def schedule_max(self, action):
        a = 0
        k=0
        # job = self.job_queue[0]
        maxx = 100000
        priority_job = 0

        job= self.job_queue[0]




        # print("from\n",self.job_queue)
        priority_rank = sorted(self.user_priority, reverse=True, key = lambda x: x[1])
        # print('user_p',self.user_priority)
        # print(priority_rank)

        alo = 0


        for a in range(len(self.job_queue)):
            # print(priority_rank[k])
            if maxx < self.job_queue[a][0] * self.job_queue[a][1]:
                maxx = self.job_queue[a][0] * self.job_queue[a][1]
                job = self.job_queue[a]

        # print("from\n",self.job_queue)

        # print(job,"allocated")



        job_width = int(job[0])
        job_height = int(job[1])
        can_use_cloud = int(job[2])
        # print("can use: ",can_use_cloud)
        job_id = int(job[4])
        when_submitted = int(job[-1])
        when_allocate = 0
        method = action[0]
        use_cloud = action[1]

        # print("job_id: ", job_id)
        time = self.time
        # print('job', job)

        # print("weight,height"+ str(job_height) + str(job_width))
        

        # print(job, type(job_width), type(job_height), type(can_use_cloud), type(when_allocate), type(use_cloud), self.on_premise_window.shape[0])

        if not use_cloud:  # オンプレミスに割り当てる場合
            if  job_width <=  self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                allocated = False  # 暫定
                for a in range(self.n_window - job_width + 5 ):  # ウィンドウをyokoに見ていき、空いている部分行列を探す
                    if a +  job_width <= self.n_window:
                        for i in range(self.n_on_premise_node- job_height + 1):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                            part_matrix = self.on_premise_window[i:i + job_height, a:a + job_width]

                            # 高さまたは幅が足りない場合、スキップ
                            # print('on_premise_window',self.on_premise_window)
                            # print('part_matrix',part_matrix)
                            if part_matrix.shape[0] < job_height or part_matrix.shape[1] < job_width:
                                continue
                            if np.all(part_matrix == 0):
                                self.on_premise_window[i:i + job_height, a:a + job_width] = 1  # ウィンドウにジョブを割り当て
                                # print('オンプレミスの上から' + str(i + 1) + '〜' + str(i + job_height) + '番目のノードに割り当て成功')
                                # print("現在のウィンドウ状態:\n", self.on_premise_window)
                                self.on_premise_window_user[i: i + job_height, a + time: a + time + job_width] = job[3]
                                self.jobs[job_id][-1] = time - 1
                                
                                self.user_priority[int(job[3])][1]  -= job_height * job_width
                                # print('reduced', self.user_priority)
                                self.user_wt[int(job[3])][1] = time + a - when_submitted
                                self.user_wt_sum[int(job[3])][1] += time + a - when_submitted
                                # print("self.on_premise_window_user:\n",self
                                allocated = True
                                self.job_allocated.append([a,job[0],job[1],job[2],job[3],job[4],job[5]])
                                self.user_wt_log.append([int(job[4]), int(job[3]), time + a - when_submitted,job[0],job[1],job[-1],1,job[2]])

                                return time+a -when_submitted, priority_job
                
                    else :
                        # print("右にスライドしてたらジョブがスライドウィンドウの右にはみ出てしまいます")
                        break

                if not allocated:
                    print('場所がありません:checkvalid異常 at cloud==1')
                    exit()
            else:
                print('ジョブがでかすぎてスライドウィンドウの右にはみ出てしまいます:checkvalid異常 at cloud==1')
                exit()


        else:  # クラウドに割り当てる場合
            if  job_width  <= self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                allocated = False  # 暫定
                for a in range(self.n_window - job_width +5 ):  # ウィンドウをyokoに見ていき、空いている部分行列を探す
                    if a +  job_width <= self.n_window:
                        for i in range(self.n_cloud_node- job_height + 1):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                            part_matrix = self.cloud_window[i:i + job_height, a:a + job_width]
                            # print("self.cloud_window: ",self.cloud_window)
                            # print("part_matrix: ",part_matrix)


                            # 高さまたは幅が足りない場合、スキップ
                            if part_matrix.shape[0] < job_height or part_matrix.shape[1] < job_width:
                                continue
                            if np.all(part_matrix == 0):
                                self.cloud_window[i:i + job_height, a:a + job_width] = 1  # ウィンドウにジョブを割り当て
                                # print('クラウドの上から' + str(i + 1) + '〜' + str(i + job_height) + '番目のノードに割り当て成功')
                                # print("在のウィンドウ状態:\n", self.on_premise_window)
                               
                                self.cloud_window_user[i:i + job_height,time +  a:a + time + job_width] = job[3]
                                # print("self.cloud_window_user:n",self.cloud_window_user)
                                self.jobs[job_id][-1] = time - 1
                                self.user_priority[int(job[3])][1] -= job_height * job_width
                                self.user_wt[int(job[3])][1] = time + a - when_submitted
                                self.user_wt_sum[int(job[3])][1] += time + a - when_submitted
                                allocated = True
                                self.job_allocated.append([a,job[0],job[1],job[2],job[3],job[4],job[5]])
                                self.user_wt_log.append([int(job[4]), int(job[3]), time + a - when_submitted])
                                return time +1 , priority_job
                    else :
                    # print("右にスライドしてたらジョブがスライドウィンドウの右にはみ出てしまいます")
                        break
                            
                
                if not allocated:
                    print('場所がありません:checkvalid異常')
                    exit()
            else:
                print('ジョブがスライドウィンドウの右にはみ出てしまいますcheckvalid異常')
                exit()

    def schedule_min(self, action):
        a = 0
        k=0
        # job = self.job_queue[0]
        minn = - 100000
        priority_job = 0

        job= self.job_queue[0]




        # print("from\n",self.job_queue)
        priority_rank = sorted(self.user_priority, reverse=True, key = lambda x: x[1])
        # print('user_p',self.user_priority)
        # print(priority_rank)

        alo = 0


        for a in range(len(self.job_queue)):
            # print(priority_rank[k])
            if minn > self.job_queue[a][0] * self.job_queue[a][1]:
                minn = self.job_queue[a][0] * self.job_queue[a][1]
                job = self.job_queue[a]

        # print("from\n",self.job_queue)

        # print(job,"allocated")



        job_width = int(job[0])
        job_height = int(job[1])
        can_use_cloud = int(job[2])
        # print("can use: ",can_use_cloud)
        job_id = int(job[4])
        when_submitted = int(job[-1])
        when_allocate = 0
        method = action[0]
        use_cloud = action[1]

        # print("job_id: ", job_id)
        time = self.time
        # print('job', job)

        # print("weight,height"+ str(job_height) + str(job_width))
        

        # print(job, type(job_width), type(job_height), type(can_use_cloud), type(when_allocate), type(use_cloud), self.on_premise_window.shape[0])

        if not use_cloud:  # オンプレミスに割り当てる場合
            if  job_width <=  self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                allocated = False  # 暫定
                for a in range(self.n_window - job_width + 5 ):  # ウィンドウをyokoに見ていき、空いている部分行列を探す
                    if a +  job_width <= self.n_window:
                        for i in range(self.n_on_premise_node- job_height + 1):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                            part_matrix = self.on_premise_window[i:i + job_height, a:a + job_width]

                            # 高さまたは幅が足りない場合、スキップ
                            # print('on_premise_window',self.on_premise_window)
                            # print('part_matrix',part_matrix)
                            if part_matrix.shape[0] < job_height or part_matrix.shape[1] < job_width:
                                continue
                            if np.all(part_matrix == 0):
                                self.on_premise_window[i:i + job_height, a:a + job_width] = 1  # ウィンドウにジョブを割り当て
                                # print('オンプレミスの上から' + str(i + 1) + '〜' + str(i + job_height) + '番目のノードに割り当て成功')
                                # print("現在のウィンドウ状態:\n", self.on_premise_window)
                                self.on_premise_window_user[i: i + job_height, a + time: a + time + job_width] = job[3]
                                self.jobs[job_id][-1] = time - 1
                                
                                self.user_priority[int(job[3])][1]  -= job_height * job_width
                                # print('reduced', self.user_priority)
                                self.user_wt[int(job[3])][1] = time + a - when_submitted
                                self.user_wt_sum[int(job[3])][1] += time + a - when_submitted
                                self.user_wt_log.append([int(job[4]), int(job[3]), time + a - when_submitted])
                                # print("self.on_premise_window_user:\n",self
                                allocated = True
                                self.job_allocated.append([a,job[0],job[1],job[2],job[3],job[4],job[5]])
                                return time+1, priority_job
                
                    else :
                        # print("右にスライドしてたらジョブがスライドウィンドウの右にはみ出てしまいます")
                        break

                if not allocated:
                    print('場所がありません:checkvalid異常 at cloud==1')
                    exit()
            else:
                print('ジョブがでかすぎてスライドウィンドウの右にはみ出てしまいます:checkvalid異常 at cloud==1')
                exit()


        else:  # クラウドに割り当てる場合
            if  job_width  <= self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                allocated = False  # 暫定
                for a in range(self.n_window - job_width +5 ):  # ウィンドウをyokoに見ていき、空いている部分行列を探す
                    if a +  job_width <= self.n_window:
                        for i in range(self.n_cloud_node- job_height + 1):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                            part_matrix = self.cloud_window[i:i + job_height, a:a + job_width]
                            # print("self.cloud_window: ",self.cloud_window)
                            # print("part_matrix: ",part_matrix)


                            # 高さまたは幅が足りない場合、スキップ
                            if part_matrix.shape[0] < job_height or part_matrix.shape[1] < job_width:
                                continue
                            if np.all(part_matrix == 0):
                                self.cloud_window[i:i + job_height, a:a + job_width] = 1  # ウィンドウにジョブを割り当て
                                # print('クラウドの上から' + str(i + 1) + '〜' + str(i + job_height) + '番目のノードに割り当て成功')
                                # print("在のウィンドウ状態:\n", self.on_premise_window)
                               
                                self.cloud_window_user[i:i + job_height,time +  a:a + time + job_width] = job[3]
                                # print("self.cloud_window_user:n",self.cloud_window_user)
                                self.jobs[job_id][-1] = time - 1
                                self.user_priority[int(job[3])][1] -= job_height * job_width
                                self.user_wt[int(job[3])][1] = time + a - when_submitted
                                self.user_wt_sum[int(job[3])][1] += time + a - when_submitted
                                allocated = True
                                self.job_allocated.append([a,job[0],job[1],job[2],job[3],job[4],job[5]])
                                self.user_wt_log.append([int(job[4]), int(job[3]), time + a - when_submitted])
                                return time +1 , priority_job
                    else :
                    # print("右にスライドしてたらジョブがスライドウィンドウの右にはみ出てしまいます")
                        break
                            
                
                if not allocated:
                    print('場所がありません:checkvalid異常')
                    exit()
            else:
                print('ジョブがスライドウィンドウの右にはみ出てしまいますcheckvalid異常')
                exit()

    def schedule_for_fairness(self, action):
        a = 0
        k=0
        # job = self.job_queue[0]
        max_priority = -10000000
        priority_job = 0

        job= self.job_queue[0]




        # print("from\n",self.job_queue)
        priority_rank = sorted(self.user_priority, reverse=True, key = lambda x: x[1])
        # print('user_p',self.user_priority)
        # print(priority_rank)

        alo = 0


        for k in range(len(self.user_priority)):
            for a in range(len(self.job_queue)):
                # print(priority_rank[k])
                if self.job_queue[a][1] != 0 and priority_rank[k][0] == self.job_queue[a][3]:


                    job = self.job_queue[a]
                    priority_job=a
                    alo = 1
                    break
            if alo == 1:
                break

        # print("from\n",self.job_queue)

        # print(job,"allocated")



        job_width = int(job[0])
        job_height = int(job[1])
        can_use_cloud = int(job[2])
        # print("can use: ",can_use_cloud)
        job_id = int(job[4])
        when_submitted = int(job[-1])
        when_allocate = 0
        method = action[0]
        use_cloud = action[1]

        # print("job_id: ", job_id)
        time = self.time
        # print('job', job)

        # print("weight,height"+ str(job_height) + str(job_width))
        

        # print(job, type(job_width), type(job_height), type(can_use_cloud), type(when_allocate), type(use_cloud), self.on_premise_window.shape[0])

        if not use_cloud:  # オンプレミスに割り当てる場合
            if  job_width <=  self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                allocated = False  # 暫定
                for a in range(self.n_window - job_width + 5 ):  # ウィンドウをyokoに見ていき、空いている部分行列を探す
                    if a +  job_width <= self.n_window:
                        for i in range(self.n_on_premise_node- job_height + 1):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                            part_matrix = self.on_premise_window[i:i + job_height, a:a + job_width]

                            # 高さまたは幅が足りない場合、スキップ
                            # print('on_premise_window',self.on_premise_window)
                            # print('part_matrix',part_matrix)
                            if part_matrix.shape[0] < job_height or part_matrix.shape[1] < job_width:
                                continue
                            if np.all(part_matrix == 0):
                                self.on_premise_window[i:i + job_height, a:a + job_width] = 1  # ウィンドウにジョブを割り当て
                                # print('オンプレミスの上から' + str(i + 1) + '〜' + str(i + job_height) + '番目のノードに割り当て成功')
                                # print("現在のウィンドウ状態:\n", self.on_premise_window)
                                self.on_premise_window_user[i: i + job_height, a + time: a + time + job_width] = job[3]
                                self.jobs[job_id][-1] = time - 1
                                
                                self.user_priority[int(job[3])][1]  -= job_height * job_width
                                # print('reduced', self.user_priority)
                                self.user_wt[int(job[3])][1] = time + a - when_submitted
                                self.user_wt_sum[int(job[3])][1] += time + a - when_submitted
                                self.user_wt_log.append([int(job[4]), int(job[3]), time + a - when_submitted,job[0],job[1],job[-1],0,job[2]])
                                # print(i,a,time)
                                # print("self.on_premise_window_user:\n",self.on_premise_window_user)
                                allocated = True
                                self.job_allocated.append([a,job[0],job[1],job[2],job[3],job[4],job[5]])
                                return time+1, priority_job
                
                    else :
                        # print("右にスライドしてたらジョブがスライドウィンドウの右にはみ出てしまいます")
                        break

                if not allocated:
                    print('場所がありません:checkvalid異常 at cloud==1')
                    exit()
            else:
                print('ジョブがでかすぎてスライドウィンドウの右にはみ出てしまいます:checkvalid異常 at cloud==1')
                exit()


        else:  # クラウドに割り当てる場合
            if  job_width  <= self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                allocated = False  # 暫定
                for a in range(self.n_window - job_width +5 ):  # ウィンドウをyokoに見ていき、空いている部分行列を探す
                    if a +  job_width <= self.n_window:
                        for i in range(self.n_cloud_node- job_height + 1):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                            part_matrix = self.cloud_window[i:i + job_height, a:a + job_width]
                            # print("self.cloud_window: ",self.cloud_window)
                            # print("part_matrix: ",part_matrix)


                            # 高さまたは幅が足りない場合、スキップ
                            if part_matrix.shape[0] < job_height or part_matrix.shape[1] < job_width:
                                continue
                            if np.all(part_matrix == 0):
                                self.cloud_window[i:i + job_height, a:a + job_width] = 1  # ウィンドウにジョブを割り当て
                                # print('クラウドの上から' + str(i + 1) + '〜' + str(i + job_height) + '番目のノードに割り当て成功')
                                # print("在のウィンドウ状態:\n", self.on_premise_window)
                               
                                self.cloud_window_user[i:i + job_height,time +  a:a + time + job_width] = job[3]
                                # print(i,a,time)
                                # print("self.cloud_window_user:\n",self.cloud_window_user)
                                self.jobs[job_id][-1] = time - 1
                                self.user_priority[int(job[3])][1] -= job_height * job_width
                                self.user_wt[int(job[3])][1] = time + a - when_submitted
                                self.user_wt_sum[int(job[3])][1] += time + a - when_submitted
                                allocated = True
                                self.job_allocated.append([a,job[0],job[1],job[2],job[3],job[4],job[5]])
                                self.user_wt_log.append([int(job[4]), int(job[3]), time + a - when_submitted,job[0],job[1],job[-1],1,job[2]])
                                return time +1 , priority_job
                    else :
                    # print("右にスライドしてたらジョブがスライドウィンドウの右にはみ出てしまいます")
                        break
                            
                
                if not allocated:
                    print('場所がありません:checkvalid異常')
                    exit()
            else:
                print('ジョブがスライドウィンドウの右にはみ出てしまいますcheckvalid異常')
                exit()

    # スケジューリング
    def schedule(self, action):

        # print(self.job_queue)
        # print(self.jobs)
        a = 0
        job = self.job_queue[0]
        job_width = int(job[0])
        job_height = int(job[1])
        # print("job_width: ",job_width)
        # print("job_height: ",job_height)

        can_use_cloud = int(job[2])
        # print("can use: ",can_use_cloud)
        job_id = int(job[4])
        when_submitted = int(job[-1])
        when_allocate = 0

        method = action[0]
        use_cloud = action[1]

        # print("job_id: ", job_id)
        time = self.time
        # print('job', job)

        # print("weight,height"+ str(job_height) + str(job_width))
        

        # print(job, type(job_width), type(job_height), type(can_use_cloud), type(when_allocate), type(use_cloud), self.on_premise_window.shape[0])

        if not use_cloud:  # オンプレミスに割り当てる場合
            if  job_width <=  self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                allocated = False  # 暫定
                for a in range(self.n_window - job_width + 5 ):  # ウィンドウをyokoに見ていき、空いている部分行列を探す
                    if a +  job_width <= self.n_window:
                        for i in range(self.n_on_premise_node- job_height + 1):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                            part_matrix = self.on_premise_window[i:i + job_height, a:a + job_width]

                            # 高さまたは幅が足りない場合、スキップ
                            # print('on_premise_window',self.on_premise_window)
                            # print('part_matrix',part_matrix)
                            if part_matrix.shape[0] < job_height or part_matrix.shape[1] < job_width:
                                continue
                            if np.all(part_matrix == 0):
                                self.on_premise_window[i:i + job_height, a:a + job_width] = 1  # ウィンドウにジョブを割り当て
                                # print('オンプレミスの上から' + str(i + 1) + '〜' + str(i + job_height) + '番目のノードに割り当て成功')
                                # print("現在のウィンドウ状態:\n", self.on_premise_window)
                                self.on_premise_window_user[i: i + job_height, a + time: a + time + job_width] = job[3]
                                # print(i,a,time)

                                # print("self.on_premise_window_user:n",self.on_premise_window_user)

                                self.jobs[job_id][-1] = time - 1
                                self.user_priority[int(job[3])][1] -= job_height * job_width
                                self.user_wt[int(job[3])][1] = time + a - when_submitted  
                                self.user_wt_sum[int(job[3])][1] += time + a - when_submitted                    
                                # print("self.on_premise_window_user:\n",self
                                allocated = True
                                self.job_allocated.append([a,job[0],job[1],job[2],job[3],job[4],job[5]])
                                self.user_wt_log.append([int(job[4]), int(job[3]), time + a - when_submitted,job[0],job[1],job[-1],0,job[2]])
                                return a
                
                    else :
                        # print("右にスライドしてたらジョブがスライドウィンドウの右にはみ出てしまいます")
                        break

                if not allocated:
                    print('場所がありません:checkvalid異常 at cloud==1')
                    exit()
            else:
                print('ジョブがでかすぎてスライドウィンドウの右にはみ出てしまいます:checkvalid異常 at cloud==1')
                exit()


        else:  # クラウドに割り当てる場合
            if  job_width  <= self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                allocated = False  # 暫定
                for a in range(self.n_window - job_width +5 ):  # ウィンドウをyokoに見ていき、空いている部分行列を探す
                    if a +  job_width <= self.n_window:
                        for i in range(self.n_cloud_node- job_height + 1):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                            part_matrix = self.cloud_window[i:i + job_height, a:a + job_width]
                            # print("self.cloud_window: ",self.cloud_window)
                            # print("part_matrix: ",part_matrix)


                            # 高さまたは幅が足りない場合、スキップ
                            if part_matrix.shape[0] < job_height or part_matrix.shape[1] < job_width:
                                continue
                            if np.all(part_matrix == 0):
                                self.cloud_window[i:i + job_height, a:a + job_width] = 1  # ウィンドウにジョブを割り当て
                                # print('クラウドの上から' + str(i + 1) + '〜' + str(i + job_height) + '番目のノードに割り当て成功')
                                # print("在のウィンドウ状態:\n", self.on_premise_window)
                               
                                self.cloud_window_user[i:i + job_height,time +  a:a + time + job_width] = job[3]
                                # print(i,a,time)
                                # print("self.cloud_window_user:n",self.cloud_window_user)
                                self.jobs[job_id][-1] = time-1
                                self.user_priority[int(job[3])][1] -= job_height * job_width
                                self.user_wt[int(job[3])][1] = time + a - when_submitted     
                                self.user_wt_sum[int(job[3])][1] += time + a - when_submitted                    

                                allocated = True
                                self.job_allocated.append([a,job[0],job[1],job[2],job[3],job[4],job[5]])
                                self.user_wt_log.append([int(job[4]), int(job[3]), time + a - when_submitted,job[0],job[1],job[-1],1,job[2]])
                                return a
                    else :
                    # print("右にスライドしてたらジョブがスライドウィンドウの右にはみ出てしまいます")
                        break
                            
                
                if not allocated:
                    print('場所がありません:checkvalid異常')
                    exit()
            else:
                print('ジョブがスライドウィンドウの右にはみ出てしまいますcheckvalid異常')
                exit()
                    

    # 観測データ(状態)を取得
    def get_observation(self):
        mm = MinMaxScaler()
        #         obs_on_premise_window = normalize(self.on_premise_window.flatten(), 0, 1) # オンプレミスの観測データ
        #         obs_cloud_window = normalize(self.cloud_window.flatten(), 0, 1) # クラウドの観測データ
        obs_on_premise_window = self.on_premise_window.flatten()
        obs_cloud_window = self.cloud_window.flatten()
        min_job_queue = [0, 0, 0, 0]
        max_job_queue = [10, 10, 1, self.max_t]
        #         obs_job_queue_obs = self.job_queue[:self.n_job_queue_obs]
        obs_job_queue_obs = self.job_queue[:10, :4].flatten()
        # print("obs_job_queue_obs_left: " + str(obs_job_queue_obs))
        # obs_job_queue_obs_right = self.job_queue[:self.n_job_queue_obs, -1:]
        # print("obs_job_queue_obs_right: " + str(obs_job_queue_obs_right))
        # obs_job_queue_obs = np.concatenate([obs_job_queue_obs_left, obs_job_queue_obs_right], axis=1)
        # obs_job_queue_obs = np.append(obs_job_queue_obs, [min_job_queue, max_job_queue], axis=0)
        # obs_job_queue_obs = mm.fit_transform(obs_job_queue_obs)[:-2]
        # obs_job_queue_obs = obs_job_queue_obs.flatten()  # ジョブキューの観測部分の観測データ
        # obs_n_job_in_job_queue_bck = np.count_nonzero(self.job_queue[self.n_job_queue_obs:, 0])  
        # obs_n_job_in_job_queue_bck = np.array([obs_n_job_in_job_queue_bck / self.n_job_queue_bck]) 

        # print('obs_job_queue_obs: \n' + str(obs_job_queue_obs))
        # print('obs_n_job_in_job_queue_bck: \n' + str(obs_n_job_in_job_queue_bck))
        #         obs_n_job_queue_bck = np.append(obs_n_job_queue_bck, [[0], [self.n_job_queue_bck]])
        #         obs_n_job_queue_bck = mm_1d.fit_transform(obs_n_job_queue_bck)[:-2]
        #         obs_job_queue_obs = normalize(self.job_queue[:self.n_job_queue_obs].flatten(), 1, ) # ジョブキューの観測部分の観測データ
        #         obs_n_job_queue_bck = np.count_nonzero(self.job_queue[self.n_job_queue_obs:, 0]) # ジョブキューのバックログ部分のジョブ数の観測データ(実行時間==0のジョブの数)
        #         print('obs_job_queue_obs: \n' + str(obs_job_queue_obs))
        #         print('obs_n_job_in_job_queue_bck: \n' + str(obs_n_job_in_job_queue_bck))
        # 結合
        observation = np.concatenate(
            [obs_on_premise_window, obs_cloud_window, obs_job_queue_obs])
        #         print(observation)
        return observation

    # 報酬を取得
    def get_reward(self, action, allocated_job, time, is_valid, job_none):
        reward_liner = 0
        reward_wt = 0
        reward_cost = 0
        reward = [0,0]
        use_cloud = action[1] 
        if job_none == False:  # ジョブキューが空でない場合            
            if is_valid:  # actionが有効だった場合
                submitted_time = allocated_job[-1]

                # 割り当てたジョブの待ち時間
                # print(self.time)
                waiting_time = self.time
                reward_wt = 100 - waiting_time

                # penalty = min(waiting_time/self.n_window, 2) + use_cloud
                # reward_liner = weight_cost * (1 - use_cloud * 2) + weight_wt * (1 - waiting_time / self.n_window)
                # reward_wt = 1 - waiting_time / self.n_window
                # reward_wt = - waiting_time
                # reward_cost = 1 - use_cloud



        # #cost or waitingtimeがゼロになるときはrewardを0にする
        # if reward_cost == 0 or reward_wt == 0:
        #     reward = [0,0]
        # else:
        #     reward = [1/(reward_cost), 1/(reward_wt)]
            
        #cost or waitingtimeがゼロになるときはrewardを0にする
        reward =[0, reward_wt] 
        # reward = [reward_liner, reward_liner]

        return reward

    # コストを計算
    def compute_cost(self, action, allocated_job, is_valid):
        if is_valid:  # actionが有効だった場合
            if action[1] == 0:  # オンプレミスに割り当てる場合
                cost = 0
            elif action[1] == 1:  # クラウドに割り当てる場合
                cost = allocated_job[0] * allocated_job[1]  # (処理時間)*(クラウドで使うノード数)をコストとする
            else:  # 割り当てない場合
                cost = -1
        else:  # actionが無効だった場合
            cost = 0  # 平均コストの計算で母数に入れないように

        return cost

    # stateをもとにactionが有効かどうかを判定
    def check_is_valid_action(self, action):
        a = 0
        
        # print("job: ",job)

        when_allocate = 0
        method = action[0]
        use_cloud = action[1]
        max_priority = -100000
        priority_job = 0
        k =0
        minn =0
        maxx =10000

        job = self.job_queue[0]

                # print("from\n",self.job_queue)
        priority_rank = sorted(self.user_priority, reverse=True, key = lambda x: x[1])

        # print(priority_rank)

        if method ==1:
            # print("from\n",self.jobs)
            # print("from\n",self.job_queue)
            
            alo = 0

            for k in range(len(self.user_priority)):
                for a in range(len(self.job_queue)):
                    if self.job_queue[a][1] != 0 and priority_rank[k][0] == self.job_queue[a][3]:
                        # print('jobset')
                        job = self.job_queue[a]
                        priority_job=a
                        alo = 1
                        break
                if alo == 1:
                    break
            
            
            # print(job,"will be allocated")

            

        elif method ==0:
            job = self.job_queue[0]

        elif method ==2:
            for a in range(len(self.job_queue)):
            # print(priority_rank[k])
                if minn > self.job_queue[a][0] * self.job_queue[a][1]:
                    job = self.job_queue[a]


        elif method ==3:
            for a in range(len(self.job_queue)):
            # print(priority_rank[k])
                if maxx < self.job_queue[a][0] * self.job_queue[a][1]:
                    job = self.job_queue[a]

        

        

        job_width = int(job[0])
        job_height = int(job[1])
        can_use_cloud = int(job[2])
        # print("can use: ",can_use_cloud)
        job_id = int(job[4])
        when_submitted = int(job[-1])
        # is_valid = True # actionが有効かどうか

        # print('job', job)
        # print('job', job)
        time = self.time

        if np.all(job == 0):  # ジョブキューが空の場合
            is_valid = False
            return is_valid,-1

        # actionが有効かどうかを判定
        if not use_cloud:  # オンプレミスに割り当てる場合
            if job_width < self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                is_valid = False  # 暫定
                for a in range(self.n_window - job_width +1 ):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                    if a +  job_width <= self.n_window:
                
                        for i in range(self.n_on_premise_node - job_height + 1):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                        
                            part_matrix = self.on_premise_window[i:i + job_height, a:a + job_width]
                            # print('on_premise_window:\n',self.on_premise_window)
                            # print("part_matrix: ",part_matrix)

                            # 高さまたは幅が足りない場合、スキップ
                            if part_matrix.shape[0] < job_height or part_matrix.shape[1] < job_width:
                                continue
                            if np.all(part_matrix == 0):
                                is_valid = True
                                # print(str(a)+'秒後、オンプレミスの上から' + str(i + 1) + '〜' + str(i + job_height) + '番目のノードに割り当てられます')
                                return is_valid, time+a-when_submitted  # 割り当て成功
                        # print('shift right')
                    else :
                        # print("右にスライドしてたらジョブがスライドウィンドウの右にはみ出てしまいます")
                        pass
                if not is_valid:
                    # print('場所がありません')
                    pass

            else:
                print('ジョブがスライドウィンドウの右にはみ出てしまいます')
                pass
            
            return is_valid, -1
        
        else:  # クラウドに割り当てる場合
            if job_width < self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                is_valid = False  # 暫定
                for a in range(self.n_window - job_width +1 ):  # ウィンドウをyokoに見ていき、空いている部分行列を探す
                    if a +  job_width <= self.n_window:
                        for i in range(self.n_cloud_node- job_height + 1):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                            part_matrix = self.cloud_window[i:i + job_height, a:a + job_width]
                            # print('cloud_window:\n ',self.cloud_window)
                            # print("part_matrix: ",part_matrix)

                                                        # 高さまたは幅が足りない場合、スキップ
                            if part_matrix.shape[0] < job_height or part_matrix.shape[1] < job_width:
                                continue
                            if np.all(part_matrix == 0):
                                is_valid = True
                                # print( str(a)+'秒後、クラウドの上から' + str(i + 1) + '〜' + str(i + job_height) + '番目のノードに割り当てられます')
                                return is_valid, time+a-when_submitted  # 割り当て成功
                        # print('今はダメ')
                    else :
                        # print("右にスライドしてたらジョブがスライドウィンドウの右にはみ出てしまいます")
                        pass
                if not is_valid:
                    # print('場所がありません')
                    pass
                    
            else:
                # print('ジョブがスライドウィンドウの右にはみ出てしまいます')
                pass
            
            return is_valid,-1

    # エピソード終了条件を判定
    def check_is_done(self):
        # 1エピソードの最大ステップ数に達するか、# 最後のジョブまでジョブキューに格納していた場合、終了する
        # print("index_next_job: ",self.index_next_job)
        # print("len(self.jobs): ",len(self.jobs))
        return self.step_count == self.max_step or (
                self.index_next_job == len(self.jobs) and np.all(self.job_queue == 0))

    # 画面への描画
    def render(self, mode='human'):
        pass

    # 終了時の処理
    def close(self):
        pass

    # 乱数の固定
    def seed(self, seed=None):
        pass