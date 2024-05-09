import gym
import numpy as np
from collections import deque
import sys
from sklearn.preprocessing import MinMaxScaler


# 学習環境
class SchedulingEnv(gym.core.Env):
    def __init__(self, max_step, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck, weight_wt,
                 weight_cost, penalty_not_allocate, penalty_invalid_action, multi_algorithm=False, jobs_set=None,
                 job_type=0):
        self.step_count = 0  # 現在のステップ数(今何ステップ目かを示す)
        self.episode = 0  # 現在のエピソード(今何エピソード目かを示す); agentに教えてもらう
        self.time = 0  # 時刻(ジョブ到着の判定に使う)
        self.max_step = max_step  # ステップの最大数(1エピソードの終了時刻)
        self.index_next_job = 0  # 次に待っている新しいジョブのインデックス 新しいジョブをジョブキューに追加するときに使う
        # self.index_next_job_ideal = 0 # 理想的な状態(処理時間を迎えたのにジョブキューがいっぱいでジョブキューに格納されていないジョブがない)であれば次に待っている新しいジョブのインデックス
        self.n_window = n_window  # スライドウィンドウの横幅
        self.n_on_premise_node = n_on_premise_node  # オンプレミス計算資源のノード数
        self.n_cloud_node = n_cloud_node  # クラウド計算資源のノード数
        self.n_job_queue_obs = n_job_queue_obs  # ジョブキューの観測部分の長さ
        self.n_job_queue_bck = n_job_queue_bck  # ジョブキューのバックログ部分の長さ
        self.rear_job_queue = 0  # ジョブキューの末尾 (== 0: ジョブキューが空)
        self.weight_wt = weight_wt  # 報酬における待ち時間の重み
        self.weight_cost = weight_cost  # 報酬におけるコストの重み
        self.all_waiting_times = []
        self.penalty_not_allocate = penalty_not_allocate  # 割り当てない(一時キューに格納する)という行動を選択した際のペナルティー
        self.penalty_invalid_action = penalty_invalid_action  # actionが無効だった場合のペナルティー
        self.on_premise_window = np.zeros((self.n_on_premise_node, self.n_window))  # オンプレミスのスライドウィンドウ
        self.cloud_window = np.zeros((self.n_cloud_node, self.n_window))  # クラウドのスライドウィンドウ
        self.job_queue = np.zeros((self.n_job_queue_obs + self.n_job_queue_bck, 6))  # ジョブキュー
        self.n_action = self.n_window * 2 + 1  # 行動数
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

    # 各ステップで実行される操作
    def step(self, action_raw):
        # ステップを進める
        self.step_count += 1

        # 時刻
        time = self.time

        # このステップで割り当てられるジョブ(cost計算に使う)
        allocated_job = self.job_queue[0]
        #         print('allocated_job: ' + str(allocated_job))

        # スカラーで与えられたactionをリストに変換
        action = self.get_converted_action(action_raw)
        #         print('action: ' + str(action))

        # actionが有効なら通常通りactionに基づいて状態を遷移させるが無効なら遷移させない
        if self.check_is_valid_action(action):
            is_valid = True
            # 状態遷移
            self.state_transition(action)
        else:
            is_valid = False

        # 観測データ(状態)を取得
        observation = self.get_observation()
        # 報酬を取得
        rewards = self.get_rewards(action, allocated_job, time, is_valid)

        # #reward_costはrewards[0]と同じ,reward_wtはrewards[1]と同じ
        # reward_cost = rewards[0]
        # reward_wt = rewards[1]

    
        # コストを取得
        cost = self.compute_cost(action, allocated_job, is_valid)

        # エピソードの終了時刻を満たしているかの判定
        done = self.check_is_done()

        info = {}
        # if self.step_count < 50:
        #     print('step: ' + str(self.step_count) + ' time: ' + str(self.time) + ' action: ' + str(
        #         action) + ' cost ' + str(cost) + ' reward: ' + str(reward) + ' rear_job_queue: ' + str(
        #         self.rear_job_queue) + ' index_next_job: ' + str(self.index_next_job))
        #     print('on_premise_node:\n' + str(self.on_premise_window))
        #     print('cloud_node:\n' + str(self.cloud_window))
        #     print('job_queue: # [processing_time, required_nodes_num, can_use_cloud]\n' + str(self.job_queue))
        #     # print('observation: \n' + str(observation))
        #     # print('waiting_time: ' + str(waiting_time))
        #     print('done: ' + str(done))
        #     # print('count_failed_to_allocating: ' + str(self.count_failed_to_allocating))
        #     print('================================================')

        #         return observation, reward, cost, time, waiting_time, done, info
        return observation, rewards, cost, time, done, info

    # スカラーのactionをリストに変換
    def get_converted_action(self, a):
        if a < 0 or a > 2 * self.n_window:
            print('UnexpectedAction')
            sys.exit()
        if a == 2 * self.n_window:
            when_allocate = -1
        else:
            when_allocate = a % self.n_window
        if a < self.n_window:
            use_cloud = 0
        elif a < 2 * self.n_window:
            use_cloud = 1
        else:
            use_cloud = -1
        action = [when_allocate, use_cloud]

        return action

    # 初期化
    # 各エピソードの最初に呼び出される
    def reset(self):
        # エピソードを1進める
        # self.episode += 1

        # 変数を初期化
        self.time = 0
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
        self.cloud_window = np.zeros((self.n_cloud_node, self.n_window))  # クラウドのスライドウィンドウ
        self.job_queue = np.zeros((self.n_job_queue_obs + self.n_job_queue_bck, 6))  # ジョブキュー
        self.rear_job_queue = 0  # ジョブキューの末尾 (== 0: ジョブキューが空)
        self.tmp_queue = deque()  # 割り当てられなかったジョブを一時的に格納するキュー

        # ジョブキューに新しいジョブを追加
        self.append_new_job2job_queue()
        # 観測データ(状態)を取得
        observation = self.get_observation()

        # print('-------------reseted-------------')

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
                    print('max_processing_time: ' + str(max_processing_time))
                    print('max_n_required_nodes: ' + str(max_n_required_nodes))

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
                # print(jobs)
        return jobs

    # 状態遷移
    def state_transition(self, action):
        if np.all(self.job_queue[0] == 0):  # 先頭ジョブが空(ジョブキューが空)
            #             print('先頭ジョブが空(ジョブキューが空)')
            # 時間を1進める
            self.time += 1
            #             print('時間を1進めました')
            # オンプレミス, クラウドそれぞれのスライディングウィンドウを1タイムスライス分だけスライド
            self.on_premise_window = np.roll(self.on_premise_window, -1, axis=1)
            self.on_premise_window[:, -1] = 0
            self.cloud_window = np.roll(self.cloud_window, -1, axis=1)
            self.cloud_window[:, -1] = 0
            # ジョブキューに一時キューのジョブを格納
            while self.tmp_queue:  # 一時キューが空でない場合繰り返す
                self.job_queue[self.rear_job_queue] = self.tmp_queue.popleft()
                self.rear_job_queue += 1
            # ジョブキューに新しいジョブを追加
            self.append_new_job2job_queue()
        elif not np.all(self.job_queue[1] == 0):  # 先頭二つのジョブが埋まっている(普通)
            #             print('先頭二つのジョブが埋まっている(普通)')
            # スケジューリング
            self.schedule(action)
        elif np.all(self.job_queue[1] == 0):  # 先頭ジョブは埋まっているが二つ目が空(ジョブキューにジョブが一つ)
            #             print('先頭ジョブは埋まっているが二つ目が空(ジョブキューにジョブが一つ)')
            # スケジューリング
            self.schedule(action)
            # 時間を1進める
            self.time += 1
            #             print('時間を1進めました')
            # オンプレミス, クラウドそれぞれのスライディングウィンドウを1タイムスライス分だけスライド
            self.on_premise_window = np.roll(self.on_premise_window, -1, axis=1)
            self.on_premise_window[:, -1] = 0
            self.cloud_window = np.roll(self.cloud_window, -1, axis=1)
            self.cloud_window[:, -1] = 0
            # ジョブキューに一時キューのジョブを格納
            while self.tmp_queue:  # 一時キューが空になるまで
                self.job_queue[self.rear_job_queue] = self.tmp_queue.popleft()
                self.rear_job_queue += 1
            # ジョブキューに新しいジョブを追加
            self.append_new_job2job_queue()
        else:
            print('state_transitionでelse分岐(ありえない)')

    # ジョブキューに新しいジョブを追加
    def append_new_job2job_queue(self):
        for i in range(self.rear_job_queue, self.n_job_queue_obs + self.n_job_queue_bck):
            # print('index_next_job: ' + str(self.index_next_job))
            if self.index_next_job == len(self.jobs):  # 最後のジョブまでジョブキューに格納した場合、脱出
                break
            head_job = self.jobs[self.index_next_job]  # 先頭ジョブ

            if head_job[0] <= self.time:  # 先頭のジョブが到着時刻を迎えていればジョブキューに追加
                # ジョブキューに格納する前に提出時間が末尾に，処理時間が先頭になるようにインデックスをずらす
                head_job = np.roll(head_job, -1)

                #                 self.job_queue[i] = head_job[1:]
                self.job_queue[i] = head_job
                self.rear_job_queue += 1
                self.index_next_job += 1

        # 理想的な状態であれば次に待っている新しいジョブのインデックスを更新

    #         self.get_index_next_job_ideal()

    # 次にジョブキューへの格納を待っているジョブで一番後ろのジョブのインデックスを返す
    #     def get_index_next_job_ideal(self):
    #         ideal_index_next_job = self.index_next_job
    #         while True:
    #             if self.jobs[ideal_index_next_job+1][0] <= self.time:
    #                 max_index_waiting_job += 1
    #             else:
    #                 break

    #         return max_index_waiting_job

    # スケジューリング
    def schedule(self, action):
        job = self.job_queue[0]
        job_width = int(job[0])
        job_height = int(job[1])
        can_use_cloud = int(job[2])
        job_id = int(job[3])
        when_submitted = int(job[-1])
        when_allocate = action[0]
        use_cloud = action[1]

        # print(job, type(job_width), type(job_height), type(can_use_cloud), type(when_allocate), type(use_cloud), self.on_premise_window.shape[0])

        # ジョブキューのスライド処理
        self.job_queue = np.roll(self.job_queue, -1, axis=0)
        self.job_queue[-1] = 0
        self.rear_job_queue -= 1

        if action[0] == -1:  # 割り当てない場合(action=(-1,phi))
            self.tmp_queue.append(job)
        elif not use_cloud:  # オンプレミスに割り当てる場合
            succeeded_in_allocating = False
            if when_allocate + job_width - 1 < self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                for i in range(self.n_on_premise_node):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                    part_matrix = self.on_premise_window.take(range(i, i + job_height), mode='wrap', axis=0).take(
                        range(when_allocate, when_allocate + job_width), mode='wrap', axis=1)  # 割り当て候補部分行列
                    if np.all(part_matrix == 0):  # 割り当てられる(割り当てる部分行列が見つかった)場合
                        succeeded_in_allocating = True
                        for j in np.arange(i, i + job_height) % self.n_on_premise_node:  # 該当部分行列に1行ずつ
                            self.on_premise_window[j, when_allocate:when_allocate + job_width] = 1  # 割り当て
                            # 割り当てたら，そのジョブの提出されてからの待ち時間を計算して格納
                            self.jobs[job_id][-1] = self.time + when_allocate + 1 - when_submitted
                        # if 300 < self.step_count and self.step_count < 330: print('オンプレミスの上から'+ str(i+1) + '〜' + str(i+1+job_height-1) + '番目のノードに割り当て成功')
                        # print('オンプレミスの上から'+ str(i+1) + '〜' + str(i+1+job_height-1) + '番目のノードに割り当て成功')
                        break
            else:  # 割り当てるジョブがスライドウィンドウの右にはみ出る場合
                print('ジョブがスライドウィンドウの右にはみ出てしまいます')
            if not succeeded_in_allocating:
                print('failed to allocate')
                self.count_failed_to_allocating += 1
        elif use_cloud:  # クラウドに割り当てる場合
            succeeded_in_allocating = False
            if not can_use_cloud:  # クラウド使用が許可されていない場合
                print('cant use cloud.')
                pass
            else:  # クラウド使用が許可されている場合
                if when_allocate + job_width - 1 < self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                    for i in range(self.n_cloud_node):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                        part_matrix = self.cloud_window.take(range(i, i + job_height), mode='wrap', axis=0).take(
                            range(when_allocate, when_allocate + job_width), mode='wrap', axis=1)  # 割り当て候補部分行列
                        if np.all(part_matrix == 0):  # 割り当てられる(割り当てる部分行列が見つかった)場合
                            succeeded_in_allocating = True
                            for j in np.arange(i, i + job_height) % self.n_cloud_node:  # 該当部分行列に1行ずつ
                                self.cloud_window[j, when_allocate:when_allocate + job_width] = 1  # 割り当て
                                # 割り当てたら，そのジョブの提出されてからの待ち時間を計算して格納
                                self.jobs[job_id][-1] = self.time + when_allocate + 1 - when_submitted
                            #                             if 300 < self.step_count and self.step_count < 330: print('クラウドの上から'+ str(i+1) + '〜' + str(i+1+job_height-1) + '番目のノードに割り当て成功')
                            #                             print('クラウドの上から'+ str(i+1) + '〜' + str(i+1+job_height-1) + '番目のノードに割り当て成功')
                            break
                else:  # 割り当てるジョブがスライドウィンドウの右にはみ出る場合
                    print('ジョブがスライドウィンドウの右にはみ出てしまいます')
                if not succeeded_in_allocating:
                    print('failed to allocate')
                    self.count_failed_to_allocating += 1

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
        obs_job_queue_obs_left = self.job_queue[:self.n_job_queue_obs, :3]
        obs_job_queue_obs_right = self.job_queue[:self.n_job_queue_obs, -1:]
        obs_job_queue_obs = np.concatenate([obs_job_queue_obs_left, obs_job_queue_obs_right], axis=1)
        obs_job_queue_obs = np.append(obs_job_queue_obs, [min_job_queue, max_job_queue], axis=0)
        obs_job_queue_obs = mm.fit_transform(obs_job_queue_obs)[:-2]
        obs_job_queue_obs = obs_job_queue_obs.flatten()  # ジョブキューの観測部分の観測データ
        obs_n_job_in_job_queue_bck = np.count_nonzero(self.job_queue[self.n_job_queue_obs:, 0])
        obs_n_job_in_job_queue_bck = np.array([obs_n_job_in_job_queue_bck / self.n_job_queue_bck])
        #         obs_n_job_queue_bck = np.append(obs_n_job_queue_bck, [[0], [self.n_job_queue_bck]])
        #         obs_n_job_queue_bck = mm_1d.fit_transform(obs_n_job_queue_bck)[:-2]
        #         obs_job_queue_obs = normalize(self.job_queue[:self.n_job_queue_obs].flatten(), 1, ) # ジョブキューの観測部分の観測データ
        #         obs_n_job_queue_bck = np.count_nonzero(self.job_queue[self.n_job_queue_obs:, 0]) # ジョブキューのバックログ部分のジョブ数の観測データ(実行時間==0のジョブの数)
        #         print('obs_job_queue_obs: \n' + str(obs_job_queue_obs))
        #         print('obs_n_job_in_job_queue_bck: \n' + str(obs_n_job_in_job_queue_bck))
        # 結合
        observation = np.concatenate(
            [obs_on_premise_window, obs_cloud_window, obs_job_queue_obs, obs_n_job_in_job_queue_bck])
        #         print(observation)
        return observation

    # 報酬を取得
    def get_rewards(self, action, allocated_job, time, is_valid):
        if is_valid:  # actionが有効だった場合
            # 目的関数の重み
            weight_wt = self.weight_wt
            weight_cost = self.weight_cost

            when_allocate = action[0]
            use_cloud = action[1]
            submitted_time = allocated_job[-1]

            # 割り当てたジョブの待ち時間
            waiting_time = time + when_allocate - submitted_time
            fairness = abs(1)
            # print(jobs)
            # print(df['waiting_time'].var())


            if when_allocate != -1:  # 割り当てた場合
                # penalty = min(waiting_time/self.n_window, 2) + use_cloud
                waiting_time_reward = 1 - waiting_time / self.n_window
                cost_reward = (1 - use_cloud * 2)
                reward_liner = weight_cost * (1 - use_cloud * 2) + weight_wt * (1 - waiting_time / self.n_window)
            else:  # 割り当てなかった場合
                # 推定待ち時間のようなものをペナルティーとして与える
                estimated_waiting_time = (time - submitted_time) / self.n_window + 1
                waiting_time_reward = -estimated_waiting_time
                cost_reward = 0
                reward_liner =-estimated_waiting_time


        else:  # actionが無効だった場合
            waiting_time_reward = -self.penalty_invalid_action
            cost_reward = -self.penalty_invalid_action
            reward_liner =-self.penalty_invalid_action


        #cost or waitingtimeがゼロになるときはrewardを0にする
        if np.all(cost_reward == 0) or np.all(waiting_time_reward == 0):
            reward = [0,0]
        else:
            reward = [1/(cost_reward), 1/(waiting_time_reward)]

        return reward_liner

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
            cost = -1  # 平均コストの計算で母数に入れないように

        return cost

    # stateをもとにactionが有効かどうかを判定
    def check_is_valid_action(self, action):
        head_job = self.job_queue[0]
        job_width = int(head_job[0])
        job_height = int(head_job[1])
        can_use_cloud = int(head_job[2])
        when_allocate, use_cloud = action
        # is_valid = True # actionが有効かどうか

        # actionが有効かどうかを判定
        if when_allocate == -1 and use_cloud == -1:  # 割り当てないというactionの場合
            is_valid = True
        elif not use_cloud:  # オンプレミスに割り当てる場合
            if when_allocate + job_width - 1 < self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                is_valid = False  # 暫定
                for i in range(self.n_on_premise_node):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                    part_matrix = self.on_premise_window.take(range(i, i + job_height), mode='wrap', axis=0).take(
                        range(when_allocate, when_allocate + job_width), mode='wrap', axis=1)  # 割り当て候補部分行列
                    if np.all(part_matrix == 0):  # 割り当てられる(割り当てる部分行列が見つかった)場合
                        is_valid = True
                        # print('オンプレミスの上から'+ str(i+1) + '〜' + str(i+1+job_height) + '番目のノードに割り当てられます')
                        break
            else:  # 割り当てるジョブがスライドウィンドウの右にはみ出る場合
                # print('ジョブがスライドウィンドウの右にはみ出てしまいます')
                is_valid = False
        else:  # クラウドに割り当てる場合
            if not can_use_cloud:  # クラウド使用が許可されていない場合
                # print('cant use cloud.')
                is_valid = False
            else:  # クラウド使用が許可されている場合
                if when_allocate + job_width - 1 < self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                    is_valid = False  # 暫定
                    for i in range(self.n_cloud_node):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                        part_matrix = self.cloud_window.take(range(i, i + job_height), mode='wrap', axis=0).take(
                            range(when_allocate, when_allocate + job_width), mode='wrap', axis=1)  # 割り当て候補部分行列
                        if np.all(part_matrix == 0):  # 割り当てられる(割り当てる部分行列が見つかった)場合
                            is_valid = True
                            # print('クラウドの上から'+ str(i+1) + '〜' + str(i+1+job_height-1) + '番目のノードに割り当てられます')
                            break
                else:  # 割り当てるジョブがスライドウィンドウの右にはみ出る場合
                    # print('ジョブがスライドウィンドウの右にはみ出てしまいます')
                    is_valid = False

        #         if is_valid:
        #             print('action[{},{}] is valid'.format(when_allocate, use_cloud))
        #         else:
        #             print('action[{},{}] is invalid'.format(when_allocate, use_cloud))
        return is_valid

    # エピソード終了条件を判定
    def check_is_done(self):
        # 1エピソードの最大ステップ数に達するか、# 最後のジョブまでジョブキューに格納していた場合、終了する
        return self.step_count == self.max_step or (
                self.index_next_job == len(self.jobs) and np.all(self.job_queue == 0) and len(self.tmp_queue) == 0)

    # 画面への描画
    def render(self, mode='human'):
        pass

    # 終了時の処理
    def close(self):
        pass

    # 乱数の固定
    def seed(self, seed=None):
        pass