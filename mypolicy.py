import numpy as np
import random
from rl.policy import Policy, EpsGreedyQPolicy
# from performance_indicator import hypervolume
# from pareto import get_non_dominated


# epsilonを徐々に小さくしていくことで探索主義から徐々に貪欲になっていくepsilon-greedy法
class MultiObjectiveEpsGreedyQPolicy(Policy):
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """

    def __init__(self, eps=1, min_eps=.01, eps_decay=.996):
        super(MultiObjectiveEpsGreedyQPolicy, self).__init__()
        self.eps = eps
        self.min_eps = min_eps
        # 次第に貪欲になるためのパラメータ
        self.eps_decay = eps_decay

    def select_action(self, q_values, env):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        nb_actions = q_values.shape[0]

        # epsにeps_decayをかけていくことで次第に貪欲にしていく
        if self.eps > self.min_eps:
            self.eps *= self.eps_decay

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
            # 有効なactionの中からランダムに採用
            # print('take a random action')
            
        else:
            action = np.argmax(q_values)

        return action

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy

        # Returns
            Dict of config
        """
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config


# ランダムなactionを取る時も有効なactionの中からランダムに選ぶようなepsilon-greedy法
class EpsGreedyQPolicy_with_action_filter(Policy):
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """

    def __init__(self, eps=.01):
        super(EpsGreedyQPolicy_with_action_filter, self).__init__()
        self.eps = eps

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        # nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            # action = np.random.randint(0, nb_actions)
            # 有効なactionの中からランダムに採用
            # print('take a random action')
            valid_actions = [i for i, q in enumerate(q_values) if q != -np.inf]
            # print('valid_actions:\n' + str(valid_actions))
            action = random.choice(valid_actions)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy

        # Returns
            Dict of config
        """
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config


# ランダムなactionを取る時も有効なactionの中からランダムに選び，epsilonを徐々に小さくしていくことで探索主義から徐々に貪欲になっていくepsilon-greedy法
class EpsGreedyQPolicy_with_action_filter_and_decay(Policy):
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """

    # 0.99996^100000==0.01
    def __init__(self, eps_decay, eps=1, min_eps=.01):
        super(EpsGreedyQPolicy_with_action_filter_and_decay, self).__init__()
        self.eps = eps
        self.min_eps = min_eps
        # 次第に貪欲になるためのパラメータ
        self.eps_decay = eps_decay

    def select_action(self, q_values, object_mode):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        # assert q_values.ndim == 1
        # nb_actions = q_values.shape[0]

        # epsにeps_decayをかけていくことで次第に貪欲にしていく

        if self.eps > self.min_eps:
            self.eps *= self.eps_decay
        # print('self.eps:\n' + str(self.eps))

        if np.random.uniform() < self.eps:
            # print('take RANDOM action')
            # action = np.random.randint(0, nb_actions)
            # 有効なactionの中からランダムに採用
            #             print('take a random action')
            # valid_actions = np.array([[-np.inf, -np.inf, -np.inf]])
            

            # print('q_values:\n' + str(q_values))
            # q_values = np.array(q_values).reshape(3, 2).T
            # print('q_values:\n' + str(q_values.T))

            if object_mode == 'two':
                # q_values = q_values[:, :2] #2目的
                # action = self.get_action3_random(q_values)
                # action = np.random.choice([0, 1, 2,3], p=[0.1, 0.4, 0.1, 0.4])
                action = np.random.choice([0,1], p=[0.1, 0.9])
            elif object_mode == 'three':
                # q_values = q_values[:, :3]
                # action = self.get_action3_random(q_values)
                action = np.random.choice([0, 1, 2,3], p=[0.1, 0.1, 0.1, 0.7])
                
            else:
                action = np.random.choice([0, 1, 2,3], p=[0.1, 0.4, 0.1, 0.4])
            # print('q_values:\n' + str(q_values))

            # for a in range(q_values.shape[0]):
            #     if -np.inf not in q_values[a]:
            #         valid_actions = np.append(valid_actions, [q_values[a]] , axis=0)
            #         #一列目を消す
            # valid_actions = np.delete(valid_actions,0, 0).T

            

            #valid_actionsからランダムに一列選び、配列にする
            # print("q_values:\n", q_values)

            # print('valid_actions:\n' + str(valid_actions))

            # print('action:\n' + str(action))
            #actionsはrand_index番目を各行から取り出したもの
            # print('q0', q_values[0][action])
            # print('q1', q_values[1][action])

        else:
            # print('take a SMART action')
            
            # print('q_values:\n' + str(q_values))
            q_values = np.array(q_values).reshape(3, 4).T

            # print('q_values:\n' + str(q_values))
            # exit()
            if object_mode == 'cost':
                action = np.argmax(q_values[:, 0:1])
            elif object_mode == 'wt':
                action = np.argmax(q_values[:, 1:2])
            elif object_mode == 'fair':
                action = np.argmax(q_values[:, 2:3])
            elif object_mode == 'two':
                q_values = q_values[:2, :2] #2目的
                # print('q_values:\n' + str(q_values))
                # exit()
                action = self.get_action2(q_values)
                action = np.argmax(q_values[:2, 0:1])
            elif object_mode == 'three':
                
                q_values = q_values[:, :2]
                action = self.get_action2(q_values)
            else:
                print('input error: choose->cost,wt,fair,two,three')
                exit()
            
            # print('action:\n' + str(action))
            
        return action

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy

        # Returns
            Dict of config
        """
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config
    
    def is_dominated(self,points, point):
        """他の点によって支配されているかどうかを判断する関数"""
        return np.any(np.all(points >= point, axis=1) & np.any(points > point, axis=1))

    def extract_pareto_front_indices(self, points):
        """パレートフロントのインデックスを抽出する関数"""
        pareto_indices = []
        for i, point in enumerate(points):
            if not self.is_dominated(points, point):
                pareto_indices.append(i)
        return pareto_indices

    def get_action3_random(self, points):
        """Compute the action scores based upon the hypervolume metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        #ref_point = np.array([10, 10]
        #ref_pointからのhypervolumeを計算する


            #
        # パレートフロントのインデックスを抽出
        pareto_indices = self.extract_pareto_front_indices(points)

        # ゼロ埋めされた配列の生成
        zero_filled = np.zeros_like(points)
        zero_filled[pareto_indices] = points[pareto_indices]
        # print('zero_filled:\n' + str(zero_filled))

        # action = np.argmax(zero_filled[:,2:3])
        # print('action:\n' + str(action))
        non_zero_row_indices = [index for index, row in enumerate(zero_filled) if np.all(row != 0)]

        # print('non_zero_row_indices:\n' + str(non_zero_row_indices))



        # ランダムにインデックスを選択
        action = random.choice(non_zero_row_indices) if non_zero_row_indices else None

        # print('action2:\n' + str(action))
        

        return action

    def get_action3(self, points):
        """Compute the action scores based upon the hypervolume metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        #ref_point = np.array([10, 10]
        #ref_pointからのhypervolumeを計算する


            #
        # パレートフロントのインデックスを抽出
        pareto_indices = self.extract_pareto_front_indices(points[:][:2])

        # ゼロ埋めされた配列の生成
        zero_filled = np.zeros_like(points)
        zero_filled[pareto_indices] = points[pareto_indices]
        # print('zero_filled:\n' + str(zero_filled))

        action = np.argmax(zero_filled[:,2:3])
        # print('action:\n' + str(action))
        # non_zero_row_indices = [index for index, row in enumerate(zero_filled) if np.all(row != 0)]

        # print('non_zero_row_indices:\n' + str(non_zero_row_indices))



        # ランダムにインデックスを選択
        # action = random.choice(non_zero_row_indices) if non_zero_row_indices else None

        # print('action2:\n' + str(action))
        

        return action
    
    def get_action3_HV(self, points):
        """Compute the action scores based upon the hypervolume metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        #ref_point = np.array([10, 10]
        #ref_pointからのhypervolumeを計算する

        hv = []

            #
        # パレートフロントのインデックスを抽出
        pareto_indices = self.extract_pareto_front_indices(points)

        # ゼロ埋めされた配列の生成
        zero_filled = np.zeros_like(points)
        zero_filled[pareto_indices] = points[pareto_indices]
        action = np.argmax(np.prod(zero_filled-[-10,-10,-10], axis=1))

        # action = np.argmax(zero_filled[:,2:3])
        # non_zero_row_indices = [index for index, row in enumerate(zero_filled) if np.all(row != 0)]

        # print('non_zero_row_indices:\n' + str(non_zero_row_indices))

        # ランダムにインデックスを選択
        # action = random.choice(non_zero_row_indices) if non_zero_row_indices else None


        # print('action:\n' + str(action))
        

        return action
    def get_action2_HV(self, points):
        """Compute the action scores based upon the hypervolume metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        #ref_point = np.array([10, 10]
        #ref_pointからのhypervolumeを計算する

        hv = []

            #
        # パレートフロントのインデックスを抽出
        pareto_indices = self.extract_pareto_front_indices(points)

        # ゼロ埋めされた配列の生成
        zero_filled = np.zeros_like(points)
        zero_filled[pareto_indices] = points[pareto_indices]

        # action = np.argmax(zero_filled[:,2:3])
        # non_zero_row_indices = [index for index, row in enumerate(zero_filled) if np.all(row != 0)]

        # print('non_zero_row_indices:\n' + str(non_zero_row_indices))
        # print('zero_filled:\n' + str(zero_filled))

        #各要素の各要素の積が大きいもののインデックスを選択
        action = np.argmax(np.prod(zero_filled-[-10,-10], axis=1))
        # print(zero_filled-[10,10])

        # print('action:\n' + str(action))

        # exit()



        # ランダムにインデックスを選択
        # action = random.choice(non_zero_row_indices) if non_zero_row_indices else None


        # print('action:\n' + str(action))
        

        return action
    
    def get_action2(self, points):
        """Compute the action scores based upon the hypervolume metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        #ref_point = np.array([10, 10, 10])
        #ref_pointからのhypervolumeを計算する
        # パレートフロントのインデックスを抽出
        pareto_indices = self.extract_pareto_front_indices(points)

        # ゼロ埋めされた配列の生成
        zero_filled = np.zeros_like(points)
        zero_filled[pareto_indices] = points[pareto_indices]
        # 
        # print('zero_filled:\n' + str(zero_filled))

        # 全ての要素がゼロでない行のインデックスを取得
        non_zero_row_indices = [index for index, row in enumerate(zero_filled) if np.all(row != 0)]

        # print('non_zero_row_indices:\n' + str(non_zero_row_indices))

        # ランダムにインデックスを選択
        action = random.choice(non_zero_row_indices) if non_zero_row_indices else None

        # print('action:\n' + str(action))

        return action

    def get_q_set(self, state: int, action: int):
        """Compute the Q-set for a given state-action pair.

        Args:
            state (int): The current state.
            action (int): The action.

        Returns:
            A set of Q vectors.
        """
        nd_array = np.array(list(self.non_dominated[state][action]))
        q_array = self.avg_reward[state, action] + self.gamma * nd_array
        return {tuple(vec) for vec in q_array}


    def calc_non_dominated(self, state: int):
        """Get the non-dominated vectors in a given state.

        Args:
            state (int): The current state.

        Returns:
            Set: A set of Pareto non-dominated vectors.
        """
        candidates = set().union(*[self.get_q_set(state, action) for action in range(self.num_actions)])
        non_dominated = get_non_dominated(candidates)
        return non_dominated
