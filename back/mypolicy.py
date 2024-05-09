import numpy as np
import random
from rl.policy import Policy, EpsGreedyQPolicy


# epsilonを徐々に小さくしていくことで探索主義から徐々に貪欲になっていくepsilon-greedy法
class EpsGreedyQPolicy_decay(Policy):
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """

    def __init__(self, eps=1, min_eps=.01, eps_decay=.99996):
        super(EpsGreedyQPolicy_decay, self).__init__()
        self.eps = eps
        self.min_eps = min_eps
        # 次第に貪欲になるためのパラメータ
        self.eps_decay = eps_decay

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        # epsにeps_decayをかけていくことで次第に貪欲にしていく
        if self.eps > self.min_eps:
            self.eps *= self.eps_decay

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
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
    def __init__(self, eps=1, min_eps=.01, eps_decay=.99996):
        super(EpsGreedyQPolicy_with_action_filter_and_decay, self).__init__()
        self.eps = eps
        self.min_eps = min_eps
        # 次第に貪欲になるためのパラメータ
        self.eps_decay = eps_decay

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        # nb_actions = q_values.shape[0]

        # epsにeps_decayをかけていくことで次第に貪欲にしていく
        if self.eps > self.min_eps:
            self.eps *= self.eps_decay

        if np.random.uniform() < self.eps:
            # action = np.random.randint(0, nb_actions)
            # 有効なactionの中からランダムに採用
            #             print('take a random action')
            valid_actions = [i for i, q in enumerate(q_values) if q != -np.inf]
            #             print('valid_actions:\n' + str(valid_actions))
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