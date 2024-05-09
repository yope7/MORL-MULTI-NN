from rl.agents.dqn import DQNAgent
import warnings
import sys
from copy import deepcopy
import random
from tensorflow.keras.layers import Lambda, Dense
from tensorflow.keras.callbacks import History
from rl.core import Agent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.util import *
from rl.callbacks import CallbackList, TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer
from sklearn.preprocessing import MinMaxScaler
import wandb
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import warnings

import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Input, Layer, Dense

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

# 省略せずに全て表示
np.set_printoptions(threshold=np.inf)

#    Copyright (C) 2010 Simon Wessing
#    TU Dortmund University
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.




class HyperVolume:
    """
    Hypervolume computation based on variant 3 of the algorithm in the paper:
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved dimension-sweep
    algorithm for the hypervolume indicator. In IEEE Congress on Evolutionary
    Computation, pages 1157-1163, Vancouver, Canada, July 2006.

    Minimization is implicitly assumed here!

    """

    def __init__(self, referencePoint):
        """Constructor."""
        self.referencePoint = referencePoint
        self.list = []


    def compute(self, front):
        """Returns the hypervolume that is dominated by a non-dominated front.

        Before the HV computation, front and reference point are translated, so
        that the reference point is [0, ..., 0].

        """

        def weaklyDominates(point, other):
            for i in range(len(point)):
                if point[i] > other[i]:
                    return False
            return True

        relevantPoints = []
        referencePoint = self.referencePoint
        dimensions = len(referencePoint)
        for point in front:
            # only consider points that dominate the reference point
            if weaklyDominates(point, referencePoint):
                relevantPoints.append(point)
        if any(referencePoint):
            # shift points so that referencePoint == [0, ..., 0]
            # this way the reference point doesn't have to be explicitly used
            # in the HV computation
            for j in range(len(relevantPoints)):
                relevantPoints[j] = [relevantPoints[j][i] - referencePoint[i] for i in range(dimensions)]
        self.preProcess(relevantPoints)
        bounds = [-1.0e308] * dimensions
        hyperVolume = self.hvRecursive(dimensions - 1, len(relevantPoints), bounds)
        return hyperVolume


    def hvRecursive(self, dimIndex, length, bounds):
        """Recursive call to hypervolume calculation.

        In contrast to the paper, the code assumes that the reference point
        is [0, ..., 0]. This allows the avoidance of a few operations.

        """
        hvol = 0.0
        sentinel = self.list.sentinel
        if length == 0:
            return hvol
        elif dimIndex == 0:
            # special case: only one dimension
            # why using hypervolume at all?
            return -sentinel.next[0].cargo[0]
        elif dimIndex == 1:
            # special case: two dimensions, end recursion
            q = sentinel.next[1]
            h = q.cargo[0]
            p = q.next[1]
            while p is not sentinel:
                pCargo = p.cargo
                hvol += h * (q.cargo[1] - pCargo[1])
                if pCargo[0] < h:
                    h = pCargo[0]
                q = p
                p = q.next[1]
            hvol += h * q.cargo[1]
            return hvol
        else:
            remove = self.list.remove
            reinsert = self.list.reinsert
            hvRecursive = self.hvRecursive
            p = sentinel
            q = p.prev[dimIndex]
            while q.cargo != None:
                if q.ignore < dimIndex:
                    q.ignore = 0
                q = q.prev[dimIndex]
            q = p.prev[dimIndex]
            while length > 1 and (q.cargo[dimIndex] > bounds[dimIndex] or q.prev[dimIndex].cargo[dimIndex] >= bounds[dimIndex]):
                p = q
                remove(p, dimIndex, bounds)
                q = p.prev[dimIndex]
                length -= 1
            qArea = q.area
            qCargo = q.cargo
            qPrevDimIndex = q.prev[dimIndex]
            if length > 1:
                hvol = qPrevDimIndex.volume[dimIndex] + qPrevDimIndex.area[dimIndex] * (qCargo[dimIndex] - qPrevDimIndex.cargo[dimIndex])
            else:
                qArea[0] = 1
                qArea[1:dimIndex+1] = [qArea[i] * -qCargo[i] for i in range(dimIndex)]
            q.volume[dimIndex] = hvol
            if q.ignore >= dimIndex:
                qArea[dimIndex] = qPrevDimIndex.area[dimIndex]
            else:
                qArea[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                if qArea[dimIndex] <= qPrevDimIndex.area[dimIndex]:
                    q.ignore = dimIndex
            while p is not sentinel:
                pCargoDimIndex = p.cargo[dimIndex]
                hvol += q.area[dimIndex] * (pCargoDimIndex - q.cargo[dimIndex])
                bounds[dimIndex] = pCargoDimIndex
                reinsert(p, dimIndex, bounds)
                length += 1
                q = p
                p = p.next[dimIndex]
                q.volume[dimIndex] = hvol
                if q.ignore >= dimIndex:
                    q.area[dimIndex] = q.prev[dimIndex].area[dimIndex]
                else:
                    q.area[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                    if q.area[dimIndex] <= q.prev[dimIndex].area[dimIndex]:
                        q.ignore = dimIndex
            hvol -= q.area[dimIndex] * q.cargo[dimIndex]
            return hvol


    def preProcess(self, front):
        """Sets up the list data structure needed for calculation."""
        dimensions = len(self.referencePoint)
        nodeList = MultiList(dimensions)
        nodes = [MultiList.Node(dimensions, point) for point in front]
        for i in range(dimensions):
            self.sortByDimension(nodes, i)
            nodeList.extend(nodes, i)
        self.list = nodeList


    def sortByDimension(self, nodes, i):
        """Sorts the list of nodes by the i-th value of the contained points."""
        # build a list of tuples of (point[i], node)
        decorated = [(node.cargo[i], node) for node in nodes]
        # sort by this value
        decorated.sort(key=lambda x: x[0])
        # write back to original list
        nodes[:] = [node for (_, node) in decorated]
            
            
            
class MultiList: 
    """A special data structure needed by FonsecaHyperVolume. 
    
    It consists of several doubly linked lists that share common nodes. So, 
    every node has multiple predecessors and successors, one in every list.

    """

    class Node: 
        
        def __init__(self, numberLists, cargo=None): 
            self.cargo = cargo 
            self.next  = [None] * numberLists
            self.prev = [None] * numberLists
            self.ignore = 0
            self.area = [0.0] * numberLists
            self.volume = [0.0] * numberLists
    
        def __str__(self): 
            return str(self.cargo)
        
        
    def __init__(self, numberLists):  
        """Constructor. 
        
        Builds 'numberLists' doubly linked lists.

        """
        self.numberLists = numberLists
        self.sentinel = MultiList.Node(numberLists)
        self.sentinel.next = [self.sentinel] * numberLists
        self.sentinel.prev = [self.sentinel] * numberLists  
        
        
    def __str__(self):
        strings = []
        for i in range(self.numberLists):
            currentList = []
            node = self.sentinel.next[i]
            while node != self.sentinel:
                currentList.append(str(node))
                node = node.next[i]
            strings.append(str(currentList))
        stringRepr = ""
        for string in strings:
            stringRepr += string + "\n"
        return stringRepr
    
    
    def __len__(self):
        """Returns the number of lists that are included in this MultiList."""
        return self.numberLists
    
    
    def getLength(self, i):
        """Returns the length of the i-th list."""
        length = 0
        sentinel = self.sentinel
        node = sentinel.next[i]
        while node != sentinel:
            length += 1
            node = node.next[i]
        return length
            
            
    def append(self, node, index):
        """Appends a node to the end of the list at the given index."""
        lastButOne = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = lastButOne
        # set the last element as the new one
        self.sentinel.prev[index] = node
        lastButOne.next[index] = node
        
        
    def extend(self, nodes, index):
        """Extends the list at the given index with the nodes."""
        sentinel = self.sentinel
        for node in nodes:
            lastButOne = sentinel.prev[index]
            node.next[index] = sentinel
            node.prev[index] = lastButOne
            # set the last element as the new one
            sentinel.prev[index] = node
            lastButOne.next[index] = node
        
        
    def remove(self, node, index, bounds): 
        """Removes and returns 'node' from all lists in [0, 'index'[."""
        for i in range(index): 
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor  
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node
    
    
    def reinsert(self, node, index, bounds):
        """
        Inserts 'node' at the position it had in all lists in [0, 'index'[
        before it was removed. This method assumes that the next and previous 
        nodes of the node that is reinserted are in the list.

        """
        for i in range(index): 
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
            



# testの時のみaction_filterを使うDQNAgent
class MyAgent(DQNAgent):
    def __init__(self, model, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck, policy=None,
                 test_policy=None, enable_double_dqn=True, enable_dueling_network=False,
                 dueling_type='avg', target_model_update=1000, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        # Validate (important) input.
        # if list(model.output.shape) != list((None, self.nb_actions)):
        #     raise ValueError(
        #         'Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(
        #             model.output, self.nb_actions))

        # Parameters.
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type
        if self.enable_dueling_network:
            # get the second last layer of the model, abandon the last layer
            layer = model.layers[-2]
            nb_action = model.output.shape[-1]
            # layer y has a shape (nb_action+1,)
            # y[:,0] represents V(s;theta)
            # y[:,1:] represents A(s,a;theta)
            y = Dense(nb_action + 1, activation='linear')(layer.output)
            # caculate the Q(s,a;theta)
            # dueling_type == 'avg'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            # dueling_type == 'max'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            # dueling_type == 'naive'
            # Q(s,a;theta) = V(s;theta) + A(s,a;theta)
            if self.dueling_type == 'avg':
                outputlayer = Lambda(
                    lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True),
                    output_shape=(nb_action,))(y)
            elif self.dueling_type == 'max':
                outputlayer = Lambda(
                    lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], axis=1, keepdims=True),
                    output_shape=(nb_action,))(y)
            elif self.dueling_type == 'naive':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)
            else:
                assert False, "dueling_type must be one of {'avg','max','naive'}"

            model = Model(inputs=model.input, outputs=outputlayer)

        # Related objects.
        self.model = model
        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy

        # State.
        self.reset_states()

        # 逆標準化のための下準備
        self.mm = MinMaxScaler()
        min_job_queue = [0, 0, 0]
        max_job_queue = [10, 10, 1]
        hoge = [min_job_queue, max_job_queue]
        normalized_hoge = self.mm.fit_transform(hoge)

        # SchedulingEnvと共通のパラメータ
        self.n_window = n_window  # スライドウィンドウの横幅
        self.n_on_premise_node = n_on_premise_node  # オンプレミス計算資源のノード数
        self.n_cloud_node = n_cloud_node  # クラウド計算資源のノード数
        self.n_job_queue_obs = n_job_queue_obs  # ジョブキューの観測部分の長さ
        self.n_job_queue_bck = n_job_queue_bck  # ジョブキューのバックログ部分の長さ

        #logにはwandbを使う.wandb.logで記録する
        wandb.init(project='dqn-scheduling')
        wandb.define_metric('waiting_time_sum', step_metric='episode_cost_without_minus')

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_error(args):
            y_true, y_pred, mask = args
            loss = huber_loss(y_true, y_pred, self.delta_clip)
            loss *= mask  # apply element-wise mask
            return K.sum(loss, axis=-1)

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_pred_1 = self.model.output[0]
        y_pred_2 = self.model.output[1]
        y_true_1 = Input(name='y_true_1', shape=(self.nb_actions,))
        y_true_2 = Input(name='y_true_2', shape=(self.nb_actions,))
        mask_1 = Input(name='mask_1', shape=(self.nb_actions,))
        mask_2 = Input(name='mask_2', shape=(self.nb_actions,))
        loss_out_1 = Lambda(clipped_masked_error, output_shape=(1,), name='loss_1')([y_true_1, y_pred_1, mask_1])
        loss_out_2 = Lambda(clipped_masked_error, output_shape=(1,), name='loss_2')([y_true_2, y_pred_2, mask_2])
        
        #modelの定義
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        trainable_model = Model(inputs=ins + [y_true_1, mask_1, y_true_2, mask_2], outputs = [loss_out_1, y_pred_1, loss_out_2, y_pred_2])
        
        combined_metrics = {
        trainable_model.output_names[1]: metrics,
        trainable_model.output_names[3]: metrics,  # 仮に同じメトリクスを使用
    }
        losses = [
            lambda y_true, y_pred: y_pred,  # 1つ目の出力の損失
            lambda y_true, y_pred: K.zeros_like(y_pred),  # メトリクス用
            lambda y_true, y_pred: y_pred,  # 2つ目の出力の損失
            lambda y_true, y_pred: K.zeros_like(y_pred),  # メトリクス用
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

        self.compiled = True

    def is_dominated(self,x, y):
        """ xがyに支配されているかどうかを判定する (numpyバージョン) """
        return np.all(x <= y) and np.any(x < y)

    def find_pareto_front_with_kdtree(self,points):
        tree = KDTree(points)
        pareto_front = []

        for i, point in enumerate(points):
            neighbors = tree.query_ball_point(point, r=100)  # 近傍点を検索
            if not any(self.is_dominated(points[neighbor], point) for neighbor in neighbors if neighbor != i):
                pareto_front.append(point)

        return np.array(pareto_front)
    
    def get_hypervolume(self, points):
        #Class HyperVolumeを用いて，pointsのhypervolumeを計算する
        reference_point = np.array([1000, 1000])
        hv = HyperVolume(reference_point)
        #pointにlenが適応できるようにする
        points = np.array(points)
        volume = hv.compute(points)
        return volume

    def calc_dist_from_pareto_front(self, points, point):
        #pointとpareto_frontとの距離を計算する
        pareto_fronts = self.find_pareto_front_with_kdtree(points)
        dist = np.inf
        for pareto_front in pareto_fronts:
            dist = min(dist, np.linalg.norm(pareto_front - point))
        return dist

    def is_neigbor_parero_front(self, points, point):
        #pointがpareto_frontのk近傍点かどうかを判定する
        k = 10
        pareto_fronts = self.find_pareto_front_with_kdtree(points)
        dist = np.inf
        for pareto_front in pareto_fronts:
            dist = min(dist, np.linalg.norm(pareto_front - point))
        if dist < k:
            return True
        else:
            return False
        
    def rank_of_pareto_dist(self, points, point):
        #pointからpareto_fromtが何番目に近いかを計算する
        pareto_fronts = self.find_pareto_front_with_kdtree(points)
        

        
    def compute_q_values_4_test(self, state):
        #         print(np.array(state).shape)
        #         print(state)
        #         print('state: ' + str(state))
        # ↓stateがもともとの分け目を記憶している状態なので普通にfor文で4つに分けられるかも？<-記憶してないから無理
        on_premise_window, cloud_window, job_queue_obs, n_job_in_job_queue_bck = np.split(state[0], [
            self.n_window * self.n_on_premise_node,
            self.n_window * self.n_on_premise_node + self.n_window * self.n_cloud_node,
            self.n_window * self.n_on_premise_node + self.n_window * self.n_cloud_node + self.n_job_queue_obs * 3])
        #         on_premise_window = state[:n_window*n_on_premise_node].reshape(n_on_premise_node, n_window)
        #         cloud_window = state[n_window*n_on_premise_node:n_window*n_on_premise_node+n_window*n_cloud_node].reshape(n_cloud_node, n_window)
        #         job_queue_obs = state[n_window*n_on_premise_node+n_window*n_cloud_node:n_window*n_on_premise_node+n_window*n_cloud_node+]
        #         n_job_queue_bck =
        #         print(on_premise_window, cloud_window, job_queue_obs, n_job_in_job_queue_bck)
        on_premise_window = on_premise_window.reshape(self.n_on_premise_node, self.n_window)
        cloud_window = cloud_window.reshape(self.n_cloud_node, self.n_window)
        job_queue_obs = job_queue_obs.reshape(self.n_job_queue_obs, 3)
        job_queue_obs = self.mm.inverse_transform(job_queue_obs)
        n_job_in_job_queue_bck *= self.n_job_queue_bck
        #         print('-------標準化後のstateから復元した標準化前の観測データ-------')
        #         print(on_premise_window, cloud_window, job_queue_obs, n_job_in_job_queue_bck)
        #         print('------------------------------------------------------')
        q_values = self.compute_batch_q_values([state]).flatten()
        #         print(np.array(q_values).shape)
        #         print(q_values)
        assert q_values.shape == (self.nb_actions,)

        # 無効なactionのq値を-infに書き換える
        for action in range(self.nb_actions):
            if not self.check_is_valid_action(action, on_premise_window, cloud_window, job_queue_obs[0]):
                q_values[action] = -np.inf
        #         print('q_values: \n' + str(q_values))
        return q_values

    # stateをもとにactionが有効かどうかを判定
    def check_is_valid_action(self, action, on_premise_window, cloud_window, head_job):
        job_width = int(head_job[0])
        job_height = int(head_job[1])
        can_use_cloud = int(head_job[2])
        action = self.get_converted_action(action)
        when_allocate, use_cloud = action
        # is_valid = True # actionが有効かどうか

        # actionが有効かどうかを判定
        if when_allocate == -1 and use_cloud == -1:  # 割り当てないというactionの場合
            is_valid = True
        elif not use_cloud:  # オンプレミスに割り当てる場合
            if when_allocate + job_width - 1 < self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                is_valid = False  # 暫定
                for i in range(self.n_on_premise_node):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                    part_matrix = on_premise_window.take(range(i, i + job_height), mode='wrap', axis=0).take(
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
                        part_matrix = cloud_window.take(range(i, i + job_height), mode='wrap', axis=0).take(
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

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        """Trains the agent on the given environment.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        reward_hold=[]

        # SchedulingEnv側で使うパラメータ
        if env.job_type == 4:
            env.point_switch_random_job_type = (nb_steps // nb_max_episode_steps) // 2
            print(str(env.point_switch_random_job_type) + 'エピソード目で切り替わる')

        if not self.compiled:
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = True

        history = History()

        episode = np.int16(0)
        self.step = np.int16(0)
        vector_for_hypervolume = [0,0]
        hv_hold = []
        vector_sum = []        
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        result_vector = []
        try:
            while self.step < nb_steps:
 
                if observation is None:  # start of a new episode

                    # 環境にepisodeを教える
                    env.episode = episode

                    #log用の変数をリセット
                    episode_step = np.int16(0)
                    episode_step_without_invalid = np.int16(0)
                    episode_reward = np.float32(0)
                    episode_cost = np.float32(0)
                    episode_time = np.float32(0)
                    episode_loss = np.float32(0)
                    hypervolume = np.float32(0)
                    action_list = []
                    reward = [0, 0]
                    episode_cost_without_minus = np.float32(0)
                    reward_final = np.float32(0)
                    episode_cost_hold = []
                    episode_cost_for_episode = np.float32(0)
                    hint = np.float32(0)
                    dist_reward = np.float32(0)
                    

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    assert observation is not None

                    # Perform random starts at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.
                    nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
                    for _ in range(nb_random_start_steps):
                        if start_step_policy is None:
                            action = env.action_space.sample()
                        else:
                            action = start_step_policy(observation)
                        if self.processor is not None:
                            action = self.processor.process_action(action)
                        observation, reward, cost, time, done, info = env.step(action)


                        observation = deepcopy(observation)
                        if self.processor is not None:
                            observation, reward, done, info = self.processor.process_step(observation, reward, done,
                                                                                          info)
                        if done:
                            warnings.warn(
                                'Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                                    nb_random_start_steps))
                            observation = deepcopy(env.reset())
                            if self.processor is not None:
                                observation = self.processor.process_observation(observation)
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).

                action, q_values = self.forward(observation)#行動を決定
                if self.processor is not None:
                    action = self.processor.process_action(action)
                action_list.append(action)
                cost = 0
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):#while done == False:に変更
                    observation, r, cost, time, done, info = env.step(action)#行動を実行
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(observation, r, done, info)
                    #rewardのそれぞれの要素にrのそれぞれの要素を足し算する
                    reward = [reward[i] + r[i] for i in range(len(r))]



                    if done:
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                metrics = self.backward(0, terminal=done)

                #rewardを記録
                # if (reward[0] == -env.penalty_invalid_action or reward[1] == -env.penalty_invalid_action) == False and reward not in reward_record:
                #     reward_record.append(reward)
                # print(reward_record)

                #もしreward[x,x]の全要素が-5でなければepisode_reward+=reward
                # if np.all(reward == env.penalty_invalid_action) == False:
                episode_reward += reward

                #costが-1でなければコスト用の変数に加算
                if cost != -1:
                    episode_cost_without_minus += cost
                    episode_step_without_invalid += 1
                episode_cost += cost

                episode_time += time


                # costが-1でなければepisode_step+=1
                    
                episode_step += 1
                self.step += 1
                # if time != -1:
                episode_time += time
                episode_loss += metrics[0]
                

                if done: # episode終了時
                    self.forward(observation)
                    self.backward(0., terminal=False)
                    # print("Actions taken in this episode: ", action_list)
                    action_list = []  # リストをリセット

                    #waitingtimeを計算する
                    i=0
                    p=0
                    end_time = time
                    waiting_time_sum = 0
                    user_wt_var = 0
                    cnt_not_allocated = 0
                    #user_wtを0で初期化
                    user_wt = [0 for p in range(6)]
                    # print('env.jobs: ' + str(env.jobs))
                          

                    while True:
                        #待ち時間を保持する変数
                        if i < len(env.jobs):  # そのエピソードで生成されたジョブの中でまだ見ていないジョブがある場合
                            # print('env.jobs[i]: ' + str(env.jobs[i]))
                            submitted_time = env.jobs[i][0]
                            if submitted_time <= end_time:  # ジョブが提出時刻を迎えていた場合
                                waiting_time = env.jobs[i][-1]
                                if waiting_time == -1:  # 割り当てられていなかった場合
                                    waiting_time = 100 # 待ち時間を大きめに設定
                                    cnt_not_allocated += 1
                                waiting_time_sum += waiting_time    # 待ち時間を記録
                                user_wt[env.jobs[i][4]] += waiting_time
                                i += 1
                                # print('waiting_time' + str(waiting_time))
                            else:  # ジョブが提出時刻を迎えていなかった場合
                                break
                        else:  # そのエピソードで生成されたジョブを全て見た場合
                            break

                    #user_wtの分散を求める
                    user_wt_var = np.var(user_wt)
                    #user_wtを正規化する

                    # print('user_wt: ' + str(user_wt))



                    

                    #2episode目からbackwardを行う。ただし1回目はbackwardの1変数目に0を入れる

                    vector_for_hypervolume = [episode_cost_without_minus, int(waiting_time_sum/10)]#適当な数字を入れておく
                    vector_sum.append(vector_for_hypervolume)
                    #hint = vector_for_hypervolumeの各要素を掛け合わせる
                    hint = vector_for_hypervolume[0] * vector_for_hypervolume[1]
                    # print("vector_sum: " + str(vector_sum))
                    hypervolume = self.get_hypervolume(vector_sum)
                    hv_hold.append(hypervolume)
                    dist_reward = self.calc_dist_from_pareto_front(vector_sum, vector_for_hypervolume)
                    print("dist_reward" + str(dist_reward))
                    #hvの最後尾とそのひとつ前の引き算をする。ただし、hvの要素数が1の場合は0を返す
                    if len(hv_hold) == 1:
                        hv_diff = 0
                    else:
                        hv_diff = hv_hold[-1] - hv_hold[-2]
                        print("hv_diff: " + str(hv_diff))
                    #hvが正の値であれば報酬を与える。そうでなければ負の報酬を与える
                    if hv_diff > 0:
                        reward_final += (1)
                        self.backward(1, terminal=False)
                        
                    else:
                        if self.is_neigbor_parero_front(vector_sum, vector_for_hypervolume):
                            reward_final += (0.2)
                            self.backward(0.2, terminal=False)
                        else:
                            reward_final += (0)
                            self.backward(0, terminal=False)
                    # print("vector_sum: " + str(vector_sum))
                    

                    
                    on_premise_window_history, cloud_window_history = env.get_window_history()
                # ウィンドウの履歴をCSVファイルに出力
                    
                    # print(on_premise_window_history.tolist())
                    # ウィンドウの履歴をリセット
                    env.reset_window_history()




                    

                    # wandbにログを保存
                    # wandb.log({"my_step":episode_step, "my_step_without_invalid,":episode_step_without_invalid,
                    #            "episode_loss":episode_loss/episode_step,'episode_reward': episode_reward/episode_step_without_invalid,"episode_cost":episode_cost/episode_step_without_invalid, "episode_time":episode_time/episode_step, 'nb_episode_steps': episode_step, 'nb_steps': self.step, 'end_time': time, 'jobs': env.jobs, 'waiting_time': waiting_time_sum/episode_step})

                    wandb.log({"my_step":episode_step, "my_step_without_invalid,":episode_step_without_invalid,
                               "episode_loss":episode_loss/episode_step,'episode_reward_cost': episode_reward[0],'epi_co_mean':episode_reward[0]/episode_step,"episode_reward_wt":episode_reward[1]/episode_step_without_invalid,"episode_cost":episode_cost/episode_step_without_invalid, "episode_time":episode_time/episode_step, 'nb_episode_steps': episode_step, 'nb_steps': self.step, 'end_time': time, 'jobs': env.jobs, 'waiting_time': waiting_time_sum/episode_step,"hypervolume":hypervolume,"hv_diff":hv_diff,"episode_cost_without_minus":episode_cost_without_minus,"waiting_time_sum":waiting_time_sum,"reward_final":reward_final,'hint':hint,'dist_reward':dist_reward,'user_wt_var':user_wt_var,'cnt_not_allocated':cnt_not_allocated})

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None
                    episode_cost = None
                    episode_time = None
                    episode_loss = None
                    episode_cost_hold = None

        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        self._on_train_end()

        print("vector_sum: " + str(vector_sum))

        return history

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        """Callback that is called before training begins.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_episodes (integer): Number of episodes to perform.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError(
                'Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)
            episode_reward = 0.
            episode_step = 0

            # 環境にepisodeを教える
            env.episode = episode

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, reward, cost, time, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(observation, r, done, info)
                callbacks.on_action_end(action)
                if done:
                    warnings.warn(
                        'Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                            nb_random_start_steps))
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    break

            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)

                action, q_values = self.forward_4_test(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, cost, time, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, reward, cost, time, done, info = self.processor.process_step(observation, r, d,
                                                                                                  info)
                    callbacks.on_action_end(action)
                    reward += r
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                    # 自作
                    'cost': cost,
                    'time': time,
                    'q_values': q_values,
                    #                     'waiting_time': waiting_time,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(observation)
            self.backward(0., terminal=False)

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
                # 自作
                'end_time': time,
                'jobs': env.jobs,
            }
            callbacks.on_episode_end(episode, episode_logs)
        callbacks.on_train_end()
        self._on_test_end()

        return history

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values_cost, q_values_wt = self.compute_q_values(state)
        #q_valuesはそれぞれの線形結合
        q_values = [q_values_cost[i] + q_values_wt[i] for i in range(len(q_values_cost))] 

        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action, q_values

    def forward_4_test(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_values_4_test(state)

        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action, q_values


# 待ち時間の最小化のみを目的とした行動を取るエージェント
class WTPAgent(Agent):
    def __init__(self, nb_actions, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck):
        self.nb_actions = nb_actions
        self.processor = None

        # SchedulingEnvと共通のパラメータ
        self.n_window = n_window  # スライドウィンドウの横幅
        self.n_on_premise_node = n_on_premise_node  # オンプレミス計算資源のノード数
        self.n_cloud_node = n_cloud_node  # クラウド計算資源のノード数
        self.n_job_queue_obs = n_job_queue_obs  # ジョブキューの観測部分の長さ
        self.n_job_queue_bck = n_job_queue_bck  # ジョブキューのバックログ部分の長さ

        # 逆標準化のための下準備
        self.mm = MinMaxScaler()
        min_job_queue = [0, 0, 0]
        max_job_queue = [10, 10, 1]
        hoge = [min_job_queue, max_job_queue]
        normalized_hoge = self.mm.fit_transform(hoge)

    def forward(self, observation):
        on_premise_window, cloud_window, job_queue_obs, n_job_in_job_queue_bck = np.split(observation, [
            self.n_window * self.n_on_premise_node,
            self.n_window * self.n_on_premise_node + self.n_window * self.n_cloud_node,
            self.n_window * self.n_on_premise_node + self.n_window * self.n_cloud_node + self.n_job_queue_obs * 3])
        on_premise_window = on_premise_window.reshape(self.n_on_premise_node, self.n_window)
        cloud_window = cloud_window.reshape(self.n_cloud_node, self.n_window)
        job_queue_obs = job_queue_obs.reshape(self.n_job_queue_obs, 3)
        job_queue_obs = self.mm.inverse_transform(job_queue_obs)
        n_job_in_job_queue_bck *= self.n_job_queue_bck

        valid_actions = []  # 有効なactionのリスト
        # 有効なactionのリストを得る
        for action in range(self.nb_actions):
            if self.check_is_valid_action(action, on_premise_window, cloud_window, job_queue_obs[0]):
                valid_actions.append(action)

        action_when_allocate_list = [i % self.n_window for i in range(self.nb_actions)]  # 各actionに対応する割り当て時間を格納したリスト
        for action in range(self.nb_actions):
            if not self.check_is_valid_action(action, on_premise_window, cloud_window,
                                              job_queue_obs[0]):  # 無効なactionの場合
                action_when_allocate_list[action] = self.n_window  # argminで選択されないような大きな値にする
            elif action == self.nb_actions - 1:  # 割り当てないactionの場合
                action_when_allocate_list[action] = self.n_window - 1  # argminで選択されうる最大の値にする

        #         print('action_when_allocate_list: ' + str(action_when_allocate_list))

        # 有効なactionのうち，割り当て時間が最小になるようなactionを選択
        action = np.argmin(action_when_allocate_list)

        return action

    def backward(self):
        pass

    def compile(self):
        pass

    def load_weights(self):
        pass

    def save_weights(self):
        pass

    def layers(self):
        pass

    # stateをもとにactionが有効かどうかを判定
    def check_is_valid_action(self, action, on_premise_window, cloud_window, head_job):
        job_width = int(head_job[0])
        job_height = int(head_job[1])
        can_use_cloud = int(head_job[2])
        action = self.get_converted_action(action)
        when_allocate, use_cloud = action

        # actionが有効かどうかを判定
        if when_allocate == -1 and use_cloud == -1:  # 割り当てないというactionの場合
            is_valid = True
        elif not use_cloud:  # オンプレミスに割り当てる場合
            if when_allocate + job_width - 1 < self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                is_valid = False  # 暫定
                for i in range(self.n_on_premise_node):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                    part_matrix = on_premise_window.take(range(i, i + job_height), mode='wrap', axis=0).take(
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
                        part_matrix = cloud_window.take(range(i, i + job_height), mode='wrap', axis=0).take(
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

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        """Trains the agent on the given environtrint.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        #         if not self.compiled:
        #             raise RuntimeError(
        #                 'Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')

        # SchedulingEnv側で使うパラメータ
        if env.job_type == 4:
            env.point_switch_random_job_type = (nb_steps // nb_max_episode_steps) // 2
            print(str(env.point_switch_random_job_type) + 'エピソード目で切り替わる')

        if action_repetition < 1:
            raise ValueError(
                'action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        episode = np.int16(0)
        self.step = np.int16(0)
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                if observation is None:  # start of a new episode

                    callbacks.on_episode_begin(episode)
                    # 環境にepisodeを教える
                    env.episode = episode

                    episode_step = np.int16(0)
                    episode_reward = np.float32(0)

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(
                            observation)
                    assert observation is not None

                    # Perform random starts at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.
                    nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(
                        nb_max_start_steps)
                    for _ in range(nb_random_start_steps):
                        if start_step_policy is None:
                            action = env.action_space.sample()
                        else:
                            action = start_step_policy(observation)
                        if self.processor is not None:
                            action = self.processor.process_action(action)
                        callbacks.on_action_begin(action)
                        observation, reward, cost, time, done, info = env.step(action)
                        observation = deepcopy(observation)
                        if self.processor is not None:
                            observation, reward, done, info = self.processor.process_step(
                                observation, reward, done, info)
                        callbacks.on_action_end(action)
                        if done:
                            warnings.warn(
                                'Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                                    nb_random_start_steps))
                            observation = deepcopy(env.reset())
                            if self.processor is not None:
                                observation = self.processor.process_observation(
                                    observation)
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = np.float32(0)
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, cost, time, done, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(
                            observation, r, done, info)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(action)
                    reward += r
                    if done:
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                #                 metrics = self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': [],
                    'episode': episode,
                    'info': accumulated_info,
                    # 自作
                    'cost': cost,
                    'time': time,
                    #                     'waiting_time': waiting_time,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.forward(observation)
                    #                     self.backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                        # 自作
                        'end_time': time,
                        'jobs': env.jobs,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return history

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        """Callback that is called before training begins.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_episodes (integer): Number of episodes to perform.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        #         if not self.compiled:
        #             raise RuntimeError(
        #                 'Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
        if action_repetition < 1:
            raise ValueError(
                'action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)

            # 環境にepisodeを教える
            env.episode = episode

            episode_reward = 0.
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(
                nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, r, cost, time, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(
                        observation, r, done, info)
                callbacks.on_action_end(action)
                if done:
                    warnings.warn(
                        'Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                            nb_random_start_steps))
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(
                            observation)
                    break

            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)

                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, cost, time, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, d, info = self.processor.process_step(
                            observation, r, d, info)
                    callbacks.on_action_end(action)
                    reward += r
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                #                 self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                    # 自作
                    'cost': cost,
                    'time': time,
                    #                     'waiting_time': waiting_time,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(observation)
            #             self.backward(0., terminal=False)

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
                # 自作
                'end_time': time,
                'jobs': env.jobs,
            }
            callbacks.on_episode_end(episode, episode_logs)
        callbacks.on_train_end()
        self._on_test_end()

        return history


# クラウド利用コストの最小化のみを目的とした行動を取るエージェント
class CostPAgent(Agent):
    def __init__(self, nb_actions, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck):
        self.nb_actions = nb_actions
        self.processor = None

        # SchedulingEnvと共通のパラメータ
        self.n_window = n_window  # スライドウィンドウの横幅
        self.n_on_premise_node = n_on_premise_node  # オンプレミス計算資源のノード数
        self.n_cloud_node = n_cloud_node  # クラウド計算資源のノード数
        self.n_job_queue_obs = n_job_queue_obs  # ジョブキューの観測部分の長さ
        self.n_job_queue_bck = n_job_queue_bck  # ジョブキューのバックログ部分の長さ

        # 逆標準化のための下準備
        self.mm = MinMaxScaler()
        min_job_queue = [0, 0, 0]
        max_job_queue = [10, 10, 1]
        hoge = [min_job_queue, max_job_queue]
        normalized_hoge = self.mm.fit_transform(hoge)

    def forward(self, observation):
        on_premise_window, cloud_window, job_queue_obs, n_job_in_job_queue_bck = np.split(observation, [
            self.n_window * self.n_on_premise_node,
            self.n_window * self.n_on_premise_node + self.n_window * self.n_cloud_node,
            self.n_window * self.n_on_premise_node + self.n_window * self.n_cloud_node + self.n_job_queue_obs * 3])
        on_premise_window = on_premise_window.reshape(self.n_on_premise_node, self.n_window)
        cloud_window = cloud_window.reshape(self.n_cloud_node, self.n_window)
        job_queue_obs = job_queue_obs.reshape(self.n_job_queue_obs, 3)
        job_queue_obs = self.mm.inverse_transform(job_queue_obs)
        n_job_in_job_queue_bck *= self.n_job_queue_bck

        # オンプレミス->クラウド->割り当てないの順にactionが有効か確かめて有効ならばそれをactionとして採用->オンプレミスに割り当てられる場合は必ずオンプレミスに割り当てる
        for a in range(self.nb_actions):
            if self.check_is_valid_action(a, on_premise_window, cloud_window, job_queue_obs[0]):
                action = a
                break

        return action

    def backward(self):
        pass

    def compile(self):
        pass

    def load_weights(self):
        pass

    def save_weights(self):
        pass

    def layers(self):
        pass

    # stateをもとにactionが有効かどうかを判定
    def check_is_valid_action(self, action, on_premise_window, cloud_window, head_job):
        job_width = int(head_job[0])
        job_height = int(head_job[1])
        can_use_cloud = int(head_job[2])
        action = self.get_converted_action(action)
        when_allocate, use_cloud = action

        # actionが有効かどうかを判定
        if when_allocate == -1 and use_cloud == -1:  # 割り当てないというactionの場合
            is_valid = True
        elif not use_cloud:  # オンプレミスに割り当てる場合
            if when_allocate + job_width - 1 < self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                is_valid = False  # 暫定
                for i in range(self.n_on_premise_node):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                    part_matrix = on_premise_window.take(range(i, i + job_height), mode='wrap', axis=0).take(
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
                        part_matrix = cloud_window.take(range(i, i + job_height), mode='wrap', axis=0).take(
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

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        """Trains the agent on the given environtrint.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        #         if not self.compiled:
        #             raise RuntimeError(
        #                 'Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')

        # SchedulingEnv側で使うパラメータ
        if env.job_type == 4:
            env.point_switch_random_job_type = (nb_steps // nb_max_episode_steps) // 2
            print(str(env.point_switch_random_job_type) + 'エピソード目で切り替わる')

        if action_repetition < 1:
            raise ValueError(
                'action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        episode = np.int16(0)
        self.step = np.int16(0)
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                if observation is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)

                    # 環境にepisodeを教える
                    env.episode = episode

                    episode_step = np.int16(0)
                    episode_reward = np.float32(0)

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(
                            observation)
                    assert observation is not None

                    # Perform random starts at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.
                    nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(
                        nb_max_start_steps)
                    for _ in range(nb_random_start_steps):
                        if start_step_policy is None:
                            action = env.action_space.sample()
                        else:
                            action = start_step_policy(observation)
                        if self.processor is not None:
                            action = self.processor.process_action(action)
                        callbacks.on_action_begin(action)
                        observation, reward, cost, time, done, info = env.step(action)
                        observation = deepcopy(observation)
                        if self.processor is not None:
                            observation, reward, done, info = self.processor.process_step(
                                observation, reward, done, info)
                        callbacks.on_action_end(action)
                        if done:
                            warnings.warn(
                                'Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                                    nb_random_start_steps))
                            observation = deepcopy(env.reset())
                            if self.processor is not None:
                                observation = self.processor.process_observation(
                                    observation)
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = np.float32(0)
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, cost, time, done, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(
                            observation, r, done, info)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(action)
                    reward += r
                    if done:
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                #                 metrics = self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': [],
                    'episode': episode,
                    'info': accumulated_info,
                    # 自作
                    'cost': cost,
                    'time': time,
                    #                     'waiting_time': waiting_time,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.forward(observation)
                    #                     self.backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                        # 自作
                        'end_time': time,
                        'jobs': env.jobs,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return history

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        """Callback that is called before training begins.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_episodes (integer): Number of episodes to perform.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        #         if not self.compiled:
        #             raise RuntimeError(
        #                 'Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
        if action_repetition < 1:
            raise ValueError(
                'action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)

            # 環境にepisodeを教える
            env.episode = episode

            episode_reward = 0.
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(
                nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, r, cost, time, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(
                        observation, r, done, info)
                callbacks.on_action_end(action)
                if done:
                    warnings.warn(
                        'Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                            nb_random_start_steps))
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(
                            observation)
                    break

            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)

                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, cost, time, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, d, info = self.processor.process_step(
                            observation, r, d, info)
                    callbacks.on_action_end(action)
                    reward += r
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                #                 self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                    # 自作
                    'cost': cost,
                    'time': time,
                    #                     'waiting_time': waiting_time,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(observation)
            #             self.backward(0., terminal=False)

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
                # 自作
                'end_time': time,
                'jobs': env.jobs,
            }
            callbacks.on_episode_end(episode, episode_logs)
        callbacks.on_train_end()
        self._on_test_end()

        return history


# 有効な行動からランダムに選んだ行動を取るエージェント
class RandomAgent(Agent):
    def __init__(self, nb_actions, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck):
        self.nb_actions = nb_actions
        self.processor = None

        # SchedulingEnvと共通のパラメータ
        self.n_window = n_window  # スライドウィンドウの横幅
        self.n_on_premise_node = n_on_premise_node  # オンプレミス計算資源のノード数
        self.n_cloud_node = n_cloud_node  # クラウド計算資源のノード数
        self.n_job_queue_obs = n_job_queue_obs  # ジョブキューの観測部分の長さ
        self.n_job_queue_bck = n_job_queue_bck  # ジョブキューのバックログ部分の長さ

        # 逆標準化のための下準備
        self.mm = MinMaxScaler()
        min_job_queue = [0, 0, 0]
        max_job_queue = [10, 10, 1]
        hoge = [min_job_queue, max_job_queue]
        normalized_hoge = self.mm.fit_transform(hoge)

    def forward(self, observation):
        on_premise_window, cloud_window, job_queue_obs, n_job_in_job_queue_bck = np.split(observation, [
            self.n_window * self.n_on_premise_node,
            self.n_window * self.n_on_premise_node + self.n_window * self.n_cloud_node,
            self.n_window * self.n_on_premise_node + self.n_window * self.n_cloud_node + self.n_job_queue_obs * 3])
        on_premise_window = on_premise_window.reshape(self.n_on_premise_node, self.n_window)
        cloud_window = cloud_window.reshape(self.n_cloud_node, self.n_window)
        job_queue_obs = job_queue_obs.reshape(self.n_job_queue_obs, 3)
        job_queue_obs = self.mm.inverse_transform(job_queue_obs)
        n_job_in_job_queue_bck *= self.n_job_queue_bck

        valid_actions = []  # 有効なactionのリスト
        # オンプレミス->クラウド->割り当てないの順にactionが有効か確かめて有効ならばそれをactionとして採用->オンプレミスに割り当てられる場合は必ずオンプレミスに割り当てる
        for a in range(self.nb_actions):
            if self.check_is_valid_action(a, on_premise_window, cloud_window, job_queue_obs[0]):
                valid_actions.append(a)
        # 有効なactionのリストからランダムにactionを選択
        action = random.choice(valid_actions)

        return action

    def backward(self):
        pass

    def compile(self):
        pass

    def load_weights(self):
        pass

    def save_weights(self):
        pass

    def layers(self):
        pass

    # stateをもとにactionが有効かどうかを判定
    def check_is_valid_action(self, action, on_premise_window, cloud_window, head_job):
        job_width = int(head_job[0])
        job_height = int(head_job[1])
        can_use_cloud = int(head_job[2])
        action = self.get_converted_action(action)
        when_allocate, use_cloud = action

        # actionが有効かどうかを判定
        if when_allocate == -1 and use_cloud == -1:  # 割り当てないというactionの場合
            is_valid = True
        elif not use_cloud:  # オンプレミスに割り当てる場合
            if when_allocate + job_width - 1 < self.n_window:  # 割り当てるジョブがスライドウィンドウの右にはみ出なければ
                is_valid = False  # 暫定
                for i in range(self.n_on_premise_node):  # ウィンドウを縦に見ていき、空いている部分行列を探す
                    part_matrix = on_premise_window.take(range(i, i + job_height), mode='wrap', axis=0).take(
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
                        part_matrix = cloud_window.take(range(i, i + job_height), mode='wrap', axis=0).take(
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

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        """Trains the agent on the given environtrint.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        #         if not self.compiled:
        #             raise RuntimeError(
        #                 'Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')

        # SchedulingEnv側で使うパラメータ
        if env.job_type == 4:
            env.point_switch_random_job_type = (nb_steps // nb_max_episode_steps) // 2
            print(str(env.point_switch_random_job_type) + 'エピソード目で切り替わる')

        if action_repetition < 1:
            raise ValueError(
                'action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        episode = np.int16(0)
        self.step = np.int16(0)
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                if observation is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)

                    # 環境にepisodeを教える
                    env.episode = episode

                    episode_step = np.int16(0)
                    episode_reward = np.float32(0)

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(
                            observation)
                    assert observation is not None

                    # Perform random starts at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.
                    nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(
                        nb_max_start_steps)
                    for _ in range(nb_random_start_steps):
                        if start_step_policy is None:
                            action = env.action_space.sample()
                        else:
                            action = start_step_policy(observation)
                        if self.processor is not None:
                            action = self.processor.process_action(action)
                        callbacks.on_action_begin(action)
                        observation, reward, cost, time, done, info = env.step(action)
                        observation = deepcopy(observation)
                        if self.processor is not None:
                            observation, reward, done, info = self.processor.process_step(
                                observation, reward, done, info)
                        callbacks.on_action_end(action)
                        if done:
                            warnings.warn(
                                'Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                                    nb_random_start_steps))
                            observation = deepcopy(env.reset())
                            if self.processor is not None:
                                observation = self.processor.process_observation(
                                    observation)
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = np.float32(0)
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, cost, time, done, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(
                            observation, r, done, info)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(action)
                    reward += r
                    if done:
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                #                 metrics = self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': [],
                    'episode': episode,
                    'info': accumulated_info,
                    # 自作
                    'cost': cost,
                    'time': time,
                    #                     'waiting_time': waiting_time,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.forward(observation)
                    #                     self.backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                        # 自作
                        'end_time': time,
                        'jobs': env.jobs,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return history

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        """Callback that is called before training begins.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_episodes (integer): Number of episodes to perform.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        #         if not self.compiled:
        #             raise RuntimeError(
        #                 'Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
        if action_repetition < 1:
            raise ValueError(
                'action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)

            # 環境にepisodeを教える
            env.episode = episode

            episode_reward = 0.
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(
                nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, r, cost, time, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(
                        observation, r, done, info)
                callbacks.on_action_end(action)
                if done:
                    warnings.warn(
                        'Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                            nb_random_start_steps))
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(
                            observation)
                    break

            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)

                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, cost, time, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, d, info = self.processor.process_step(
                            observation, r, d, info)
                    callbacks.on_action_end(action)
                    reward += r
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                #                 self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                    # 自作
                    'cost': cost,
                    'time': time,
                    #                     'waiting_time': waiting_time,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(observation)
            #             self.backward(0., terminal=False)

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
                # 自作
                'end_time': time,
                'jobs': env.jobs,
            }
            callbacks.on_episode_end(episode, episode_logs)
        callbacks.on_train_end()
        self._on_test_end()

        return history