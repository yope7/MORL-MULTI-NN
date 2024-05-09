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
from rl.memory import SequentialMemory
import keras.backend as K
from keras.models import Model, model_from_config
from keras.layers import Lambda, Input, Layer, Dense
from tensorflow.keras.optimizers import Adam
import statistics

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

# 省略せずに全て表示
np.set_printoptions(threshold=np.inf)

def get_soft_target_model_updates(target, source, tau):
    target_weights = target.trainable_weights + sum([l.non_trainable_weights for l in target.layers], [])
    source_weights = source.trainable_weights + sum([l.non_trainable_weights for l in source.layers], [])
    assert len(target_weights) == len(source_weights)

    # Create updates.
    updates = []
    for tw, sw in zip(target_weights, source_weights):
        updates.append((tw, tau * sw + (1. - tau) * tw))
    return updates

def clone_model(model, custom_objects={}):
    # Requires Keras 1.0.7 since get_config has breaking changes.
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone



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
    def __init__(self, model1, model2, model3, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck,policy, nb_episodes_warmup,object_mode,
                 test_policy=None, enable_double_dqn = False, enable_dueling_network=False,
                 dueling_type='avg',  *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

                # Related objects.
        self.models = [model1, model2, model3]
        self.object_mode = object_mode

        # Validate (important) input.
        # if list(model.output.shape) != list((None, self.nb_actions)):
        #     raise ValueError(
        #         'Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(
        #             model.output, self.nb_actions))
        self.batch_size = 32
        # Parameters.

        self.target_model_update = 1e-2

        self.memory1 = SequentialMemory(limit=50000, window_length=1)
        self.memory2 = SequentialMemory(limit=50000, window_length=1)
        self.memory3 = SequentialMemory(limit=50000, window_length=1)

        self.nb_episodes_warmup = nb_episodes_warmup

        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type
        if self.enable_dueling_network:
            # get the second last layer of the model, abandon the last layer
            layer = self.models[0].layers[-2]
            nb_action = self.models[0].output.shape[-1]
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



        # if policy is None:
        #     policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy

        # State.
        # self.reset_states()
        self.reset_states(self.models[0])
        self.reset_states(self.models[1])
        self.reset_states(self.models[2])


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
        wandb.init(project='last2',name=self.object_mode)

    def compile(self, optimizer, metrics=[]):
        optimizer1=Adam(learning_rate=1e-3)
        optimizer2=Adam(learning_rate=1e-3)
        optimizer3=Adam(learning_rate=1e-3)
        metrics += [mean_q]  # register default metrics

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model1 = clone_model(self.models[0], self.custom_model_objects)
        self.target_model1.compile(optimizer='sgd', loss='mse')
        self.models[0].compile(optimizer='sgd', loss='mse')

        self.target_model2 = clone_model(self.models[1], self.custom_model_objects)
        self.target_model2.compile(optimizer='sgd', loss='mse')
        self.models[1].compile(optimizer='sgd', loss='mse')

        self.target_model3 = clone_model(self.models[2], self.custom_model_objects)
        self.target_model3.compile(optimizer='sgd', loss='mse')
        self.models[2].compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates1 = get_soft_target_model_updates(self.target_model1, self.models[0], self.target_model_update)
            optimizer1 = AdditionalUpdatesOptimizer(optimizer1, updates1)
            updates2 = get_soft_target_model_updates(self.target_model2, self.models[1], self.target_model_update)
            optimizer2 = AdditionalUpdatesOptimizer(optimizer2, updates2)
            updates3 = get_soft_target_model_updates(self.target_model3, self.models[2], self.target_model_update)
            optimizer3 = AdditionalUpdatesOptimizer(optimizer3, updates3)

        def clipped_masked_error(args):
            y_true, y_pred, mask = args
            loss = huber_loss(y_true, y_pred, self.delta_clip)
            loss *= mask  # apply element-wise mask
            return K.sum(loss, axis=-1)

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_pred_1 = self.models[0].output
        y_pred_2 = self.models[1].output
        y_pred_3 = self.models[2].output

        y_true_1 = Input(name='y_true_1', shape=(self.nb_actions,))
        y_true_2 = Input(name='y_true_2', shape=(self.nb_actions,))
        y_true_3 = Input(name='y_true_3', shape=(self.nb_actions,))

        mask_1 = Input(name='mask_1', shape=(self.nb_actions,))
        mask_2 = Input(name='mask_2', shape=(self.nb_actions,))
        mask_3 = Input(name='mask_3', shape=(self.nb_actions,))

        loss_out_1 = Lambda(clipped_masked_error, output_shape=(1,), name='loss_1')([y_true_1, y_pred_1, mask_1])
        loss_out_2 = Lambda(clipped_masked_error, output_shape=(1,), name='loss_2')([y_true_2, y_pred_2, mask_2])
        loss_out_3 = Lambda(clipped_masked_error, output_shape=(1,), name='loss_3')([y_true_3, y_pred_3, mask_3])
        
        #modelの定義
        ins1 = [self.models[0].input] if type(self.models[0].input) is not list else self.models[0].input
        ins2 = [self.models[1].input] if type(self.models[1].input) is not list else self.models[1].input
        ins3 = [self.models[2].input] if type(self.models[2].input) is not list else self.models[2].input

        trainable_model1 = Model(inputs=ins1 + [y_true_1, mask_1 ], outputs = [loss_out_1, y_pred_1])
        trainable_model2 = Model(inputs=ins2 + [y_true_2, mask_2 ], outputs = [loss_out_2, y_pred_2])
        trainable_model3 = Model(inputs=ins3 + [y_true_3, mask_3 ], outputs = [loss_out_3, y_pred_3])
        
        combined_metrics1 = {
        trainable_model1.output_names[1]: metrics,
    }
        combined_metrics2 = {
        trainable_model2.output_names[1]: metrics,
    }
        combined_metrics3 = {
        trainable_model3.output_names[1]: metrics,
    }
        
        losses1 = [
            lambda y_true_1, y_pred_1: y_pred_1,  # 1つ目の出力の損失
            lambda y_true_1, y_pred_1: K.zeros_like(y_pred_1),  # メトリクス用
        ]
        losses2 = [
            lambda y_true_2, y_pred_2: y_pred_2,  # 1つ目の出力の損失
            lambda y_true_2, y_pred_2: K.zeros_like(y_pred_2),  # メトリクス用
        ]

        losses3 = [
            lambda y_true_3, y_pred_3: y_pred_3,  # 1つ目の出力の損失
            lambda y_true_3, y_pred_3: K.zeros_like(y_pred_3),  # メトリクス用
        ]
        trainable_model1.compile(optimizer=optimizer, loss=losses1, metrics=combined_metrics1)
        trainable_model2.compile(optimizer=optimizer, loss=losses2, metrics=combined_metrics2)
        trainable_model3.compile(optimizer=optimizer, loss=losses3, metrics=combined_metrics3)

        self.trainable_model1 = trainable_model1
        self.trainable_model2 = trainable_model2
        self.trainable_model3 = trainable_model3

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
    
    def get_hypervolume3(self, points):
        #Class HyperVolumeを用いて，pointsのhypervolumeを計算する
        self.reference_point = np.array([1000, 1000, 1000])
        hv = HyperVolume(self.reference_point)
        #pointにlenが適応できるようにする
        points = np.array(points)
        volume = hv.compute(points)
        return volume
    
    def get_hypervolume2(self, points):
        #Class HyperVolumeを用いて，pointsのhypervolumeを計算する
        self.reference_point = np.array([1000, 1000])
        hv = HyperVolume(self.reference_point)
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
        
    def compute_batch_q_values1(self, state_batch):
        batch = self.process_state_batch(state_batch)
        q_values = self.models[0].predict_on_batch(batch)
        # print('q_values: ' + str(q_values))
        return q_values
    
    def compute_batch_q_values2(self, state_batch):
        batch = self.process_state_batch(state_batch)
        q_values = self.models[1].predict_on_batch(batch)
        # print('q_values: ' + str(q_values))
        return q_values

    def compute_batch_q_values3(self, state_batch):
        batch = self.process_state_batch(state_batch)
        q_values = self.models[2].predict_on_batch(batch)
        # print('q_values: ' + str(q_values))
        return q_values
    
    def compute_q_values1(self, state):
        q_values_cost = self.compute_batch_q_values1([state])
        # assert q_values.shape == (self.nb_actions,)
        return q_values_cost
    
    def compute_q_values2(self, state):
        q_values_wt = self.compute_batch_q_values2([state])
        # assert q_values.shape == (self.nb_actions,)
        return q_values_wt
    
    def compute_q_values3(self, state):
        q_values_hv = self.compute_batch_q_values3([state])
        # assert q_values.shape == (self.nb_actions,)
        return q_values_hv
        

    def fit(self, env, nb_episodes, action_repetition=1, callbacks=None, verbose=1,
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

        self.episode = np.int16(0)
        self.step = np.int16(0)
        
        vector_for_hypervolume = [0,0]
        hv_hold = []
        vector_sum = []        
        vector_sum2 = []
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        result_vector = []
        try:
            while self.episode < nb_episodes:
 
                if observation is None:  # start of a new episode

                    # 環境にepisodeを教える
                    env.episode = self.episode

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
                    action_list = []  # リストをリセット
                    cnt_action2 = 0
                    cnt_action2_rate = 0
                    cost_sum = 0
                    epi_c_r = 0
                    metrics1 = []
                    metrics2 = []
                    metrics3 = []
                    var_reward_sum = 0
                    wt_step_sum = 0
                    wt_reward_sum = 0
                    fairness_count = 0
                    fifo_reward_sum = 0
                    count = 0

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states(self.models[0])
                    self.reset_states(self.models[1])
                    self.reset_states(self.models[2])

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

                action, q_values = self.forward(observation,self.object_mode)#行動を決定
                if self.processor is not None:
                    action = self.processor.process_action(action)
                # print('action: ' + str(action))
                action_list.append(action)
                cost = 0
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):

                    observation, r, cost, time, done, info, var_reward,std_reward, var_after,wt_step,fairness,user_wt,user_wt_sum,is_fifo,user_wt_log,time_reward_new= env.step(action)#行動を実行
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(observation, r, done, info)
                    #rewardのそれぞれの要素にrのそれぞれの要素を足し算する
                    reward = [reward[i] + r[i] for i in range(len(r))]

                    var_reward_sum += var_reward

                    # print('var_after: ' + str(var_after))

                    if done:
                        break




                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                # print('step: ' + str(self.step))
                # self.backward1(0, terminal=done\
                if  cost != 0:
                    cost_reward = -1
                else:
                    cost_reward = 1

                epi_c_r += cost_reward


                

                fairness_count += fairness

                usr_wt = max(wt[1] for wt in user_wt)
                # print(usr_wt)
                
                user_wt_sum_tmp = 0

                # print('user_wt_sum: ' + str(user_wt_sum))

                if wt_step == -1:
                    wt_step_sum += 0
                else:
                    wt_step_sum += wt_step

                

                
                for wt in user_wt_sum:
                    if wt[1] != -100:
                        user_wt_sum_tmp += wt[1]

                # print(user_wt_sum_tmp)
                        
                else:
                    user_wt_avg = user_wt_sum_tmp/100

                if usr_wt == -100:
                    # print("time")
                    wt_reward = 0
                elif usr_wt <= 5:
                    wt_reward = 1
                elif usr_wt > 5:
                    wt_reward = -1

                # print(user_wt)
                # print(user_wt_avg)

                # wt_reward = usr_wt

                wt_reward_sum +=time_reward_new

                if is_fifo == 1:
                    fifo_reward = 1
                elif is_fifo == 0:
                    fifo_reward = 0
                else:
                    fifo_reward = 0

                fifo_reward_sum += fifo_reward


                # print('wt_step: ' + str(wt_step))


                metrics1 = self.backward(0,self.memory1, cost_reward, terminal=done)
                metrics2 = self.backward(1,self.memory2, time_reward_new, terminal=done)
                metrics3 = self.backward(2,self.memory3, fairness, terminal=done)

                # wandb.log({'reward':reward[0],'cost':cost,'time':time,'var_reward':var_reward,'wt_step':wt_step,'fairness':fairness,'user_wt':usr_wt,'user_wt_sum':user_wt_sum_tmp,'fifo':is_fifo,'time_reward':time_reward_new})




                # wt_step_sum += usr_wt
                # print('wt_step_sum: ' + str(wt_step_sum))
                # print(wt_step)



                
                # self.backward3(0, terminal=done)
                # print('step: ' + str(self.step))
                # metrics2 = self.backward2(reward[1], terminal=done)

                #rewardを記録
                # if (reward[0] == -env.penalty_invalid_action or reward[1] == -env.penalty_invalid_action) == False and reward not in reward_record:
                #     reward_record.append(reward)
                # print(reward_record)

                #もしreward[x,x]の全要素が-5でなければepisode_reward+=reward
                # if np.all(reward == env.penalty_invalid_action) == False:
                episode_reward += reward

                #costが-1でなければコスト用の変数に加算

                episode_cost += cost

                episode_time += time


                # costが-1でなければepisode_step+=1
                    
                episode_step += 1
                self.step += 1
                # if time != -1:
                episode_time += time
                # episode_loss += metrics1[0]
                # episode_loss += metrics2[0]
                

                if done: # episode終了時
                    self.forward(observation, self.object_mode)
                    metrics1 = self.backward(0,self.memory1, 0, terminal=True)
                    metrics2 = self.backward(1,self.memory2, 0, terminal=True)
                    metrics3 = self.backward(2,self.memory3, 0, terminal=True)
                    
                    # metrics1 = self.backward1(0, terminal=True)
                    # metrics2 = self.backward2(2.2-episode_cost_without_minus/30, terminal=True)
                    # metrics3 = self.backward3(0, terminal=False)
                    print('episode' + str(self.episode) + '終了')

                    users = {i: [] for i in range(10)}
                    users_mean = []
                    users_last = []
                    users_median = []
                    std_mean = 0
                    std_last = 0
                    coeff_last = 0
                    coeff_median = 0

                    maxmin = 0


                    # データをイテレートして、適切なユーザーリストに追加

                    
                    for column in user_wt_log:
                        user_id = column[1]
                        if user_id in users:
                            users[user_id].append(column[2])
                    # print(users)
                    for column in users:
                        users_mean.append(np.mean(users[column]))
                        if len(users[column]) > 0:
                            users_last.append(np.max(users[column]))
                            users_median.append(statistics.median(users[column]))

                    #users_meanの一つめを削除
                    users_mean.pop(0)
                    std_mean = np.std(users_mean)
                    users_mean.pop(0)
                    users_median.pop(0)
                    std_last = np.std(users_last)
                    coeff_last = std_last/np.mean(users_last) 
                    coeff_median = np.std(users_median)/np.mean(users_median)

                    # print(users_mean)
                    maxmin = max(users_mean) - min(users_mean)


                    if self.episode == 0:
                        print('getmap1')
                        env.get_map(name=self.object_mode+'_map_0')
                        print('user_wt_log: ' + str(user_wt_log))
                        # exit()

                    if self.episode == 100:
                        print('getmap2')
                        env.get_map(name='self.object_mode'+'map_100')
                        print('user_wt_log: ' + str(user_wt_log))
                        # exit()

                    # if self.episode == 500:
                    #     env.get_map(name='map_500')
                    #     # exit(
                    if self.episode == 250:
                        print("vector_sum: " + str(vector_sum))
                        print('user_wt_sum: ' + str(user_wt_sum))
                        print('user_wt_log: ' + str(user_wt_log))
                        env.get_map(name=str(self.object_mode)+'map_250')
                    if self.episode == 500:
                        print('vector_sum: ' + str(vector_sum))
                        print('user_wt_sum: ' + str(user_wt_sum))
                        print('user_wt_log: ' + str(user_wt_log))
                        env.get_map(name=str(self.object_mode)+'map_500')

                    if self.episode == 1000:
                        print("vector_sum: " + str(vector_sum))
                        print('user_wt_sum: ' + str(user_wt_sum))
                        env.get_map(name='map2_2')
                        print('user_wt_log: ' + str(user_wt_log))
                        print('user_wt_sum: ' + str(user_wt_sum))
                        env.get_map(name=str(self.object_mode)+'map_1000')  

                    if self.episode == 1001:
                        print('user_wt_log: ' + str(user_wt_log))
                    if self.episode == 1002:
                        print('user_wt_log: ' + str(user_wt_log))
                    if self.episode == 1003:
                        print('user_wt_log: ' + str(user_wt_log))
                    if self.episode == 1004:
                        print('user_wt_log: ' + str(user_wt_log))
                    if self.episode == 1005:
                        print('user_wt_log: ' + str(user_wt_log))
                    if self.episode == 1006:
                        print('user_wt_log: ' + str(user_wt_log))
                    if self.episode == 1007:
                        print('user_wt_log: ' + str(user_wt_log))
                    if self.episode == 1008:
                        print('user_wt_log: ' + str(user_wt_log))
                    if self.episode == 1009:
                        print('user_wt_log: ' + str(user_wt_log))
                    if self.episode == 1010:
                        print('user_wt_log: ' + str(user_wt_log))

                    
                    if self.episode == 2000:
                        # env.get_map(name='10map_2000_fixdaaaaaaa2')
                        print("vector_sum: " + str(vector_sum))
                        print('user_wt_sum: ' + str(user_wt_sum))
                        print('user_wt_log: ' + str(user_wt_log))
                        print('user_wt_sum: ' + str(user_wt_sum))
                        env.get_map(name=str(self.object_mode)+'map_2000')
                        # exit()

                    if self.episode == 3000:
                        env.get_map(name=str(self.object_mode)+'map_3000')


                    

                    # print("Actions taken in this episode: ", action_list)
                    #action listのなかで，2をカウント
                    # cnt_action2 = action_list.count(2)
                    # len_action = len(action_list)
                    # cnt_action2_rate = cnt_action2/len_action

                    # print(cnt_action2_rate)
                    

                    

                    #waitingtimeを計算する
                    i=0
                    p=0
                    end_time = time
                    waiting_time_sum = 0
                    user_wt_var = 0
                    cnt_not_allocated = 0
                    #user_wtを0で初期化
                    # user_wt = [0 for p in range(1000)]
                    # print('env.jobs: ' + str(env.jobs))
                          

                    # while True:
                    #     #待ち時間を保持する変数
                    #     if i < len(env.jobs):  # そのエピソードで生成されたジョブの中でまだ見ていないジョブがある場合
                    #         # print('env.jobs[i]: ' + str(env.jobs[i]))
                    #         submitted_time = env.jobs[i][0]
                    #         if submitted_time <= end_time:  # ジョブが提出時刻を迎えていた場合
                    #             waiting_time = env.jobs[i][-1]
                    #             if waiting_time == -1:  # 割り当てられていなかった場合
                    #                 waiting_time = 100 # 待ち時間を大きめに設定
                    #                 cnt_not_allocated += 1
                    #             waiting_time_sum += waiting_time    # 待ち時間を記録
                    #             user_wt[env.jobs[i][4]] += waiting_time
                    #             i += 1
                    #             # print('waiting_time' + str(waiting_time))
                    #         else:  # ジョブが提出時刻を迎えていなかった場合
                    #             break
                    #     else:  # そのエピソードで生成されたジョブを全て見た場合
                    #         break

                    # #user_wtの要素が0であれば排除
                    # user_wt = [user_wt[p] for p in range(len(user_wt)) if user_wt[p] != 0]
                    # #user_wtの分散を求める
                
                    # user_wt_var = np.var(user_wt)
                    # #平均を求める
                    # user_wt_mean = np.mean(user_wt)
                    # #user_wtの変動係数
                    # user_wt_var_norm = user_wt_var/user_wt_mean


                    # print('user_wt: ' + str(user_wt))


                    

                    #2episode目からbackwardを行う。ただし1回目はbackwardの1変数目に0を入れる

                    vector_for_hypervolume = [user_wt_avg, episode_cost/10, var_after]
                    vector_sum.append(vector_for_hypervolume)
                    vector_sum2.append([user_wt_avg, episode_cost/10])
                    #hint = vector_for_hypervolumeの各要素を掛け合わせる
                    hint = vector_for_hypervolume[0] * vector_for_hypervolume[1] * var_after

                    hint2 = np.linalg.norm(vector_for_hypervolume)
                    hypervolume = self.get_hypervolume3(vector_sum)
                    hypervolume2 = self.get_hypervolume2(vector_sum2)
                    # hv_hold.append(hypervolume)
                    # dist_reward = self.calc_dist_from_pareto_front(vector_sum, vector_for_hypervolume)
                    # print("dist_reward" + str(dist_reward))
                    #hvの最後尾とそのひとつ前の引き算をする。ただし、hvの要素数が1の場合は0を返す
                    # if len(hv_hold) == 1:
                    #     hv_diff = 0
                    # else:
                    #     hv_diff = hv_hold[-1] - hv_hold[-2]
                    #     # print("hv_diff: " + str(hv_diff))
                    # #hvが正の値であれば報酬を与える。そうでなければ負の報酬を与える
                    # if hv_diff > 0:
                    #     reward_final += (1)
                        # self.backward1(reward, terminal=False)
                        # self.backward2(reward, terminal=False)
                        
                    # else:
                    #     if self.is_neigbor_parero_front(vector_sum, vector_for_hypervolume):
                    #         reward_final += (0.2)
                    #         reward = 0.2
                    #         # self.backward1(reward, terminal=False)
                    #         # self.backward2(reward, terminal=False)
                    #     else:
                    #         reward_final += (0)
                    #         reward = 0
                    #         # self.backward1(reward, terminal=False)
                    #         # self.backward2(reward, terminal=False) 




                    

                    
                # ウィンドウの履歴をCSVファイルに出力
                    
                    # print(on_premise_window_history.tolist())
                    # ウィンドウの履歴をリセット

                    env.reset_window_history()

                    if user_wt_avg ==0:
                        user_wt_avg = 1




                    

                    # wandbにログを保存
                    # wandb.log({"my_step":episode_step, "my_step_without_invalid,":episode_step_without_invalid,
                    #            "episode_loss":episode_loss/episode_step,'episode_reward': episode_reward/episode_step_without_invalid,"episode_cost":episode_cost/episode_step_without_invalid, "episode_time":episode_time/episode_step, 'nb_episode_steps': episode_step, 'nb_steps': self.step, 'end_time': time, 'jobs': env.jobs, 'waiting_time': waiting_time_sum/episode_step})
                    # table = wandb.Table(data =[ episode_cost_without_minus, waiting_time_sum],columns = ["x", "y"])
                    wandb.log({"episode_time":episode_time/episode_step, 'nb_episode_steps': episode_step,'end_time': time,"episode_cost":episode_cost_without_minus,"waiting_time_sum":waiting_time_sum,'hint':hint,'dist_reward':dist_reward,'user_wt_var':user_wt_var,'mean_cost':episode_cost,'episode_reward_c':epi_c_r,"var":var_after/user_wt_avg,'var2':var_after,'var3':var_after*var_after,"var_reward_sum":var_reward_sum,"wt_step_sum":wt_step_sum,"wt_reward_sum":wt_reward_sum,'fairness_count':fairness_count,'episodetime':episode_time,'time':time,'user_wt_avg':user_wt_avg,'loss_1':metrics1[0],'loss_2':metrics2[0],'loss_3':metrics3[0],'hint2':hint2,'hypervolume':hypervolume,'hypervolume2':hypervolume2,'fifo_reward':fifo_reward_sum,'std_mean':std_mean,'std_reward':std_reward,"maxmin":maxmin,'std_last':std_last,'coeff_last':coeff_last,'coeff_median':coeff_median})
                
                    # 'episode_c_w_m':wandb.plot.scatter(table, "x", "y",
                                #  title="Custom Y vs X Scatter Plot")
                    self.episode += 1
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
        env.get_map(name=str(self.object_mode)+'map')

        print("vector_sum: " + str(vector_sum))
        print('user_wt_sum: ' + str(user_wt_sum))
        print('user_wt_log: ' + str(user_wt_log))

        return history


    def reset_states(self, model):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            model.reset_states()
            if model == self.models[0]:
                self.target_model1.reset_states()
            elif model == self.models[1]:
                self.target_model2.reset_states()
            elif model == self.models[2]:
                self.target_model3.reset_states()

    def forward(self, observation, object_mode):
        #行動価値を推定する関数
        #Q値を算出し，policy内の関数に基づいてアクションを選択，それぞれを返す



        # Select an action.
        state1 = self.memory1.get_recent_state(observation)
        q_values1 = self.compute_q_values1(state1)

        state2 = self.memory2.get_recent_state(observation)
        q_values2 = self.compute_q_values2(state2)
        # print('q_values_cost: ' + str(q_values))

        state3 = self.memory3.get_recent_state(observation)
        q_values3 = self.compute_q_values3(state3)

        q_values = [q_values1,q_values2,q_values3]
        

        if self.training:
            action = self.policy.select_action(q_values=q_values, object_mode=self.object_mode)
        else:
            action = self.test_policy.select_action(q_values=q_values, object_mode=self.object_mode)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action, q_values
   
    def backward(self, model,memory,reward, terminal):
        #input: model（NN）, memory（NN関係のメモリ）, reward（報酬）, terminal（episode終了か否か）
        #報酬に基づいてNNを更新する関数
        #NNは3つ以内（0,1,2）のみ対応．4つ以上になったら追加実装必要



        if self.step % self.memory_interval == 0:
            if model == 0:
                self.memory1.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)
                metrics = [np.nan for _ in self.metrics_names1]
            elif model == 1:
                self.memory2.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)
                metrics = [np.nan for _ in self.metrics_names2]
            elif model == 2:
                self.memory3.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)
                metrics = [np.nan for _ in self.metrics_names3]
            
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            # print('not training')
            return metrics

        # Train the network on a single stochastic batch.
        if self.episode > self.nb_episodes_warmup and self.episode % self.train_interval == 0:
            experiences = memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            # print('reward_batch.shape: ' + str(reward_batch.shape))
            # print('self.batch_size: ' + str(self.batch_size))
            # assert reward_batch.shape == (self.batch_size,)
            # assert terminal1_batch.shape == reward_batch.shape
            # assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                q_values = self.models[model].predict_on_batch(state1_batch)
                assert q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                if model == 0:
                    target_q_values = self.target_model1.predict_on_batch(state1_batch)
                elif model == 1:
                    target_q_values = self.target_model2.predict_on_batch(state1_batch)
                elif model == 2:
                    target_q_values = self.target_model3.predict_on_batch(state1_batch)

                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                if model == 0:
                    target_q_values = self.target_model1.predict_on_batch(state1_batch)
                elif model == 1:
                    target_q_values = self.target_model2.predict_on_batch(state1_batch)
                elif model == 2:
                    target_q_values = self.target_model3.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)

            targets = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size,))
            masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch

            # print('reward_batch: ' + str(reward_batch))

   
            # assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch


            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ins = [state0_batch] if type(self.models[model].input) is not list else state0_batch
            # print('ins.shape: ' + str(np.array(ins).shape))
            # print('targets.shape: ' + str(np.array(targets).shape))
            # print('masks.shape: ' + str(np.array(masks).shape))
            # print('dummy_targets.shape: ' + str(np.array(dummy_targets).shape))
            # print('metrics.shape: ' + str(np.array(metrics1).shape))

            if model == 0:

                metrics = self.trainable_model1.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
            
            if model == 1:
                    
                metrics = self.trainable_model2.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
            
            if model == 2:
                metrics = self.trainable_model3.train_on_batch(ins + [targets, masks], [dummy_targets, targets])

            metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()
            

        return metrics

    @property
    def metrics_names1(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model1.output_names) == 2
        dummy_output_name = self.trainable_model1.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model1.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names
    @property
    def metrics_names2(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model2.output_names) == 2
        dummy_output_name = self.trainable_model2.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model2.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names2[:]
        return names
    @property
    def metrics_names3(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model3.output_names) == 2
        dummy_output_name = self.trainable_model3.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model3.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names3[:]
        return names