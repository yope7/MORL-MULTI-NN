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


# testの時のみaction_filterを使うDQNAgent
class DQNAgent_without_action_filter(DQNAgent):
    def __init__(self, model, n_window, n_on_premise_node, n_cloud_node, n_job_queue_obs, n_job_queue_bck, policy=None,
                 test_policy=None, enable_double_dqn=False, enable_dueling_network=False,
                 dueling_type='avg', target_model_update=1000, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        # Validate (important) input.
        if list(model.output.shape) != list((None, self.nb_actions)):
            raise ValueError(
                'Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(
                    model.output, self.nb_actions))

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
                            observation, reward, done, info = self.processor.process_step(observation, reward, done,
                                                                                          info)
                        callbacks.on_action_end(action)
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

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                action, q_values = self.forward(observation)
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
                        observation, r, done, info = self.processor.process_step(observation, r, done, info)
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
                metrics = self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics,
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

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.forward(observation)
                    self.backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                        # 自作
                        'end_time': time,
                        'jobs': env.jobs,
                    }
                    #                     print('---------------------')
                    #                     print(episode_logs['jobs'])
                    #                     print('---------------------')
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

    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
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
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                q_values = self.model.predict_on_batch(state1_batch)
                assert q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                target_q_values = self.target_model.predict_on_batch(state1_batch)
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
            assert discounted_reward_batch.shape == reward_batch.shape
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
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
            print('ins.shape: ' + str(np.array(ins).shape))
            print('targets.shape: ' + str(np.array(targets).shape))
            print('masks.shape: ' + str(np.array(masks).shape))
            print('dummy_targets.shape: ' + str(np.array(dummy_targets).shape))
            print('metrics.shape: ' + str(np.array(metrics).shape))
            print('metrics: ' + str(metrics))

            exit()
            metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model.output_names) == 2
        dummy_output_name = self.trainable_model.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

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
        q_values = self.compute_q_values(state)

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