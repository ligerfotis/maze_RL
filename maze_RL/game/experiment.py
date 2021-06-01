from statistics import mean

import numpy as np
import pandas as pd
import math
import random
import csv
import time
from datetime import timedelta
from tqdm import tqdm

from maze_RL.maze3D.Maze3DEnv import Maze3D
from maze_RL.maze3D.assets import *
from maze_RL.maze3D.config import pause
from maze_RL.maze3D.utils import convert_actions
from maze_RL.maze3D.config import left_down, right_down, left_up, center

column_names = ["actions_x", "actions_y", "tray_rot_x", "tray_rot_y", "tray_rot_vel_x", "tray_rot_vel_y",
                "ball_pos_x", "ball_pos_y", "ball_vel_x", "ball_vel_y"]


class Experiment:
    def __init__(self, environment, agent=None, load_models=False, config=None, discrete=False):
        self.train_fps_list = []
        self.test_fps_list = []
        self.test_step_duration_list = []
        self.online_update_duration_list = []
        self.step_duration_list = []
        self.counter = 0
        self.test = 0
        self.config = config
        self.env = environment
        self.agent = agent
        # todo: fix it
        self.discrete_input = True  # discrete
        self.best_score = None
        self.best_reward = None
        self.best_score_episode = -1
        self.best_score_length = -1
        self.total_steps = 0  # total number of timesteps for all games
        self.action_history = []
        self.score_history = []
        self.game_duration_list = []
        self.length_list = []
        self.grad_updates_durations = []
        self.test_length_list = []
        self.test_score_history = []
        self.test_game_duration_list = []
        self.discrete = config['SAC']['discrete'] if 'SAC' in config.keys() else None
        self.second_human = config['game']['second_human'] if 'game' in config.keys() else None
        self.duration_pause_total = 0
        if load_models:
            self.agent.load_models()
        self.df = pd.DataFrame(columns=column_names)
        self.df_test = pd.DataFrame(columns=column_names)
        self.max_episodes = None
        self.max_timesteps = None
        self.avg_grad_updates_duration = 0
        self.human_actions = [0, 0]
        self.agent_action = [0, 0]
        self.total_timesteps = None
        self.max_timesteps_per_game = None
        self.save_models = True
        self.game = None
        self.test_max_timesteps = self.config['Experiment']['test_loop']['max_duration'] if 'test_loop' in config[
            'Experiment'].keys() else None
        self.test_max_episodes = self.config['Experiment']['test_loop']['max_episodes'] if 'test_loop' in config[
            'Experiment'].keys() else None  # 10
        self.update_cycles = None
        self.mode = config['Experiment']['loop']
        self.max_games = self.config['Experiment'][self.mode]['max_episodes']
        self.starting_training_criterion = self.config['Experiment'][self.mode]['start_training_step_on_episode']
        self.total_training_sessions = self.max_games / self.agent.update_interval
        self.max_game_duration = self.config['Experiment'][self.mode]['max_duration']
        self.action_duration = self.config['Experiment'][self.mode]['action_duration']

        self.distance_travel_list = []
        self.test_distance_travel_list = []
        self.reward_list = []
        self.test_reward_list = []
        self.last_time = 0
        self.key_pressed_count = 0
        self.last_pressed = None

    # New experiment function
    def max_episodes_mode(self, goal, maze):
        flag = True
        current_timestep = 0  # current timestep for each game
        running_reward = 0  # total cumulative reward across all games
        avg_length = 0

        # todo: wrap up in a function
        # The max number of timesteps depends on the maximum episode duration. Each loop (human action, agent action,
        # environment update) needs approximately 16ms.
        self.max_episodes = self.config['Experiment']['max_episodes_mode']['max_episodes']
        # todo: this is not consistent with every system. We could substitute the denominator with the 1/fps,
        #  because fps may change
        self.max_timesteps = int(self.config['Experiment'][self.mode]['max_duration'] * self.env.fps)
        self.best_score = -100 - 1 * self.max_timesteps * self.action_duration
        self.best_reward = self.best_score
        
        for i_episode in range(1, self.max_games + 1):
            start = time.time()
            # At the beginning of each game, reset the environment and
            # several variables
            observation = self.env.reset()  # stores the state of the environment
            reset = True  # used to reset the graphics environment when a new game starts
            timedout = False  # used to check if we hit the maximum game duration
            game_reward = 0  # keeps track of the rewards for each game
            dist_travel = 0  # keeps track of the ball's travelled distance
            test_offline_score = 0
            randomness_threshold = self.config['Experiment'][self.mode]['stop_random_agent']

            print("Episode: " + str(i_episode))

            actions = [0, 0, 0, 0]  # store the pressed keys. [0, 0, 0, 0] means that no key is pressed
            duration_pause = 0  # keeps track of the pause time
            self.save_models = True  # flag for saving RL models
            tmp_time = 0  # used to check if 200ms have passed and the agent needs to take a new action
            redundant_end_duration = 0  # duration in the game that is not playable by the user

            interaction = None

            for timestep in range(1, self.max_timesteps + 1):
                game_start_time = time.time()
                self.total_steps += 1
                current_timestep += 1

                # compute agent's action every 200 ms
                if not self.second_human:
                    # 200 ms have passed from previous action that was chosen.
                    if time.time() - tmp_time > self.action_duration:
                        if interaction is not None:
                            running_reward += interaction[2]
                            game_reward += interaction[2]
                            self.save_experience(interaction)

                        # get new action from the agent
                        tmp_time = time.time()  # reset timer for the next 200 ms
                        flag = self.compute_agent_action(observation=observation, randomness_criterion=i_episode,
                                                         randomness_threshold=randomness_threshold, flag=flag)

                        # online training every 200 ms on a batch from the replay buffer
                        # check if we are testing the model or not using an RL agent
                        # todo: wrap up in a function
                        start_online_update = time.time()
                        if not self.config['game']['test_model'] and not self.second_human:
                            # check if we should do online updates and the replay buffer has been filled with enough
                            # transitions
                            if self.config['Experiment']['online_updates'] and i_episode >= self.starting_training_criterion:
                                if self.discrete:
                                    self.agent.learn()
                                    self.agent.soft_update_target()
                        self.online_update_duration_list.append(time.time() - start_online_update)

                # get human action in the form [y axis rotation, x axis rotation]
                # the human's action is saved in the self.human_actions variable
                duration_pause, _ = self.getKeyboard(duration_pause, actions)

                # get human-agent action pair in the form [agent action, human action(x-axis rotation)]
                action = self.get_action_pair()

                if timestep == self.max_timesteps:
                    observation_, reward, done, tmp_redundant_end_duration, train_fps = self.env.step(action=action, timedout=True,
                                                                                           goal=goal, reset=reset)
                else:
                    observation_, reward, done, tmp_redundant_end_duration, train_fps = self.env.step(action=action, timedout=False,
                                                                                           goal=goal, reset=reset)
                self.train_fps_list.append(train_fps)
                redundant_end_duration += tmp_redundant_end_duration
                if reset:
                    reset = False

                # add experience to buffer
                interaction = [observation, self.agent_action, reward, observation_, done]

                test_offline_score += -1 if not done else 0

                # compute travelled distance
                dist_travel += math.sqrt((observation[0] - observation_[0]) * (observation[0] - observation_[0]) +
                                         (observation[1] - observation_[1]) * (observation[1] - observation_[1]))

                observation = observation_
                new_row = {'actions_x': action[0], 'actions_y': action[1], "ball_pos_x": observation[0],
                           "ball_pos_y": observation[1], "ball_vel_x": observation[2], "ball_vel_y": observation[3],
                           "tray_rot_x": observation[4], "tray_rot_y": observation[5], "tray_rot_vel_x": observation[6],
                           "tray_rot_vel_y": observation[7]}

                # append row to the dataframe
                self.df = self.df.append(new_row, ignore_index=True)
                
                step_duration = time.time() - game_start_time
                self.step_duration_list.append(step_duration)
                # if the game ended, proceed with the next game
                if done:
                    # save the last interaction of a successful game, because the loop will break
                    self.save_experience(interaction)
                    break

            # todo: wrap up in a function
            # keep track of best game reward
            if self.best_reward < game_reward:
                self.best_reward = game_reward

            # keep track of total pause duration
            end = time.time()
            self.duration_pause_total += duration_pause
            print(redundant_end_duration)
            game_duration = end - start - duration_pause - redundant_end_duration
            print("Game duration: " + str(game_duration))

            # todo: wrap up in a function
            # keep track of the game reward history 
            self.score_history.append(game_reward)
            self.reward_list.append(game_reward)

            # keep track of the game duration
            self.game_duration_list.append(game_duration)

            # keep track of the ball's travelled distance
            self.distance_travel_list.append(dist_travel)

            # log_interval: 10  
            # print avg reward in the interval
            log_interval = self.config['Experiment'][self.mode]['log_interval']
            avg_ep_duration = np.mean(self.game_duration_list[-log_interval:])
            avg_score = np.mean(self.score_history[-log_interval:])

            self.length_list.append(current_timestep)
            avg_length += current_timestep

            # todo: wrap up in a function
            # off policy learning
            print("Replay buffer size: {}".format(len(self.agent.memory.storage)))
            if not self.config['game']['test_model'] and i_episode >= self.config['Experiment'][self.mode][
                'start_training_step_on_episode']:  # 10
                print("update interval: {}".format(self.agent.update_interval))
                print(i_episode)
                if i_episode % self.agent.update_interval == 0:
                    print("off policy learning.")
                    self.updates_scheduler()
                    print(self.update_cycles)
                    print(self.max_episodes)
                    print(self.max_timesteps)
                    if self.update_cycles > 0:
                        grad_updates_duration = self.grad_updates(self.update_cycles)
                        self.grad_updates_durations.append(grad_updates_duration)

                        # save the models after each grad update
                        self.agent.save_models()

                    # Test trials
                    if i_episode % self.config['Experiment']['test_interval'] == 0 and self.test_max_episodes > 0:
                        self.test_agent(goal, maze)
                        print("Continue Training.")

            # todo: wrap up in a function
            # logging
            if self.config["game"]["verbose"]:  # true
                if not self.config['game']['test_model']:
                    running_reward, avg_length = self.print_logs(i_episode, running_reward, avg_length, log_interval,
                                                                 avg_ep_duration)
                current_timestep = 0
        update_cycles = math.ceil(
            self.config['Experiment'][self.mode]['total_update_cycles'])
        if not self.second_human and update_cycles > 0:
            try:
                self.avg_grad_updates_duration = np.mean(self.grad_updates_durations)
            except:
                print("Exception when calc grad_updates_durations")

    # Οld loop wrt maximum interactions (currently has bugs)
    def max_interactions_mode(self, goal):
        # Experiment 2 loop
        flag = True
        current_timestep = 0
        observation = self.env.reset()
        timedout = False
        game_reward = 0
        actions = [0, 0, 0, 0]  # all keys not pressed

        # todo: wrap up in a void function
        self.best_score = -50 - 1 * self.config['Experiment']['max_interactions_mode']['max_timesteps_per_game']
        self.best_reward = self.best_score
        self.total_timesteps = self.config['Experiment']['max_interactions_mode']['total_timesteps']
        self.max_timesteps_per_game = self.config['Experiment']['max_interactions_mode']['max_timesteps_per_game']
        self.save_models = True
        self.game = 0

        avg_length = 0
        duration_pause = 0
        running_reward = 0
        start = time.time()

        for timestep in range(1, self.total_timesteps + 1):
            self.total_steps += 1
            current_timestep += 1

            # get agent's action
            if not self.second_human:
                randomness_threshold = self.config['Experiment']['max_interactions_mode'][
                    'start_training_step_on_timestep']
                randomness_critirion = timestep
                flag = self.compute_agent_action(observation, randomness_critirion, randomness_threshold, flag)

            # compute keyboard action
            duration_pause, _ = self.getKeyboardOld(actions, duration_pause)

            # get final action pair
            action = self.get_action_pair()

            # check if we reached the maximum number of timesteps
            if current_timestep == self.max_timesteps_per_game:
                timedout = True

            # Environment step
            observation_, reward, done = self.env.step(action, timedout, goal,
                                                       self.config['Experiment']['max_interactions_mode'][
                                                           'action_duration'])

            # keep track of old state, new state, action pair, reward
            interaction = [observation, self.agent_action, reward, observation_, done]

            # add experience to buffer
            self.save_experience(interaction)

            # online train
            if not self.config['game']['test_model'] and not self.second_human:
                if self.config['Experiment']['online_updates']:
                    if self.discrete:
                        self.agent.learn()
                        self.agent.soft_update_target()

            # todo: wrap up in a function | duplicate
            new_row = {'actions_x': action[0], 'actions_y': action[1], "ball_pos_x": observation[0],
                       "ball_pos_y": observation[1], "ball_vel_x": observation[2], "ball_vel_y": observation[3],
                       "tray_rot_x": observation[4], "tray_rot_y": observation[5], "tray_rot_vel_x": observation[6],
                       "tray_rot_vel_y": observation[7]}

            # append row to the dataframe
            self.df = self.df.append(new_row, ignore_index=True)
            observation = observation_

            # off policy learning
            if not self.config['game']['test_model'] and self.total_steps >= \
                    self.config['Experiment']['max_interactions_mode'][
                        'start_training_step_on_timestep']:
                update_cycles = math.ceil(
                    self.config['Experiment']['max_interactions_mode']['update_cycles'])
                if self.total_steps % self.agent.update_interval == 0 and update_cycles > 0:
                    grad_updates_duration = self.grad_updates(update_cycles)
                    self.grad_updates_durations.append(grad_updates_duration)

                    # save the models after each grad update
                    self.agent.save_models()

                    # Test trials
                    if self.test_max_episodes > 0:
                        self.test_agent(goal)
                        print("Continue Training.")

            running_reward += reward
            game_reward += reward

            if done:
                end = time.time()
                self.game += 1
                if self.best_reward < game_reward:
                    self.best_reward = game_reward
                self.duration_pause_total += duration_pause
                game_duration = end - start - duration_pause

                self.game_duration_list.append(game_duration)
                self.score_history.append(game_reward)

                log_interval = self.config['Experiment']['max_interactions_mode']['log_interval']
                avg_ep_duration = np.mean(self.game_duration_list[-log_interval:])
                avg_score = np.mean(self.score_history[-log_interval:])

                self.length_list.append(current_timestep)
                avg_length += current_timestep

                # logging
                if self.config["game"]["save"]:
                    if not self.config['game']['test_model']:
                        running_reward, avg_length = self.print_logs(self.game, running_reward, avg_length,
                                                                     log_interval,
                                                                     avg_ep_duration)

                current_timestep = 0
                observation = self.env.reset()
                timedout = False
                game_reward = 0
                actions = [0, 0, 0, 0]  # all keys not pressed
                start = time.time()

        if not self.second_human:
            self.avg_grad_updates_duration = np.mean(self.grad_updates_durations)

    # used for human-only games
    def test_human(self, goal):
        self.max_episodes = self.config['Experiment']['max_episodes_mode']['max_episodes']
        self.max_timesteps = int(self.config['Experiment']['max_episodes_mode']['max_duration'] / 0.016)
        duration_pause = 0
        observation = self.env.reset()
        timedout = False
        tmp_time = 0
        for i_episode in range(1, self.max_episodes + 1):
            self.env.reset()
            actions = [0, 0, 0, 0]  # all keys not pressed
            reset = True
            timedout = False
            start_time = time.time()
            for timestep in range(1, self.max_timesteps + 2):
                # New version
                duration_pause, actions = self.getKeyboard(duration_pause, actions)
                action = convert_actions(actions)
                _, _, done, redundant_end_duration = self.env.step(action, timedout, goal, reset)

                if reset:
                    reset = False
                if timestep == self.max_timesteps:
                    timedout = True

                if done:
                    break

    def getKeyboard(self, duration_pause, actions):
        if not self.discrete_input:
            pg.key.set_repeat(10)  # argument states the difference (in ms) between consecutive press events
        else:
            actions = [0, 0, 0, 0]
        space_pressed = True
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return 1
            if event.type == pg.KEYDOWN:
                self.last_time = time.time()
                if event.key == pg.K_SPACE and space_pressed:
                    space_pressed = False
                    start_pause = time.time()
                    actions = [0, 0, 0, 0]
                    pause()
                    end_pause = time.time()
                    duration_pause += end_pause - start_pause
                if event.key == pg.K_q:
                    exit(1)
                if event.key in self.env.keys:
                    self.key_pressed_count = 0
                    self.last_pressed = self.env.keys_fotis[event.key]
                    actions = [0, 0, 0, 0]
                    actions[self.env.keys_fotis[event.key]] = 1

            if event.type == pg.KEYUP:
                if event.key in self.env.keys:
                    actions[self.env.keys_fotis[event.key]] = 0

        self.human_actions = convert_actions(actions)
        return duration_pause, actions

    # ref @ sac_maze3d_test.py sac_maze3d_train.py
    def save_info(self, chkpt_dir, experiment_duration, total_games, goal):
        info = {}
        info['goal'] = goal
        info['experiment_duration'] = experiment_duration
        info['best_score'] = self.best_score
        info['best_score_episode'] = self.best_score_episode
        info['best_reward'] = self.best_reward
        info['best_score_length'] = self.best_score_length
        info['total_steps'] = self.total_steps
        info['total_games'] = total_games
        info['fps'] = self.env.fps
        info['avg_grad_updates_duration'] = self.avg_grad_updates_duration
        w = csv.writer(open(chkpt_dir + '/rest_info.csv', "w"))
        for key, val in info.items():
            w.writerow([key, val])

    # ref @ experiment.py and maze3D/utils.py
    def get_action_pair(self):
        if self.second_human:
            action = self.human_actions
        else:
            if self.config['game']['agent_only']:
                action = self.get_agent_only_action()
            else:
                action = [self.agent_action, self.human_actions[1]]
        self.action_history.append(action)
        return action

    # ref @ experiment.py
    def save_experience(self, interaction):
        observation, agent_action, reward, observation_, done = interaction
        if not self.second_human:
            if self.discrete:
                self.agent.memory.add(observation, agent_action, reward, observation_, done)
            else:
                self.agent.remember(observation, agent_action, reward, observation_, done)

    # not used anywhere
    def save_best_model(self, avg_score, game, current_timestep):
        if avg_score > self.best_score:
            self.best_score = avg_score
            self.best_score_episode = game
            self.best_score_length = current_timestep
            if not self.config['game']['test_model'] and self.save_models and not self.second_human:
                self.agent.save_models()

    # ref @ experiment.py 
    def grad_updates(self, update_cycles=None):
        start_grad_updates = time.time()
        end_grad_updates = 0
        if not self.second_human:
            print("Performing {} updates".format(update_cycles))
            for _ in tqdm(range(update_cycles)):
                if self.discrete:
                    self.agent.learn()
                    self.agent.soft_update_target()
                else:
                    self.agent.learn()
            end_grad_updates = time.time()

        return end_grad_updates - start_grad_updates

    # ref @ experiment.py
    def print_logs(self, game, running_reward, avg_length, log_interval, avg_ep_duration):
        if game % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            log_reward = int((running_reward / log_interval))

            print(
                'Episode {}\tTotal timesteps {}\tavg length: {}\tTotal reward(last {} episodes): {}\tBest Score: {}\tavg '
                'episode duration: {}'.format(game, self.total_steps, avg_length,
                                              log_interval,
                                              log_reward, self.best_reward,
                                              timedelta(
                                                  seconds=avg_ep_duration)))
            running_reward = 0
            avg_length = 0
        return running_reward, avg_length

    def test_print_logs(self, avg_score, avg_length, best_score, duration):
        print(
            'Avg Score: {}\tAvg length: {}\tBest Score: {}\tTest duration: {}'.format(avg_score,
                                                                                      avg_length, best_score,
                                                                                      timedelta(seconds=duration)))

    # ref @ experiment.py
    def compute_agent_action(self, observation, randomness_criterion=None, randomness_threshold=None, flag=True):
        if self.discrete:
            if randomness_criterion is not None and randomness_threshold is not None \
                    and randomness_criterion <= randomness_threshold:
                # Pure exploration
                if self.config['game']['agent_only']:
                    self.agent_action = np.random.randint(pow(2, self.env.action_space.actions_number))
                else:
                    self.agent_action = np.random.randint(self.env.action_space.actions_number)
                self.save_models = False
                if flag:
                    print("Using Random Agent")
                    flag = False
            else:  # Explore with actions_prob
                self.save_models = True
                self.agent_action = self.agent.actor.sample_act(observation)
                if not flag:
                    print("Using SAC Agent")
                    flag = True
        else:
            self.save_models = True
            self.agent_action = self.agent.choose_action(observation)
        return flag

    # ref @ experiment.py
    def test_agent(self, goal, maze, randomness_critirion=None, ):  # this is used in loop
        # test loop
        print("Testing the agent.")
        current_timestep = 0
        self.test += 1
        print('(test_agent) Test {}'.format(self.test))
        best_score = 0
        for game in range(1, self.test_max_episodes + 1):  # 10 test episodes
            observation = self.env.reset()
            reset = True
            timedout = False
            game_reward = 0
            start = time.time()
            dist_travel = 0

            actions = [0, 0, 0, 0]  # all keys not pressed
            duration_pause = 0
            tmp_time = 0
            for timestep in range(1, self.test_max_timesteps + 1):
                test_game_start_time = time.time()
                current_timestep += 1

                if time.time() - tmp_time > self.config['Experiment']['max_episodes_mode']['action_duration']:
                    tmp_time = time.time()
                    randomness_threshold = self.config['Experiment']['max_interactions_mode'][
                        'start_training_step_on_timestep']
                    self.compute_agent_action(observation, randomness_critirion, randomness_threshold)

                # get human action
                duration_pause, _ = self.getKeyboard(duration_pause, actions)

                # get human-agent action pair
                action = self.get_action_pair()

                if timestep == self.test_max_timesteps:
                    timedout = True

                observation_, reward, done, redundant_end_duration, test_fps = self.env.step(action, timedout, goal, reset)
                self.test_fps_list.append(test_fps)

                if reset:
                    reset = False
                # compute agent's action
                # randomness_threshold = self.config['Experiment']['max_interactions_mode']['start_training_step_on_timestep']
                # self.compute_agent_action(observation, randomness_criterion, randomness_threshold)

                # # compute keyboard action and do env step
                # duration_pause, _, _, _, _ = self.getKeyboardOld(actions, duration_pause, observation, timedout, self.discrete)

                # maze.board.update()
                # glClearDepth(1000.0)
                # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                # maze.board.draw()
                # pg.display.flip()
                # maze.dt = pg.time.Clock().tick(maze.fps)
                # fps = pg.time.Clock().get_fps()
                # pg.display.set_caption("Running at " + str(int(fps)) + " fps")

                # action = self.get_action_pair();

                # if timestep == self.test_max_timesteps:
                #     timedout = True

                # actionAgent = []  # for the agent action
                # actionAgent.append(self.agent_action)  # sto compute_agent_action pio panw to exei kanei update auto
                # actionAgent.append(0)  # no action for the human

                # # Environment step for agent action
                # #observation_, reward, done = self.env.step(actionAgent, timedout, goal,
                # #                                           self.config['Experiment']['max_episodes_mode']['action_duration'])

                # observation_, reward, done = self.env.step_with_timestep(actionAgent, timedout, goal, timestep,
                #                                            self.config['Experiment']['max_episodes_mode']['action_duration'])

                # compute distance_travel
                dist_travel += math.sqrt((observation[0] - observation_[0]) * (observation[0] - observation_[0]) +
                                         (observation[1] - observation_[1]) * (observation[1] - observation_[1]))

                observation = observation_
                new_row = {'actions_x': action[0], 'actions_y': action[1], "ball_pos_x": observation[0],
                           "ball_pos_y": observation[1], "ball_vel_x": observation[2], "ball_vel_y": observation[3],
                           "tray_rot_x": observation[4], "tray_rot_y": observation[5], "tray_rot_vel_x": observation[6],
                           "tray_rot_vel_y": observation[7]}
                # append row to the dataframe
                self.df_test = self.df_test.append(new_row, ignore_index=True)

                game_reward += reward
                test_step_duration = time.time() - test_game_start_time
                self.test_step_duration_list.append(test_step_duration)
                if done:
                    break

            end = time.time()

            self.duration_pause_total += duration_pause
            game_duration = end - start - duration_pause
            episode_score = self.config['Experiment']['test_loop']['max_score'] + game_reward  # max_score=200

            # pare apo edw ta stoixeia
            self.test_length_list.append(current_timestep)
            best_score = episode_score if episode_score > best_score else best_score
            self.test_score_history.append(episode_score)

            self.test_game_duration_list.append(game_duration)
            self.test_reward_list.append(game_reward)
            self.test_distance_travel_list.append(dist_travel)

            current_timestep = 0

        # logging
        self.test_print_logs(np.mean(self.test_score_history[-10:]), np.mean(self.test_length_list[-10:]), best_score,
                             sum(self.test_game_duration_list[-10:]))

    # ref @ experiment.py
    def get_agent_only_action(self):
        # up: 0, down:1, left:2, right:3, upleft:4, upright:5, downleft: 6, downright:7
        if self.agent_action == 0:
            return [1, 0]
        elif self.agent_action == 1:
            return [-1, 0]
        elif self.agent_action == 2:
            return [0, -1]
        elif self.agent_action == 3:
            return [0, 1]
        elif self.agent_action == 4:
            return [1, -1]
        elif self.agent_action == 5:
            return [1, 1]
        elif self.agent_action == 6:
            return [-1, -1]
        elif self.agent_action == 7:
            return [-1, 1]
        else:
            print("Invalid agent action")

    # not used anywhere
    def test_loop(self):  # mhn asxoleisai
        # test loop
        current_timestep = 0
        self.test += 1
        print('(test loop) Test {}'.format(self.test))
        goals = [left_down, right_down, left_up, ]
        for game in range(1, self.test_max_episodes + 1):
            # randomly choose a goal
            current_goal = random.choice(goals)

            dist_travel = 0

            observation = self.env.reset()
            timedout = False
            game_reward = 0
            start = time.time()

            actions = [0, 0, 0, 0]  # all keys not pressed
            duration_pause = 0
            self.save_models = False
            for timestep in range(1, self.test_max_timesteps + 1):
                self.total_steps += 1
                current_timestep += 1

                # compute agent's action
                self.compute_agent_action(observation)
                # compute keyboard action
                duration_pause, _ = self.getKeyboardOld(actions, duration_pause)
                # get final action pair
                action = self.get_action_pair()

                if timestep == self.max_timesteps:
                    timedout = True

                # Environment step
                # observation_, reward, done = self.env.step(action, timedout, current_goal,
                #                                           self.config['Experiment']['test_loop']['action_duration'])

                observation_, reward, done = self.env.step_with_timestep((action, timedout, current_goal, timestep,
                                                                          self.config['Experiment']['test_loop'][
                                                                              'action_duration']))

                # compute distance_travel
                dist_travel += math.sqrt((observation[0] - observation_[0]) * (observation[0] - observation_[0]) +
                                         (observation[1] - observation_[1]) * (observation[1] - observation_[1]))

                observation = observation_
                new_row = {'actions_x': action[0], 'actions_y': action[1], "ball_pos_x": observation[0],
                           "ball_pos_y": observation[1], "ball_vel_x": observation[2], "ball_vel_y": observation[3],
                           "tray_rot_x": observation[4], "tray_rot_y": observation[5], "tray_rot_vel_x": observation[6],
                           "tray_rot_vel_y": observation[7]}
                # append row to the dataframe
                self.df_test = self.df_test.append(new_row, ignore_index=True)

                game_reward += reward

                if done:
                    break

            end = time.time()

            self.duration_pause_total += duration_pause
            game_duration = end - start - duration_pause

            self.test_score_history.append(self.config['Experiment']['test_loop']['max_score'] + game_reward)
            self.test_length_list.append(current_timestep)
            self.test_game_duration_list.append(game_duration)
            self.test_reward_list.append(game_reward)
            self.test_distance_travel_list.append(dist_travel)

            current_timestep = 0

    # ref @ experiment.py
    def updates_scheduler(self):
        update_list = [22000, 1000, 1000, 1000, 1000, 1000, 1000]
        total_update_cycles = self.config['Experiment']['max_episodes_mode']['total_update_cycles']
        online_updates = 0
        if self.config['Experiment']['online_updates']:
            # maximum cumulative online gradient updates for the whole experiment
            online_updates = self.max_games * self.max_game_duration / self.action_duration
            print("max game:{} max game duration: {} total online updates:{}".format(self.max_games, self.max_game_duration, online_updates))

        if self.update_cycles is None:
            self.update_cycles = total_update_cycles - online_updates

        if self.config['Experiment']['scheduling'] == "descending":
            self.counter += 1
            if not (math.ceil(self.max_episodes / self.agent.update_interval) == self.counter):
                self.update_cycles /= 2

        elif self.config['Experiment']['scheduling'] == "big_first":
            if self.config['Experiment']['online_updates']:
                if self.counter == 1:
                    self.update_cycles = update_list[self.counter]
                else:
                    self.update_cycles = 0
            else:
                self.update_cycles = update_list[self.counter]

            self.counter += 1

        else:
            self.update_cycles = (total_update_cycles - online_updates) / math.ceil(
                self.max_episodes / self.agent.update_interval)

        self.update_cycles = math.ceil(self.update_cycles)
