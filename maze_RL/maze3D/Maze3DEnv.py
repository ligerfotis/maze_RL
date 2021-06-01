import random
import time
# import maze3D.rewards
from maze_RL.game import rewards
from maze_RL.maze3D.gameObjects import *
from maze_RL.maze3D.assets import *
from maze_RL.maze3D.utils import get_distance_from_goal, checkTerminal
from maze_RL.rl_models.utils import get_config
from maze_RL.maze3D.config import layout_up_right, layout_down_right, layout_up_left

layouts = [layout_down_right, layout_up_left, layout_up_right]


class ActionSpace:
    def __init__(self):
        # self.actions = list(range(0, 14 + 1))
        # self.shape = 1
        self.actions = list(range(0, 3))
        self.shape = 2
        self.actions_number = len(self.actions)
        self.high = self.actions[-1]
        self.low = self.actions[0]

    def sample(self):
        # return [random.sample([0, 1, 2], 1), random.sample([0, 1, 2], 1)]
        return np.random.randint(self.low, self.high + 1, 2)


class Maze3D:
    def __init__(self, config=None, config_file=None):
        # choose randomly one starting point for the ball
        self.fps_list = []
        self.config = get_config(config_file) if config_file is not None else config
        current_layout = random.choice(layouts)
        self.discrete_input = self.config['game']['discrete']
        self.rl = True if 'SAC' in self.config.keys() else False
        self.board = GameBoard(current_layout, self.discrete_input, self.rl)
        self.keys = {pg.K_UP: 1, pg.K_DOWN: 2, pg.K_LEFT: 4, pg.K_RIGHT: 8}
        self.keys_fotis = {pg.K_UP: 0, pg.K_DOWN: 1, pg.K_LEFT: 2, pg.K_RIGHT: 3}
        self.running = True
        self.done = False
        self.observation = self.get_state()  # must init board fisrt
        self.action_space = ActionSpace()
        self.observation_shape = (len(self.observation),)
        self.dt = None
        self.fps = 60
        rewards.main(self.config)

    def step(self, action, timedout, goal, reset):
        timeStart = 0
        fps = 0
        if not reset:
            self.board.handleKeys(action)
            self.board.update()
            glClearDepth(1000.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.board.draw()
            pg.display.flip()

            self.dt = clock.tick(self.fps)
            fps = clock.get_fps()
            self.fps_list.append(fps)
            pg.display.set_caption("Running at " + str(int(fps)) + " fps")
            self.observation = self.get_state()
        if reset:
            timeStart = time.time()
            i = 0
            self.board.update()
            while time.time() - timeStart <= 5:
                glClearDepth(1000.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                self.board.draw(mode=1, idx=i)
                pg.display.flip()
                time.sleep(1)
                i += 1
        goal_reached = checkTerminal(self.board.ball, goal)
        if goal_reached or timedout:
            timeStart = time.time()
            i = 0
            self.board.update()
            while time.time() - timeStart <= 3:
                glClearDepth(1000.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                if goal_reached:
                    self.board.draw(mode=2, idx=i)
                else:
                    self.board.draw(mode=3, idx=i)
                pg.display.flip()
                time.sleep(1)
                i += 1
            self.done = True
        reward = rewards.reward_function_maze(self.done, timedout, ball=self.board.ball, goal=goal)
        if timeStart != 0:
            return self.observation, reward, self.done, time.time() - timeStart, fps
        else:
            return self.observation, reward, self.done, 0, fps

    def get_state(self):
        # [ball pos x | ball pos y | ball vel x | ball vel y|  theta(x) | phi(y) |  theta_dot(x) | phi_dot(y) | ]
        return np.asarray(
            [self.board.ball.x, self.board.ball.y, self.board.ball.velocity[0], self.board.ball.velocity[1],
             self.board.rot_x, self.board.rot_y, self.board.velocity[0], self.board.velocity[1]])

    def reset(self):
        self.__init__(config=self.config)
        return self.observation
