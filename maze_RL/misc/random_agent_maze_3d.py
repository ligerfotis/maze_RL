from maze3D.Maze3DEnv import Maze3D
from maze3D.assets import *


def main():
    maze = Maze3D()
    action = 0
    observation = maze.reset()
    while maze.running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                maze.running = False
        action = maze.action_space.sample()
        observation_, done = maze.step(action)
        if done:
            break

    pg.quit()


if __name__ == '__main__':
    main()
    exit(0)
