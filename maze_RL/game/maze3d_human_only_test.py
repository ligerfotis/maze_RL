import sys
from experiment import Experiment
from maze3D.Maze3DEnv import Maze3D
from rl_models.utils import get_config


def main(argv):
    # get configuration
    # sets goal, whether the input is discrete or continuous
    # and the number of episodes and timesteps
    test_config = get_config(argv[0])

    # creating environment
    maze = Maze3D(config_file=argv[0])

    # create the experiment
    experiment = Experiment(maze, config=test_config, discrete=test_config['game']['discrete'])

    # set the goal
    goal = test_config["game"]["goal"]

    # Test loop
    experiment.test_human(goal)

    pg.quit()


if __name__ == '__main__':
    main(sys.argv[1:])
    exit(0)
