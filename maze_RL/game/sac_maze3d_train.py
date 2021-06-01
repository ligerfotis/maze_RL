# Virtual environment
from maze_RL.maze3D.Maze3DEnv import Maze3D
from maze_RL.maze3D.assets import *
from maze_RL.maze3D.utils import save_logs_and_plot

# Experiment
from experiment import Experiment

# RL modules
from maze_RL.rl_models.sac_agent import Agent
from maze_RL.rl_models.sac_discrete_agent import DiscreteSACAgent
from maze_RL.rl_models.utils import get_config, get_plot_and_chkpt_dir, get_sac_agent

import sys
import time
from datetime import timedelta

"""
The code of this work is based on the following github repos:
https://github.com/kengz/SLM-Lab
https://github.com/EveLIn3/Discrete_SAC_LunarLander/blob/master/sac_discrete.py
"""

def main(argv):
    # get configuration
    config = get_config(argv[0])

    # creating environment
    maze = Maze3D(config_file=argv[0])

    chkpt_dir, load_checkpoint_name = [None, None]
    if config["game"]["save"]:
        # create the checkpoint and plot directories for this experiment
        chkpt_dir, plot_dir, load_checkpoint_name = get_plot_and_chkpt_dir(config, argv[1])

    # create the SAC agent
    sac = get_sac_agent(config, maze, chkpt_dir)
    
    # create the experiment
    experiment = Experiment(maze, sac, config=config, discrete=config['game']['discrete'])

    start_experiment = time.time()

    # set the goal
    goal = config["game"]["goal"]

    # training loop 
    # max_timesteps_mode runs with maximum timesteps
    # max_interactions_mode runs with maximum interactions (human-agent actions)
    loop = config['Experiment']['loop']
    if loop == 'max_episodes_mode':
        experiment.max_episodes_mode(goal, maze)
    elif loop == 'max_interactions_mode':
        experiment.max_interactions_mode(goal)
    else:
        print("Unknown training mode")
        exit(1)

    end_experiment = time.time()
    experiment_duration = timedelta(seconds=end_experiment - start_experiment - experiment.duration_pause_total)

    print('Total Experiment time: {}'.format(experiment_duration))

    if config["game"]["save"]:
        # save training logs to a pickle file
        experiment.df.to_pickle(plot_dir + '/training_logs.pkl')

        if not config['game']['test_model']:
            # todo: rename experiment.game to game_step
            total_games = experiment.max_episodes if loop == 'max_episodes_mode' else experiment.total_timesteps
            # save rest of the experiment logs and plot them
            save_logs_and_plot(experiment, chkpt_dir, plot_dir, total_games)
            experiment.save_info(chkpt_dir, experiment_duration, total_games, goal)
    pg.quit()


if __name__ == '__main__':
    main(sys.argv[1:])
    exit(0)
