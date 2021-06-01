from maze_RL.rl_models.utils import plot
import numpy as np

read_dir = 'tmp/max_episodes_mode_discrete_O_O_a_28K_every10_Shafti_descending_fotis_2/'
plot_dir = 'plots/max_episodes_mode_O_O_a_28K_every10_Shafti_descending_fotis_2'

# game_step_durations = 'game_step_durations.csv'
# pure_rewards_test = 'pure_rewards_test.csv'
# test_score_history = 'test_score_history.csv'
#
# total_fps = 'total_fps.csv'
# train_fps = 'train_fps.csv'
# test_fps = 'test_fps.csv'

game_step_durations = np.genfromtxt(read_dir + 'game_step_durations.csv', delimiter=',')
pure_rewards_test = np.genfromtxt(read_dir + 'pure_rewards_test.csv', delimiter=',')
test_score_history = np.genfromtxt(read_dir + 'test_score_history.csv', delimiter=',')
total_fps = np.genfromtxt(read_dir + 'total_fps.csv', delimiter=',')
train_fps = np.genfromtxt(read_dir + 'train_fps.csv', delimiter=',')
test_fps = np.genfromtxt(read_dir + 'test_fps.csv', delimiter=',')

plot(game_step_durations, plot_dir + "/game_step_durations.png",
     x=[i + 1 for i in range(len(game_step_durations))])

plot(pure_rewards_test, plot_dir + "/pure_rewards_test.png",
     x=[i + 1 for i in range(len(pure_rewards_test))])

plot(test_score_history, plot_dir + "/test_score_history.png",
     x=[i + 1 for i in range(len(test_score_history))])

plot(total_fps, plot_dir + "/total_fps.png",
     x=[i + 1 for i in range(len(total_fps))])

plot(train_fps, plot_dir + "/train_fps.png",
     x=[i + 1 for i in range(len(train_fps))])

plot(test_fps, plot_dir + "/test_fps.png",
     x=[i + 1 for i in range(len(test_fps))])
