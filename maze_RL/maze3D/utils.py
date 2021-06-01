import math
import numpy as np
from scipy.spatial import distance

from maze_RL.maze3D.config import left_down, right_down, left_up, center
from maze_RL.rl_models.utils import plot_learning_curve, plot, plot_test_score

goals = {"left_down": left_down, "left_up": left_up, "right_down": right_down}

goal_offset = 24


def checkTerminal(ball, goal):
    goal = goals[goal]
    if distance.euclidean([ball.x, ball.y], goal) < 5:
        return True
    return False


def get_distance_from_goal(ball, goal):
    goal = goals[goal]
    return math.sqrt(math.pow(ball.x - goal[0], 2) + math.pow(ball.y - goal[1], 2))


def convert_actions(actions):
    # gets a list of 4 elements. it is called from getKeyboard()
    action = []
    if actions[0] == 1:
        action.append(1)
    elif actions[1] == 1:
        action.append(2)
    else:
        action.append(0)
    if actions[2] == 1:
        action.append(1)
    elif actions[3] == 1:
        action.append(2)
    else:
        action.append(0)
    return action


def save_logs_and_plot(experiment, chkpt_dir, plot_dir, max_episodes):
    # score_history a list with the reward for each episode
    x = [i + 1 for i in range(len(experiment.score_history))]
    np.savetxt(chkpt_dir + '/scores.csv', np.asarray(experiment.score_history), delimiter=',')

    # action_history as returned by get_action_pair: a dyad agent and human {-1,0,1}
    actions = np.asarray(experiment.action_history)

    x_actions = [i + 1 for i in range(len(actions))]
    # Save logs in files
    np.savetxt(chkpt_dir + '/actions.csv', actions, delimiter=',')
    # np.savetxt('tmp/sac_' + timestamp + '/action_side.csv', action_side, delimiter=',')
    np.savetxt(chkpt_dir + '/epidode_durations.csv', np.asarray(experiment.game_duration_list), delimiter=',')

    np.savetxt(chkpt_dir + '/game_step_durations.csv', np.asarray(experiment.step_duration_list), delimiter=',')
    np.savetxt(chkpt_dir + '/online_update_durations.csv', np.asarray(experiment.online_update_duration_list), delimiter=',')
    np.savetxt(chkpt_dir + '/total_fps.csv', np.asarray(experiment.env.fps_list), delimiter=',')
    np.savetxt(chkpt_dir + '/train_fps.csv', np.asarray(experiment.train_fps_list), delimiter=',')
    np.savetxt(chkpt_dir + '/test_fps.csv', np.asarray(experiment.test_fps_list), delimiter=',')

    np.savetxt(chkpt_dir + '/distance_travel.csv', np.asarray(experiment.distance_travel_list), delimiter=',')
    np.savetxt(chkpt_dir + '/distance_travel_test.csv', np.asarray(experiment.test_distance_travel_list), delimiter=',')
    np.savetxt(chkpt_dir + '/pure_rewards.csv', np.asarray(experiment.reward_list), delimiter=',')
    np.savetxt(chkpt_dir + '/pure_rewards_test.csv', np.asarray(experiment.test_reward_list), delimiter=',')

    # length_list is the last timestep of the episode
    np.savetxt(chkpt_dir + '/avg_length_list.csv', np.asarray(experiment.length_list), delimiter=',')
    np.savetxt(chkpt_dir + '/grad_updates_durations.csv', experiment.grad_updates_durations, delimiter=',')

    # test logs
    np.savetxt(chkpt_dir + '/test_episode_duration_list.csv', experiment.test_game_duration_list, delimiter=',')
    np.savetxt(chkpt_dir + '/test_score_history.csv', experiment.test_score_history, delimiter=',')
    np.savetxt(chkpt_dir + '/test_length_list.csv', experiment.test_length_list, delimiter=',')

    plot_learning_curve(x, experiment.score_history, plot_dir + "/scores.png")
    plot(experiment.length_list, plot_dir + "/length.png", x=[i + 1 for i in range(max_episodes)])
    plot(experiment.game_duration_list, plot_dir + "/epidode_durations.png",
         x=[i + 1 for i in range(max_episodes)])

    plot(experiment.step_duration_list, plot_dir + "/game_step_durations.png",
         x=[i + 1 for i in range(len(experiment.step_duration_list))])
    plot(experiment.test_step_duration_list, plot_dir + "/test_game_step_durations.png",
         x=[i + 1 for i in range(len(experiment.test_step_duration_list))])

    plot(experiment.online_update_duration_list, plot_dir + "/online_updates_durations.png",
         x=[i + 1 for i in range(len(experiment.online_update_duration_list))])
    plot(experiment.env.fps_list, plot_dir + "/total_fps.png",
         x=[i + 1 for i in range(len(experiment.env.fps_list))])
    plot(experiment.train_fps_list, plot_dir + "/train_fps.png",
         x=[i + 1 for i in range(len(experiment.train_fps_list))])
    plot(experiment.test_fps_list, plot_dir + "/test_fps.png",
         x=[i + 1 for i in range(len(experiment.test_fps_list))])

    plot(experiment.grad_updates_durations, plot_dir + "/grad_updates_durations.png",
         x=[i + 1 for i in range(len(experiment.grad_updates_durations))])

    # plot test logs
    x = [i + 1 for i in range(len(experiment.test_length_list))]
    plot_test_score(experiment.test_score_history, plot_dir + "/test_scores.png")
    plot(experiment.test_length_list, plot_dir + "/test_length.png",
         x=x)
    plot(experiment.test_game_duration_list, plot_dir + "/test_episode_duration.png",
         x=x)

    plot_learning_curve(x, experiment.reward_list, plot_dir + "/rewards_train.png")
    plot_learning_curve(x, experiment.test_reward_list, plot_dir + "/rewards_test.png")

    try:
        plot_test_score(experiment.score_history, plot_dir + "/test_scores_mean_std.png")
    except:
        print("An exception occurred while plotting")
