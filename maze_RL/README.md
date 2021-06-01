# Maze 3D Collaborative Learning on shared task

### Description
A human-agent collaborative game in a [virtual environment](https://github.com/ThanasisTs/maze_RL/tree/main/maze3D) based on the work of Shafti et al. (2020) [1]. Collaborative Learning is achieved through Deep Reinforcement Learning (DRL). The Soft-Actor Critic (SAC) algorithm is used [2] with modifications for discrete action space [3].

### Installation
* Run `source install_dependencies/install.sh`. A python virtual environment will be created and the necessary libraries will be installed. Furthermore, the directory of the repo will be added to the `PYTHONPATH` environmental variable.

### Run
* Run `python game/maze3d_human_only_test.py game/config/config_human_test.yaml` for human-only game.
* Run `python game/sac_maze3d_train.py game/config/<config_sac> <participant_name>` for human-agent game.
  * Notes before training: 
     * Set the <participant_name> to the name of the participant.
     * The program will create a `/tmp` and a `/plot` folder (if they do not exist). The `/tmp` folder contains CSV files with information of the game. The `/plot` folder contains figures for tha game. See [here](#Experiment-Result-Output-Files) for more details.
     * The program will automatically create an identification number after your name on each folder name created
  

### Configuration
* In the game/config folder several YAML files exist for the configuration of the game. The main parameters are listed below.
    * `game/discrete`: True if the keyboard input is discrete (False for continuous). Details regarding the discrete and continuous human input mode can be found [here](https://github.com/ThanasisTs/maze_RL/tree/main/game)
    * `SAC/reward_function`: Type of reward function. Details about the predefined reward funtions and how to define a new one can be found [here](https://github.com/ThanasisTs/maze_RL/tree/main/game).
    * `Experiment/loop`: Choose how the game will be terminated; either when a number of episodes or a number of interactions is completed.
    * `SAC/discrete`: Discrete or normal SAC (Currently only the discrete SAC is compatible with the game)
  
### Play
![Game](https://github.com/ThanasisTs/maze_RL/blob/main/pictures/maze_tilt.png)

* `Human only`: Use Left and Right arrows to control the tilt of the tray around its y-axis and use Up and Down arrows to control the tile of the tray around its x-axis as shown in the previous picture
* `Human-Agent`: Use Left and Right arrows to control the tilt of the tray around its y-axis
* Press once the spacekey to pause and a second time to resume
* Press q to exit the experiment.

## Citation

If you use this repository in your publication please cite below:
```
Fotios Lygerakis, Maria Dagioglou, and Vangelis Karkaletsis. 2021. Accelerating Human-Agent Collaborative Reinforcement Learning. InThe 14th PErvasive Technologies Related to Assistive Environments Conference (PETRA2021), June 29-July 2, 2021, Corfu, Greece.ACM, New York, NY, USA, 3 pages.https://doi.org/10.1145/3453892.3454004
```
### Experiment Result Output Files
Contents of a`/tmp` folder. The terms "training/testing _trial_", "game step" and "experiment" are explained in [4] in detail:
  * `actions.csv`: All the actions performed during the experiment in the format ( a<sup>agent</sup> [4], a<sup>human</sup>).
  * `avg_length_list.csv`: The length of each training _trial_ in terms of game step.
  * `test_length_list.csv`: The length of each test _trial_ in terms of game step.
  * `config_sac.yaml`: The configuration file used for this experiment. It's purpose it to be able to replicate this experiment.
  * `episode_durations.csv`: The total duration of each training _trial_.
  * `test_episode_duration_list.csv`: The total duration of each testing _trial_.
  * `grad_updates_durations.csv`: The total duration of an offline gradient update session for each _trial_. In combination with the `episode_durations.csv` are used to calculate the cumulative total time elapsed as shown on Figure 4 of [4]. 
  * `scores.csv`: The total score for each training _trial_.
  * `test_score_history.csv`: The total score for each testing _trial_. The mean and standard error of the mean over each session is used in [4] for figures 2 and 3
  * `rest_info.csv`: goal position, total experiment duration, best score achieved, the _trial_ that achieved the best score, the best reward achieved, the length of the game trial with the best score, the total amount of time steps for the whole experiment, the total number of _games_ played, the fps the game run on and the average offline gradient update duration over all sessions.

Contents of a`/plot` folder:
  * `episode_durations.png`
  * `grad_updates_durations.png`
  * `length.png`
  * `scores.png`
  * `test_episode_duration.png`
  * `test_length.png`
  * `test_scores.png`
  * `test_scores_mean_std.png`
  * `training_logs.pkl`: a pandas framework saves in pickle format that contains the action and state for each training game step.

### References
[1] Shafti, Ali, et al. "Real-world human-robot collaborative reinforcement learning." arXiv preprint arXiv:2003.01156 (2020).

[2] https://github.com/kengz/SLM-Lab

[3] Christodoulou, Petros. "Soft actor-critic for discrete action settings." arXiv preprint arXiv:1910.07207 (2019).

[4] Fotios Lygerakis, Maria Dagioglou, and Vangelis Karkaletsis. 2021. Accelerating Human-Agent Collaborative Reinforcement Learning. InThe 14th PErvasive Technologies Related to Assistive Environments Conference (PETRA2021), June 29-July 2, 2021, Corfu, Greece.ACM, New York, NY, USA, 3 pages.https://doi.org/10.1145/3453892.3454004
