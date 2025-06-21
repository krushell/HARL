from harl.common.base_logger import BaseLogger
from functools import reduce
import numpy as np
import time

class MultiCameraLogger(BaseLogger):
    def __init__(self,args, algo_args, env_args, num_agents, writter, run_dir):
        super(MultiCameraLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        self.win_key = "won"


    def get_task_name(self):
        return self.env_args["scenario"]

    def init(self, episodes):
        """Initialize the logger."""
        self.start = time.time()
        self.episodes = episodes
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []

        self.detected_rate = []
        self.real_detected_rate = []
        self.coverage_rate = []


    def per_step(self, data):
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        self.infos = infos
        dones_env = np.all(dones, axis=1)
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env

        for i in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[i]:
                self.done_episodes_rewards.append(self.train_episode_rewards[i])
                self.detected_rate.append(infos[i][0]["detected_rate"])
                self.real_detected_rate.append(infos[i][0]["real_detected_rate"])
                self.coverage_rate.append(infos[i][0]["coverage_rate"])
                self.train_episode_rewards[i] = 0

    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        self.total_num_steps = (
                self.episode
                * self.algo_args["train"]["episode_length"]
                * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = time.time()
        print("Current Time: ", time.strftime("%m-%d %H:%M:%S", time.localtime(self.end)))
        print(
            "Updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        self.log_train(actor_train_infos, critic_train_info)

        print(
            "Average step reward is {}.".format(
                critic_train_info["average_step_rewards"]
            )
        )


        if len(self.done_episodes_rewards) > 0:
            aver_episode_rewards = np.mean(self.done_episodes_rewards)
            aver_detected_rate = np.mean(self.detected_rate)
            aver_real_detected_rate = np.mean(self.real_detected_rate)
            aver_coverage_rate = np.mean(self.coverage_rate)
            print(
                "{}: average coverage rate is {}, average real coverage rate is {},average detected rate is {}.\n".format(
                self.env_args["scenario"], aver_detected_rate, aver_real_detected_rate,aver_coverage_rate
                )
            )
            self.writter.add_scalars(
                "train_episode_rewards",
                {"aver_rewards": aver_episode_rewards},
                self.total_num_steps,
            )
            self.writter.add_scalars(
                "detected_rate",
                {"aver_detected_rate": aver_detected_rate},
                self.total_num_steps,
            )
            self.writter.add_scalars(
                "real_detected_rate",
                {"aver_real_detected_rate": aver_real_detected_rate},
                self.total_num_steps,
            )

            self.writter.add_scalars(
                "coverage_rate",
                {"aver_coverage_rate": aver_coverage_rate},
                self.total_num_steps,
            )

            self.done_episodes_rewards = []
            self.real_detected_rate = []
            self.detected_rate = []
            self.coverage_rate = []






