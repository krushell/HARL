import copy
import gym
import mate

class MultiCameraEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.env = mate.make(args["scenario"])
        self.n_agents = self.env.num_cameras
        self.share_observation_space = self.repeat(self.env.state_space)
        self.observation_space = self.repeat(self.env.observation_space[0])
        self.action_space = self.repeat(self.env.action_space[0])
        self._seed = 0
        if self.env.action_space[0].__class__.__name__ == "Box":
            self.discrete = False
        else:
            self.discrete = True

        # if "max_cycles" in self.args:
        #     self.max_cycles = self.args["max_cycles"]
        #     self.args["max_cycles"] += 1
        # else:
        #     self.max_cycles = 1000
        #     self.args["max_cycles"] = 1001
        # self.cur_step = 0
        self.env.reset()

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        if self.discrete:
            obs, rew, done, info = self.env.step(actions.flatten()[0])
        else:
            obs, rew, done, info = self.env.step(actions)
        # self.cur_step += 1
        obs = list(obs)
        done = self.repeat(done)
        rew = self.repeat([rew])
        s_obs = self.repeat(self.env.state())
        for agent in range(self.n_agents):
            info[agent]["bad_transition"] = True
        return (
            obs,
            s_obs,
            rew,
            done,
            info,
            self.get_avail_actions(),
        )

    def reset(self):
        """Returns initial observations and states"""
        self._seed+=1
        # self.cur_step = 0
        obs = list(self.env.reset(seed=self._seed))
        s_obs = self.repeat(self.env.state())
        return obs, s_obs, self.get_avail_actions()

    def get_avail_actions(self):
        if self.discrete:
            avail_actions = [[1] * self.action_space[0].n]
            return avail_actions
        else:
            return None

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)
