# -*- coding: utf-8 -*-
import gym
import re
import numpy as np


class OpenAI_MDPToolbox:

    """Class to convert Discrete Open AI Gym environemnts to MDPToolBox environments.
    You can find the list of available gym environments here: https://gym.openai.com/envs/#classic_control
    You'll have to look at the source code of the environments for available kwargs; as it is not well documented.
    """

    def __init__(self, openAI_env_name:str, render:bool=False, **kwargs):
        """Create a new instance of the OpenAI_MDPToolbox class
        :param openAI_env_name: Valid name of an Open AI Gym env
        :type openAI_env_name: str
        :param render: whether to render the Open AI gym env
        :type rander: boolean
        """
        self.env_name = openAI_env_name

        self.env = gym.make(self.env_name, **kwargs)
        self.env.reset()

        if render:
            self.env.render()

        self.transitions = self.env.P
        self.actions = int(re.findall(r'\d+', str(self.env.action_space))[0])
        self.states = int(re.findall(r'\d+', str(self.env.observation_space))[0])
        self.P = np.zeros((self.actions, self.states, self.states))
        self.R = np.zeros((self.states, self.actions))
        self.convert_PR()

    def convert_PR(self):
        """Converts the transition probabilities provided by env.P to MDPToolbox-compatible P and R arrays
        """
        for state in range(self.states):
            for action in range(self.actions):
                for i in range(len(self.transitions[state][action])):
                    tran_prob = self.transitions[state][action][i][0]
                    state_ = self.transitions[state][action][i][1]
                    self.R[state][action] += tran_prob*self.transitions[state][action][i][2]
                    self.P[action, state, state_] += tran_prob


def converter(env_name:str, render:bool=False, **kwargs):
    """
    Generate a MDPToolbox-formatted version of a *discrete* OpenAI Gym environment.
    You can find the list of available gym environments here: https://gym.openai.com/envs/#classic_control
    You'll have to look at the source code of the environments for available kwargs; as it is not well documented.
    This function is used to generate a transition probability
    (``A`` × ``S`` × ``S``) array ``P`` and a reward (``S`` × ``A``) matrix
    ``R``.
    Parameters
    ---------
    env_name : str
        The name of the Open AI gym environment to model.
    render : bool
        Flag to render the environment via gym's `render()` function.

    Returns
    -------
    out : tuple
        ``out[0]`` contains the transition probability matrix P  and ``out[1]``
        contains the reward matrix R.
    Examples
    --------
    >>> import hiive.mdptoolbox.example
    >>> from gym.envs.toy_text.frozen_lake import generate_random_map
    >>> random_map = generate_random_map(size=10, p=0.98)
    >>> P, R = hiive.mdptoolbox.example.openai("FrozenLake-v0", desc=random_map)
    >>> P
    array([[[0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
    <BLANKLINE>
        [0., 0., 0., ..., 1., 0., 0.],
        [0., 0., 0., ..., 0., 1., 0.],
        [0., 0., 0., ..., 0., 0., 1.]]])
    >>> R
    array([[ -1.,  -1.,  -1.,  -1.,  -1., -10.],
       [ -1.,  -1.,  -1.,  -1.,  -1., -10.],
       [ -1.,  -1.,  -1.,  -1.,  -1., -10.],
       ...,
       [ -1.,  -1.,  -1.,  -1., -10., -10.],
       [ -1.,  -1.,  -1.,  -1., -10., -10.],
       [ -1.,  -1.,  -1.,  -1., -10., -10.]])
    >>> P, R = hiive.mdptoolbox.example.openai("Taxi-v3", True)
    +---------+
    |R: | : :G|
    | : | : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+`
    """

    env = OpenAI_MDPToolbox(env_name, render, **kwargs)
    return env.P, env.R
