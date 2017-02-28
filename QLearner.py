"""
Reinforcement learning using a QLearning algorithm.
"""

import numpy as np
import random as rand


class QLearner(object):
    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.5,
                 radr=0.99,
                 dyna=0,
                 verbose=False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna
        self.num_actions = num_actions
        self.num_states = num_states

        self.Q = np.random.uniform(-1.0, 1.0, (num_states, num_actions))

        # initialize T table for dynaQ
        self.T_count = np.empty([num_states, num_actions, num_states])
        self.T_count.fill(0.000001)
        self.T = np.zeros([num_states, num_actions, num_states])

        # initialize R table for dynaQ
        self.R = np.zeros((num_states, num_actions))

    def __update_Q(self, s, a, s_prime, r):
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (
            r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime])])

    def __dyna(self):
        for step in range(self.dyna):
            s = np.random.randint(self.num_states)
            a = np.random.randint(self.num_actions)
            s_prime = np.argmax(self.T[s, a])
            r = self.R[s, a]
            self.__update_Q(s, a, s_prime, r)

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        if np.random.rand() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s])
        if self.verbose:
            print("s ={} a={}".format(s, action))
        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        # update q table
        self.__update_Q(self.s, self.a, s_prime, r)

        # build T table
        self.T_count[self.s, self.a, s_prime] += 1
        self.T[self.s, self.a, s_prime] = self.T_count[self.s, self.a, s_prime] / np.sum(self.T_count[self.s, self.a])

        # build R table
        self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r

        if self.dyna > 0:
            self.__dyna()

        # decide on random action
        if np.random.rand() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s_prime])

        # update the random action rate
        self.rar *= self.radr

        # update the last state and action for next iteration
        self.s = s_prime
        self.a = action

        if self.verbose:
            print("s={} a={} r={}".format(s_prime, action, r))
        return action

    def __log_message(self, message):
        if self.verbose:
            print(message)


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
