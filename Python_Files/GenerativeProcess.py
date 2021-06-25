import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import softmax
from scipy.stats import entropy
from scipy.stats import dirichlet


class GenerativeProcess:
    """
    Generative process class, responsible for generating observations and conforming to agent actions
    """
    def __init__(self, nr_of_possible_states=None, nr_of_possible_outcomes=None,
                 p_outcomes_given_state=None,
                 p_states_given_action_and_state=None, starting_states=None,
                 nr_of_agents=1):
        """
        Initialization function for the generative process
        :param nr_of_possible_states: Number of possible states an agent can be in
        :param nr_of_possible_outcomes: Number of possible outcomes an agent can observe
        :param p_outcomes_given_state: Probability of each outcome given a state
        :param p_states_given_action_and_state: Probability of each state given an action and state
        :param starting_states: True state in which each agent starts
        :param nr_of_agents: Number of agents
        """
        if nr_of_possible_states is None:
            nr_of_possible_states = 5

        if nr_of_possible_outcomes is None:
            nr_of_possible_outcomes = 5

        if p_states_given_action_and_state is None:
            p_states_given_action_and_state = self.create_transition_matrix(nr_of_possible_states)

        if p_outcomes_given_state is None:
            p_outcomes_given_state = np.zeros((nr_of_possible_states, nr_of_possible_outcomes))
            p_outcomes_given_state[0, 0] = 1  # Always get nothing in the DR
            p_outcomes_given_state[1:nr_of_possible_states] = [
                                                                          1 / nr_of_possible_outcomes] * nr_of_possible_outcomes

        if starting_states is None:
            starting_states = [[1, 0, 0, 0, 0]] * nr_of_agents

        self.agent_states = starting_states

        self.p_outcomes_given_state = p_outcomes_given_state  # A: Expected outcomes given the state
        self.p_future_states_given_control = p_states_given_action_and_state  # B: Expected future state given the current state and the control states(?)
        self.actions = [0, 1, 2]

    def act(self, agent_i):
        """
        Generate an observation and an outcome for an agent
        :param agent_i: Agent index, this is required because the generative process keeps track of the true state of
                        every agent separately
        :return: An outcome based on the agent's true state
        """
        agent_state = self.agent_states[agent_i]

        # Sample an outcome
        outcome = np.random.multinomial(1, np.dot(agent_state,
                                                  self.p_outcomes_given_state))
        return outcome

    # Observations
    def observe(self, action, agent_i):
        """
        Observe an action made by an agent
        :param action: The action made by an agent
        :param agent_i: Agent index, this is required because the generative process keeps track of the true state of
                        every agent separately
        """
        # Sample agent state
        future_state = np.random.multinomial(1, np.dot(self.agent_states[agent_i],
                                                       self.p_future_states_given_control[action]))
        self.agent_states[agent_i] = future_state  # Update agent state

    def create_transition_matrix(self, size=5):
        """
        Create Default state transition matrix
        :param size: Number of possible states agent can be in
        :return: Default state transition matrix
        """
        transition_matrix = np.zeros((2, size, size))
        transition_matrix[0, :, 0] = 0.9
        transition_matrix[0, :, 1:size] = 0.025
        transition_matrix[1, :, 0] = 0.05
        transition_matrix[1, :, 1:size] = 0.95 / (size - 1)
        return transition_matrix

