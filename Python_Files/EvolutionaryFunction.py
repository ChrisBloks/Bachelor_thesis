import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import softmax
from scipy.stats import entropy
from scipy.stats import dirichlet

from Environment_Final.GenerativeModel import GenerativeModel


class EvolutionaryFunction:
    """
    Evolutionary function class, responsible for creating new generations of agents and keeping track of agent data
    """

    def __init__(self, nr_of_agents=20, agent_nr_of_possible_states=5, utilities_per_state=None,
                 agent_expected_outcomes=None,
                 agents=None):
        """
        Initialization function of the EvolutionaryFunction
        :param nr_of_agents: Number of agents per generation
        :param agent_nr_of_possible_states: Number of possible states an agent can be in
        :param utilities_per_state: List specifying the evolutionary utility per state
        :param agent_expected_outcomes:  List specifying the default expected outcomes of agents
        :param agents: List of agents that takes the form of [[agents], [agent utilities]]
        """
        if utilities_per_state is None:
            utilities_per_state = [0, 1, 2, 3, 4]  # Default evolutionary utility per state

        if agent_expected_outcomes is None:
            agent_expected_outcomes = [0.2, 0.2, 0.2, 0.2, 0.2]  # By default an agent does not expect anything

        if agents is None:
            self.agents = []  # In this case agents should be created using the create_new_generation() function
            self.agent_utilities = []
        else:
            self.agents = agents[0]
            self.agent_utilities = agents[1]

        self.nr_of_agents = nr_of_agents
        self.agent_nr_of_possible_states = agent_nr_of_possible_states
        self.agent_actions = []
        self.utilities_per_state = utilities_per_state
        self.current_generation_states = [0, 0, 0, 0, 0]
        self.agent_expected_outcomes = agent_expected_outcomes

        # Generational Data collection
        # Comments show default values, same values with which simulations were run
        self.utility_per_generation = []  # [0: Total comulative, 1: average utlity, 2: most utility]
        self.actions_per_generation = []  # [0: Proportion left, 1: Proportion stay, 2: proportion right]
        self.expected_outcomes_per_generation = []  # [p(o_0), p(o_1), p(o_2), p(o_3), p(o_4)]
        self.surprise_per_generation = []  # [Average surprise per agent per generation]
        self.agents_per_generation = []  # [By default always same as nr_of_agents]
        self.states_per_generation = []  # [state occupation per agent per generation]

    def create_new_generation(self, new_generation=None):
        """
        For creating either a blank-slate generation or setting a new generation with a list of agents
        :param new_generation: optional list of agents
        """
        # By default this happens
        if new_generation == None:
            self.agents = [
                GenerativeModel(size=self.agent_nr_of_possible_states, expected_outcomes=self.agent_expected_outcomes)
                for i in range(self.nr_of_agents)]  # Create a new blank-slate generation
            self.agent_utilities = np.ones(self.nr_of_agents)
            self.agent_actions = np.zeros((self.nr_of_agents, 3))  # Assuming 3 possible actions

        # If we have already created a new generation (using select_probabilistically() for example)
        else:
            self.agents = new_generation
            self.agent_utilities = np.zeros(len(new_generation))
            self.agent_actions = np.zeros((self.nr_of_agents, 3))  # Assuming 3 possible action
            self.current_generation_states = [0, 0, 0, 0, 0]

    def select_probabilistically(self, N, killoff=None):
        """
        Probabilistically selects parents for a new generation of agents, then creates the agents.
        :param N: Number of agents
        :param killoff: A number specifying how many of the bottom agents to kill off (0% chance of reproduction)
        :return: A new generation of agents
        """
        agent_utilities = np.array(self.agent_utilities)  # For normal evolutionary pressure
        total_utility = np.sum(agent_utilities)
        p = agent_utilities / total_utility

        # if killoff is None:  # Uncomment this for heightened evolutionary pressure
        #     killoff = round(N / 10)  # Kill off the bottom (least evolutionary utility) 10% of agents
        # eligible_agents = np.argsort(agent_utilities)[killoff:]  # Select agents which may reproduce
        # total_utility = np.sum(agent_utilities[eligible_agents])
        # p = np.zeros(N)
        # p[eligible_agents] = agent_utilities[eligible_agents] / total_utility  # Specify the probabilities of
        # # reproducing for each agent


        new_generation = []
        parents = np.random.choice(a=np.arange(N), size=N, p=p)  # Select parents
        # parents = np.append(parents, np.argmax(agent_utilities))  # Optional: Best performing agent always reproduces

        for p in parents:
            # Mutate Prior over outcomes given state
            new_predicted_outcomes_given_state = self.mutate_predicted_outcomes_given_state(
                self.agents[p].starting_A_alphas) + \
                                                 self.agents[p].A_alphas / \
                                                 np.sum(self.agents[p].A_alphas, axis=1)[
                                                     ..., None] * 5 # Normalize and multiply by 5 so that ending alphas have limited influence on the starting alphas

            # Mutate Prior over state given action and state
            new_state_transition_function = self.mutate_state_transition_function(self.agents[p].starting_B_alphas) + \
                                            self.agents[p].B_alphas / \
                                            np.sum(self.agents[p].B_alphas, axis=2)[
                                                ..., None] * 5  # Normalize and multiply by 5 so that ending alphas have limited influence on the starting alphas

            # Mutate Expected outcomes
            new_expected_outcomes = self.mutate_expected_outcomes(self.agents[p].expected_outcomes)

            # Add child to the new generation
            new_generation += [
                GenerativeModel(expected_outcomes=new_expected_outcomes,
                                starting_B_alphas=new_state_transition_function,
                                starting_A_alphas=new_predicted_outcomes_given_state)]
        return new_generation

    def mutate_predicted_outcomes_given_state(self, predicted_outcomes_given_state, mu=0, sigma=0.5):
        """
        Mutates the given list of predicted outcomes per state by adding a sample from a normal distribution
        to the probability of each state, then normalizing.
        :param predicted_outcomes_given_state: Must be a 2-dimensional list or ndarray
        :param sigma: standard deviation
        :param mu: mean
        :return:
        """
        predicted_outcomes_given_state = np.array(predicted_outcomes_given_state)
        new_predicted_outcomes_given_state = predicted_outcomes_given_state + np.random.normal(mu, sigma,
                                                                                               predicted_outcomes_given_state.shape)
        new_predicted_outcomes_given_state[
            new_predicted_outcomes_given_state <= 0] = 0.001  # Alphas must be larger than 0
        new_predicted_outcomes_given_state = new_predicted_outcomes_given_state / \
                                             np.sum(new_predicted_outcomes_given_state, axis=1)[
                                                 ..., None] * 5  # Multiple by 5 and normalize so that alpha values do not grow too large
        return new_predicted_outcomes_given_state

    def mutate_state_transition_function(self, state_transition_function, mu=0, sigma=0.5):
        """
        Mutates the given list of predicted states per action and state by adding a sample from a normal distribution
        to the probability of each state, then normalizing.
        :param state_transition_function: Must be a 3-dimensional list or ndarray
        :param sigma: standard deviation
        :param mu: mean
        :return: Mutated version of the state transition function
        """
        state_transition_function = np.array(state_transition_function)
        new_state_transition_function = state_transition_function + np.random.normal(mu, sigma,
                                                                                     state_transition_function.shape)
        new_state_transition_function[new_state_transition_function <= 0] = 0.001  # Alphas must be larger than 0
        new_state_transition_function = new_state_transition_function / \
                                        np.sum(new_state_transition_function, axis=2)[
                                            ..., None] * 5  # Multiple by 5 and normalize so that alpha values do not grow too large

        return new_state_transition_function

    def mutate_expected_outcomes(self, expected_outcomes, mu=0, sigma=0.02):
        """
        Mutates the given list of expected outcomes by adding a sample from a normal distribution to the probability
        of each outcome, then normalizing.
        :param expected_outcomes: Must be a 1-dimensional list or ndarray
        :param sigma: standard deviation
        :param mu: mean
        :return: Mutated version of expected outcomes
        """
        expected_outcomes = np.array(expected_outcomes)
        new_expected_outcomes = np.clip(expected_outcomes + np.random.normal(mu, sigma, expected_outcomes.shape), 0.001,
                                        1)
        norm = np.sum(new_expected_outcomes)  # Normalization
        new_expected_outcomes /= norm
        return new_expected_outcomes

    def evolutionary_utility(self, outcome):
        """
        Returns the utility of a certain outcome
        :param outcome: array specifying outcome(s)
        :return: evolutionary utility of outcomes
        """
        utility = np.dot(self.utilities_per_state, outcome)
        return utility

    def store_generation_data(self):
        """
        Store relevant data of this generation for plotting purposes
        """
        total_utility = np.sum(self.agent_utilities)
        average_utility = total_utility / len(self.agent_utilities)
        most_utility = np.max(self.agent_utilities)
        self.utility_per_generation.append([total_utility, average_utility, most_utility])
        self.actions_per_generation.append(np.sum(self.agent_actions, axis=0) / np.sum(
            self.agent_actions))  # Only store the average proportion of actions
        self.expected_outcomes_per_generation.append(
            self.get_expected_outcomes())  # Only store the average expected outcomes
        self.agents_per_generation.append(
            self.nr_of_agents)  # Normally this should always be the same, may not be for different ways of generation creation

    def get_expected_outcomes(self):
        """
        Get the average expected outcomes of the current generation
        :return: Average expected outcomes per agent
        """
        expected_outcomes = np.zeros((self.nr_of_agents, self.agents[0].expected_outcomes.shape[0]))
        for i, a in enumerate(self.agents):
            expected_outcomes[i] = a.expected_outcomes
        return expected_outcomes

    def get_avg_expected_outcomes(self):
        """
        Get the average expected outcomes of the current generation
        :return: Average expected outcomes per agent
        """
        average_expected_outcomes = np.zeros(self.agent_nr_of_possible_states)
        for a in self.agents:
            average_expected_outcomes += a.expected_outcomes
        average_expected_outcomes /= len(self.agents)
        return average_expected_outcomes
