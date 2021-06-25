import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import softmax
from scipy.stats import entropy
from scipy.stats import dirichlet


class GenerativeModel:
    """
    Generative model class, this class represents an agent, it will observe outcomes and generate actions.
    """
    def __init__(self, size=None, starting_A_alphas=None, starting_B_alphas=None,
                 expected_outcomes=None, starting_state=None):
        """
        :param size: Number of possible states the agent can be in
        :param starting_A_alphas: 2-dimensional list of alphas that inform dirichlet distributions from which
                                    priors over outcomes given states are generated
        :param starting_B_alphas: 3-dimensional list of alphas that inform dirichlet distributions from which
                                    priors over states given action and states are generated.
        :param expected_outcomes: List specifying the expected outcomes of this agent
        :param starting_state: Which state(s) the agent thinks it is in at the start of its lifespan
        """

        if size is None:
            size = 5

        if starting_A_alphas is None:
            starting_A_alphas = np.ones((size, 5))
        predicted_outcomes_given_state = np.zeros((size, size))

        # Generate probability distributions
        for s in range(size):
            predicted_outcomes_given_state[s] = dirichlet.mean(starting_A_alphas[s])

        if starting_B_alphas is None:
            starting_B_alphas = np.ones((3, size, size))
        predicted_future_states_given_previous_state_and_action = np.zeros((3, size, size))

        # Generate probability distributions
        for a in range(starting_B_alphas.shape[0]):
            for s in range(starting_B_alphas.shape[1]):
                predicted_future_states_given_previous_state_and_action[a][s] = dirichlet.mean(starting_B_alphas[a][s])

        if expected_outcomes is None:
            expected_outcomes = [0.2] * 5

        if starting_state is None:
            starting_state = [0.2, 0.2, 0.2, 0.2, 0.2]

        self.A_alphas = starting_A_alphas.copy()  # The A alphas inform the predicted outcomes given a state
        self.B_alphas = starting_B_alphas.copy()  # The B alphas inform the predicted state given an action and state
        self.starting_A_alphas = starting_A_alphas
        self.starting_B_alphas = starting_B_alphas

        self.predicted_outcomes_given_state = predicted_outcomes_given_state  # A: Predicted outcomes given the state
        self.predicted_future_states_given_previous_state_and_action = predicted_future_states_given_previous_state_and_action  # B: Predicted future state given the current state and the control states(?)
        self.expected_outcomes = np.array(expected_outcomes)  # C: Preferred outcomes given the model
        self.current_state = starting_state  # D: Starting state given the model
        self.Actions = [0, 1, 2]

        # Remember which action was taken last timestep, this is important for learning the state transition function
        self.action_taken = 0

    # Actions
    def act(self):
        """
        Generate an action
        :return: The action the agent chooses
        """
        action_qualities = []
        for action in self.Actions:
            # Determine predicted future state based on this action and inferred current state
            predicted_state = np.dot(self.current_state,
                                     self.predicted_future_states_given_previous_state_and_action[action])
            # Determine action quality
            predicted_quality = self.determine_quality(predicted_state, self.predicted_outcomes_given_state,
                                                       self.expected_outcomes)
            action_qualities.append(predicted_quality)

        action_probabilities = softmax(np.array(action_qualities))  # Change action qualities into probabilities
        best_action = np.random.choice(self.Actions, p=action_probabilities)
        self.action_taken = best_action  # Remember chosen action
        return best_action

    # Observations
    def observe(self, outcome):
        """
        Observe the outcome
        :param outcome: The outcome an agent observes
        :return: The state an agent infers it is in and the surprise (prediction error) the agent experienced at the
                    observation, these are for data storing purposes mainly
        """
        # Infer state of the agent
        predicted_state = self.aproximate_bayesian_inference(outcome)
        number_of_states = len(predicted_state)

        surprise = entropy(outcome, np.dot(predicted_state, self.predicted_outcomes_given_state))

        # Update distribution parameters based on outcome and inferred state, weighted by the surprise
        self.A_alphas += np.multiply(np.resize(outcome, (5, 5)).T, predicted_state).T * surprise
        self.B_alphas[self.action_taken] += (np.resize(self.current_state, (
        number_of_states, number_of_states)).T * predicted_state) * surprise

        # Update P(S|O) and P(S_t| A_{t-1}, S_{t-1})
        for s in range(number_of_states):
            self.predicted_future_states_given_previous_state_and_action[self.action_taken][s] = dirichlet.mean(
                self.B_alphas[self.action_taken][s])
            self.predicted_outcomes_given_state[s] = dirichlet.mean(self.A_alphas[s])

        # Update current state of the agent
        self.current_state = predicted_state
        return predicted_state, surprise

    def determine_quality(self, pred_state, pred_outcome_given_state, expected_outcome):
        """
        Determine the quality of a predicted state
        :param pred_state: Predicted state
        :param pred_outcome_given_state: Predicted outcomes given a state
        :param expected_outcome: Expected outcomes
        :return:
        """
        pred_state = np.array(pred_state)
        pred_outcome_given_state = np.array(pred_outcome_given_state)
        expected_outcome = np.array(expected_outcome)

        # Determine predicted outcome
        pred_outcome = np.dot(pred_state, pred_outcome_given_state)

        # Calculate predicted uncertainty as the expectation
        # of the entropy of the outcome, weighted by the
        # probability of that outcome
        pred_ent = np.sum(pred_state * entropy(pred_outcome_given_state, axis=1))

        # Calculate predicted divergence as the Kullback-Leibler
        # divergence between the predicted outcome and the expected outcome
        pred_div = entropy(pk=pred_outcome, qk=expected_outcome)

        # Return the sum of the negatives of these two
        return -pred_ent - pred_div

    def aproximate_bayesian_inference(self, observed_outcome):
        """
        Infer the state of the agent using an approximation of bayesian inference
        :param observed_outcome: The outcome observed by the agent
        :return: A probability distribution over states
        """
        observed_outcome = np.array(observed_outcome)
        expected_outcomes_given_state = np.array(self.predicted_outcomes_given_state)
        previous_action = self.action_taken
        previous_state = np.array(self.current_state)
        expected_future_states_given_control = np.array(self.predicted_future_states_given_previous_state_and_action)

        inferred_state = softmax(np.log(np.dot(expected_outcomes_given_state, observed_outcome)) + np.log(
            np.dot(previous_state, expected_future_states_given_control[previous_action])))
        return inferred_state