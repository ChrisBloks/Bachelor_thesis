import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import softmax
from scipy.stats import entropy
from scipy.stats import dirichlet

from Environment_Final.EvolutionaryFunction import EvolutionaryFunction
from Environment_Final.GenerativeModel import GenerativeModel
from Environment_Final.GenerativeProcess import GenerativeProcess


def Evolution_sandbox(T=20, G=2, A=1, size=5, GP=None, EF=None, cont=False):
    """
    This function runs the evolution simulation
    :param T: Number of timesteps per generation
    :param G: Number of generations
    :param A: Number of agents per generation
    :param size: Number of possible states the agent can be in
    :param GP: Generative Process
    :param EF: Evolutionary Function
    :param cont: If this is true, the simulation will continue with the agents the Evolutionary Function already has,
                    otherwise a new simulation is run with a blank-slate first generation
    :return: The Evolutionary Function, the Evolutionary Function contains relevant data
    """
    if GP is None:
        GP = GenerativeProcess(nr_of_possible_states=size, nr_of_agents=A)
    if EF is None:
        EF = EvolutionaryFunction(nr_of_agents=A)

    for g in range(G):
        # print(g)
        agent_states_this_generation = []
        true_agent_states_this_generation = []

        # The first generation, determine if we start from scratch or continue with the agents the
        # EvolutionaryFunction already has.
        if g == 0 and not cont:
            EF.create_new_generation()

        EF.create_new_generation(EF.select_probabilistically(N=A))
        starting_states = generate_starting_locations(EF.nr_of_agents, size)
        GP.agent_states = starting_states
        true_agent_states_this_generation.append(np.sum(starting_states, axis=0) / EF.nr_of_agents)
        surprise_this_generation = []

        for t in range(T):
            agent_states_this_timestep = [0, 0, 0, 0, 0]
            true_agent_states_this_timestep = [0, 0, 0, 0, 0]
            surprise_this_timestep = []
            for a in range(EF.nr_of_agents):
                # Agent acts, environment 'observes'
                agent_action = EF.agents[a].act()
                EF.agent_actions[a][agent_action] += 1
                GP.observe(agent_action, a)

                # Environments 'acts', agent observes
                outcome = GP.act(a)
                inferred_state, surprise = EF.agents[a].observe(outcome)
                EF.agent_utilities[a] += [EF.evolutionary_utility(outcome)]

                # Collect timestep data
                agent_states_this_timestep += inferred_state
                true_agent_states_this_timestep += GP.agent_states[a]
                surprise_this_timestep += [surprise]
            surprise_this_generation += [surprise_this_timestep]

            # Collect data generational data
            agent_states_this_generation.append(np.array(agent_states_this_timestep).T / EF.nr_of_agents)
            true_agent_states_this_generation.append(np.array(true_agent_states_this_timestep).T / EF.nr_of_agents)

        EF.surprise_per_generation += [surprise_this_generation]
        EF.states_per_generation.append(true_agent_states_this_generation)
        EF.store_generation_data()

        # For plotting (Optional)

        # Plot surprise this generation
        plt.figure()
        plt.plot(np.mean(surprise_this_generation, axis=1))
        plt.title('Average surprise of generation over time ' + str(g))
        plt.ylim(-0.1, 4)
        plt.xlabel('Timestep')
        plt.ylabel('Surprise')
        # plt.savefig('Surprise' + str(g))
        plt.show()

        # Plot inferred agent states of this generation
        # take_X_timestep_average(10, agent_states_this_generation, 'Average inferred agent states of generation ' + str(g), 'State probability', 'Timestep')
        plt.figure()
        plt.plot(agent_states_this_generation)
        plt.title('Average inferred state of generation ' + str(g))
        plt.legend(['0', '1', '2', '3', '4'])
        plt.ylim(-0.1, 1.2)
        plt.xlabel('Timestep')
        plt.ylabel('state probability')
        # plt.savefig('Test inferred states' + str(g))
        plt.show()

        # Plot true agent states of this generation
        #         take_X_timestep_average(10, true_agent_states_this_generation, 'Average true state of generation ' + str(g) , 'state', 'Timestep')
        plt.figure()
        plt.plot(true_agent_states_this_generation)
        plt.title('Average true state of generation ' + str(g))
        plt.legend(['0', '1', '2', '3', '4'])
        plt.ylim(-0.1, 1.2)
        plt.xlabel('Timestep')
        plt.ylabel('state')
        # plt.savefig('Test true states' + str(g))
        plt.show()
    return EF


def generate_starting_locations(nr_of_agents, nr_of_possible_states, p=None):
    """
    Generate semi-random starting locations for a number of agents
    :param nr_of_agents: Number of agents
    :param nr_of_possible_states: Number of possible states an agent can be in
    :param p: Probability of each state
    :return: A list of starting locations
    """
    if p is None:
        p = np.ones(nr_of_possible_states) / nr_of_possible_states
    starting_states = []
    for a in range(nr_of_agents):
        starting_states.append(np.random.multinomial(1, p))
    return starting_states


def single_agent(T=1000, GP=None, agent=None, utility=None):
    """
    Function for examining the behaviour of a singular agent
    :param T: Number of timesteps the agent will 'live'
    :param GP: Generative Process
    :param agent: The agent
    :param utility: Utility per outcome
    :return: The agent
    """
    if GP is None:
        GP = GenerativeProcess(nr_of_agents=1)
    if agent is None:
        agent = GenerativeModel()
    if utility is None:
        utility = [0, 2, 4, 6, 8]

    starting_states = generate_starting_locations(1, 5)
    GP.agent_states = [starting_states[0]]
    agent_state = []
    agent_actions = []
    agent_utility = []
    true_agent_state = [starting_states[0]]
    observed_outcomes = []
    agent_surprise = []
    for t in range(T):
        # Agent acts, environment 'observes'
        agent_action = agent.act()
        GP.observe(agent_action, 0)

        # Environments 'acts', agent observes
        outcome = GP.act(0)
        inferred_state, surprise = agent.observe(outcome)

        agent_utility += [[np.dot(utility, outcome)]]

        x = np.zeros(3)
        x[agent_action] = 1
        agent_actions.append(x)
        agent_state.append(inferred_state)
        observed_outcomes.append(outcome)
        agent_surprise.append([surprise])
        true_agent_state.append(GP.agent_states[0])

    avg_t = 1000
    take_X_timestep_average(avg_t, agent_surprise, str(avg_t) + ' Timestep average surprise', 'Timestep', 'Surprise',
                            ylim=(-0.1, 2), legend=['agent utility'])
    take_X_timestep_average(avg_t, agent_utility, str(avg_t) + ' Timestep average utility', 'Timestep', 'Utility',
                            ylim=(-0.1, 20), legend=['agent utility'])
    take_X_timestep_average(avg_t, agent_actions, str(avg_t) + ' Timestep average agent action', 'Timestep',
                            'Action proportion', legend=['left', 'stay', 'right'])
    take_X_timestep_average(avg_t, agent_state, str(avg_t) + ' Timestep average inferred agent state', 'Timestep',
                            'State proportion')
    take_X_timestep_average(avg_t, true_agent_state, str(avg_t) + ' Timestep average true agent state', 'Timestep',
                            'State proportion')
    take_X_timestep_average(avg_t, observed_outcomes, str(avg_t) + ' Timestep average observed outcome', 'Timestep',
                            'Observation proportion',
                            legend=['outcome 0', 'outcome 1', 'outcome 2', 'outcome 3', 'outcome 4'])
    return agent


def take_X_timestep_average(X, values, title=None, xlabel=None, ylabel=None, ylim=None, legend=None):
    """
    Plot a sliding-window timestep average of some list of values
    :param X: Size of the window
    :param values: Some list of values
    :param title: Title of the plot
    :param xlabel: xlabel of the plot
    :param ylabel: ylabel of the plot
    :param ylim: ylim of the plot
    :param legend: legend of the plot
    :return: Returns the list of averages
    """
    if ylabel is None:
        ylabel = 'Y_value'
    if xlabel is None:
        xlabel = 'Timestep'
    if title is None:
        title = 'plot'
    if ylim is None:
        ylim = (-0.1, 1.2)
    if legend is None:
        legend = ['state 0', 'state 1', 'state 2', 'state 3', 'state 4']
    values = np.array(values)
    values = np.insert(values, 0, np.zeros([X, values.shape[1]]), axis=0)
    averages = []
    for t in range(values.shape[0]):
        averages.append(np.sum(values[t - X:t], axis=0) / X)

    plt.figure()
    plt.plot(averages[X:])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(ylim[0], ylim[1])
    plt.legend(legend)
    plt.savefig('THESIS_not_modified ' + str(title))
    plt.show()
    return averages


def run_simulation():
    """
    Runs an evolution simulation,
    Change parameters in this function to adjust the simulation
    :return: Evolutionary Function
    """
    utilities = [1, 5, 25, 125, 625]  # Evolutionary utility of each outcome

    # P_gp(O|S)
    predicted_outcomes_given_state = np.array([[1, 0.0, 0.0, 0.0, 0.0],
                                               [0.05, 0.8, 0.05, 0.05, 0.05],
                                               [0.05, 0.05, 0.8, 0.05, 0.05],
                                               [0.05, 0.05, 0.05, 0.8, 0.05],
                                               [0.05, 0.05, 0.05, 0.05, 0.8]])

    # P_gp(S_{t+1}|A_t,S_t)
    predicted_future_states_given_control = np.array([[[0.1, 0.1, 0, 0, 0.8],
                                                       [0.8, 0.1, 0.1, 0, 0],
                                                       [0, 0.8, 0.1, 0.1, 0],
                                                       [0, 0, 0.8, 0.1, 0.1],
                                                       [0.1, 0, 0, 0.8, 0.1]],

                                                      [[0.9, 0.05, 0, 0, 0.05],
                                                       [0.05, 0.9, 0.05, 0, 0],
                                                       [0, 0.05, 0.9, 0.05, 0],
                                                       [0, 0, 0.05, 0.9, 0.05],
                                                       [0.05, 0, 0, 0.05, 0.9]],

                                                      [[0.1, 0.8, 0, 0, 0.1],
                                                       [0.1, 0.1, 0.8, 0, 0],
                                                       [0, 0.1, 0.1, 0.8, 0],
                                                       [0, 0, 0.1, 0.1, 0.8],
                                                       [0.8, 0, 0, 0.1, 0.1]]])

    T = 1825  # Number of timesteps each agent lives
    G = 20  # Number of generations
    A = 20  # Number of agetns per generation

    GP = GenerativeProcess(nr_of_agents=A, p_states_given_action_and_state=predicted_future_states_given_control,
                           p_outcomes_given_state=predicted_outcomes_given_state)
    EF = EvolutionaryFunction(nr_of_agents=A, utilities_per_state=utilities)
    EF = Evolution_sandbox(T=T, G=G, A=A, GP=GP, EF=EF)
    return EF


# EF = run_simulation()

def run_single_agent():
    """
    Runs a single agent simulation,
    Change parameters in this function to adjust the simulation
    :return: The agent after simulation
    """
    # P_gp(O|S)
    predicted_outcomes_given_state = np.array([[1, 0.0, 0.0, 0.0, 0.0],
                                               [0.05, 0.8, 0.05, 0.05, 0.05],
                                               [0.05, 0.05, 0.8, 0.05, 0.05],
                                               [0.05, 0.05, 0.05, 0.8, 0.05],
                                               [0.05, 0.05, 0.05, 0.05, 0.8]])

    # P_gp(S_{t+1}|A_t,S_t)
    predicted_future_states_given_control = np.array([[[0.1, 0.1, 0, 0, 0.8],
                                                       [0.8, 0.1, 0.1, 0, 0],
                                                       [0, 0.8, 0.1, 0.1, 0],
                                                       [0, 0, 0.8, 0.1, 0.1],
                                                       [0.1, 0, 0, 0.8, 0.1]],

                                                      [[0.9, 0.05, 0, 0, 0.05],
                                                       [0.05, 0.9, 0.05, 0, 0],
                                                       [0, 0.05, 0.9, 0.05, 0],
                                                       [0, 0, 0.05, 0.9, 0.05],
                                                       [0.05, 0, 0, 0.05, 0.9]],

                                                      [[0.1, 0.8, 0, 0, 0.1],
                                                       [0.1, 0.1, 0.8, 0, 0],
                                                       [0, 0.1, 0.1, 0.8, 0],
                                                       [0, 0, 0.1, 0.1, 0.8],
                                                       [0.8, 0, 0, 0.1, 0.1]]])
    A = 1

    GP = GenerativeProcess(nr_of_agents=A, p_states_given_action_and_state=predicted_future_states_given_control,
                           p_outcomes_given_state=predicted_outcomes_given_state)

    agent_A_alphas = (predicted_outcomes_given_state + 0.001)  # Starting A alphas of the agent
    agent_B_alphas = (predicted_future_states_given_control + 0.001)  # Starting B alphas of the agent

    expected_outcomes = [0.2, 0.2, 0.2, 0.2, 0.2]  # Expected outcomes of the agent
    agent = GenerativeModel(starting_A_alphas=agent_A_alphas, starting_B_alphas=agent_B_alphas,
                            expected_outcomes=expected_outcomes)

    agent = single_agent(T=10000, GP=GP, agent=agent)
    return agent

# agent = run_single_agent()


"""
Below are some functions relating to plotting
"""


# Adapted from https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence=0.95):
    """
    Creates confidence intervals for the expected outcomes per generation.
    :data: 3-dimensional data with shape [G, A, O]
            G is the number of generations
            A is the number of agents per generations
            O is the number of possible outcomes
    :return: the means of the data,
                the means plus the positive edge of the confidence interval,
                and the means plus the negative edge of the confidence interval
    """
    a = 1.0 * np.array(data)
    n = a.shape[1]
    m, se = np.mean(a, axis=1), scipy.stats.sem(a, axis=1)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def plot_data(EF, T=1000):
    """
    Plots the average expected outcomes per generation,
            the average surprise per generation,
            the average proportion of actions per generation,
            and the average utility per timestep per generation
    :param EF: Evolutionary Function
    :param T: Timesteps per generation
    """
    y = np.array(EF.expected_outcomes_per_generation)
    n = y.shape[0]
    x = range(n)

    plt.plot(x, np.mean(y, axis=1))
    plt.title("Expected outcomes per generation")
    plt.xlabel('Generation')
    plt.ylabel('Outcome probability')
    plt.legend(['Outcome 0', 'Outcome 1', 'Outcome 2', 'Outcome 3', 'Outcome 4'])
    #     plt.savefig('Expected_outcomes_per_generation')
    plt.show()

    plt.plot(np.mean(np.mean(EF.surprise_per_generation, axis=1), axis=1))
    plt.title("Surprise per generation")
    plt.xlabel('Generation')
    plt.ylabel('Surprise')
    #     plt.savefig('Surprise_per_generation')
    plt.show()

    plt.plot(np.array(EF.actions_per_generation)[:, 0])
    plt.plot(np.array(EF.actions_per_generation)[:, 1])
    plt.plot(np.array(EF.actions_per_generation)[:, 2])
    plt.legend(['Left', 'Stay', 'Right'])
    plt.title('Proportion of actions per generation')
    plt.xlabel('Generation')
    plt.ylabel('Action proportion')
    #     plt.savefig('Agent_actions')
    plt.show()

    plt.plot(np.array(EF.utility_per_generation)[:, 1] / T)
    plt.title('Average utility per timestep per generation')
    plt.xlabel('Generation')
    plt.ylabel('Average utility')
    #     plt.savefig('Average_utility')
    plt.show()


def plot_expected_outcomes(EF):
    """
    Plots the average expected outcomes per generation with (by default) 95% confidence intervals
    :param EF: Evolutionary function
    """
    y = np.array(EF.expected_outcomes_per_generation)
    n = y.shape[0]
    x = range(n)

    plt.plot(x, np.mean(y, axis=1))
    plt.plot(x, np.ones(n) * 0.2, color='black')
    # Outcome 0
    mean, lower, upper = mean_confidence_interval(y[:, :, 0])
    ci_m = np.mean(y[:, :, 0], axis=1)
    plt.fill_between(x, ci_m - lower, ci_m + upper, alpha=.1)

    mean, lower, upper = mean_confidence_interval(y[:, :, 1])
    ci_m = np.mean(y[:, :, 1], axis=1)
    plt.fill_between(x, ci_m - lower, ci_m + upper, alpha=.1)

    mean, lower, upper = mean_confidence_interval(y[:, :, 2])
    ci_m = np.mean(y[:, :, 2], axis=1)
    plt.fill_between(x, ci_m - lower, ci_m + upper, alpha=.1)

    mean, lower, upper = mean_confidence_interval(y[:, :, 3])
    ci_m = np.mean(y[:, :, 3], axis=1)
    plt.fill_between(x, ci_m - lower, ci_m + upper, alpha=.1)

    mean, lower, upper = mean_confidence_interval(y[:, :, 4])
    ci_m = np.mean(y[:, :, 4], axis=1)
    plt.fill_between(x, ci_m - lower, ci_m + upper, alpha=.1)

    plt.title("Expected outcomes per generation")
    plt.xlabel('Generation')
    plt.ylabel('Outcome probability')
    plt.legend(['Outcome 0', 'Outcome 1', 'Outcome 2', 'Outcome 3', 'Outcome 4', 'baseline'])
    #     plt.savefig('Confidence_intervals')
    plt.show()


def plot_expected_outcomes_seperate(EF):
    """
    Plots the average expected outcomes per generation with (by default) 95% confidence intervals,
    each outcome is plotted separately.
    :param EF: Evolutionary Function
    """
    y = np.array(EF.expected_outcomes_per_generation)
    n = y.shape[0]
    x = range(n)

    # Outcome 0
    plt.plot(x, np.mean(y[:, :, 0], axis=1))
    plt.plot(x, np.ones(n) * 0.2, color='black')
    mean, lower, upper = mean_confidence_interval(y[:, :, 0])
    ci_m = np.mean(y[:, :, 0], axis=1)
    plt.fill_between(x, ci_m - lower, ci_m + upper, alpha=.1)
    plt.title("Expected outcomes per generation")
    plt.xlabel('Generation')
    plt.ylabel('Outcome probability')
    plt.legend(['Outcome 0'])
    plt.ylim([0, 1])
    # plt.savefig('Confidence_interval_outcome_0')
    plt.show()

    # Outcome 1
    plt.plot(x, np.mean(y[:, :, 1], axis=1), color='orange')
    plt.plot(x, np.ones(n) * 0.2, color='black')
    mean, lower, upper = mean_confidence_interval(y[:, :, 1])
    ci_m = np.mean(y[:, :, 1], axis=1)
    plt.fill_between(x, ci_m - lower, ci_m + upper, color='orange', alpha=.1)
    plt.title("Expected outcomes per generation")
    plt.xlabel('Generation')
    plt.ylabel('Outcome probability')
    plt.legend(['Outcome 1'])
    plt.ylim([0, 1])
    # plt.savefig('Confidence_interval_outcome_1')
    plt.show()

    # Outcome 2
    plt.plot(x, np.mean(y[:, :, 2], axis=1), color='green')
    plt.plot(x, np.ones(n) * 0.2, color='black')
    mean, lower, upper = mean_confidence_interval(y[:, :, 2])
    ci_m = np.mean(y[:, :, 2], axis=1)
    plt.fill_between(x, ci_m - lower, ci_m + upper, color='green', alpha=.1)
    plt.title("Expected outcomes per generation")
    plt.xlabel('Generation')
    plt.ylabel('Outcome probability')
    plt.legend(['Outcome 2'])
    plt.ylim([0, 1])
    # plt.savefig('Confidence_interval_outcome_2')
    plt.show()

    # Outcome 3
    plt.plot(x, np.mean(y[:, :, 3], axis=1), color='red')
    plt.plot(x, np.ones(n) * 0.2, color='black')
    mean, lower, upper = mean_confidence_interval(y[:, :, 3])
    ci_m = np.mean(y[:, :, 3], axis=1)
    plt.fill_between(x, ci_m - lower, ci_m + upper, color='red', alpha=.1)
    plt.title("Expected outcomes per generation")
    plt.xlabel('Generation')
    plt.ylabel('Outcome probability')
    plt.legend(['Outcome 3'])
    plt.ylim([0, 1])
    # plt.savefig('Confidence_interval_outcome_3')
    plt.show()

    # Outcome 4
    plt.plot(x, np.mean(y[:, :, 4], axis=1), color='purple')
    plt.plot(x, np.ones(n) * 0.2, color='black')
    mean, lower, upper = mean_confidence_interval(y[:, :, 4])
    ci_m = np.mean(y[:, :, 4], axis=1)
    plt.fill_between(x, ci_m - lower, ci_m + upper, color='purple', alpha=.1)
    plt.title("Expected outcomes per generation")
    plt.xlabel('Generation')
    plt.ylabel('Outcome probability')
    plt.legend(['Outcome 4'])
    plt.ylim([0, 1])
    # plt.savefig('Confidence_interval_outcome_4')
    plt.show()


