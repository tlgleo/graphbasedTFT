import numpy as np
from graphbasedTFT.tit_for_tat import TFT_improved, TFT_inertia, Nice_algo, Egoist_algo, Traitor_algo, LateNice_algo, TFT_improved_beta
from graphbasedTFT.graph_flow_functions import *

class Agent:
    def __init__(self, ident, name, n_agents = 4, max_coop_matrix = None):
        self.ident = ident
        self.name = name
        self.max_coop_matrix = max_coop_matrix # fixed (short term), it can vary in the long rum
        self.current_coop_matrix = max_coop_matrix # elements-wise between max_coop_matrix and coope
        self.n_agents = n_agents
        self.coop_sub_graph = np.ones([self.n_agents, self.n_agents])  # a flow sub graph of max_coop_matrix defining chosen cooperation
        self.step_t = 0

    def reset(self):
        self.current_coop_matrix = self.max_coop_matrix  # elements-wise between max_coop_matrix and coope
        self.coop_sub_graph = np.ones(
            [self.n_agents, self.n_agents])  # a flow sub graph of max_coop_matrix defining chosen cooperation
        self.step_t = 0

    def act(self, coop_matrix, step_t):
        pass


# Agent with graph-structure
class Agent_Graph(Agent):
    def __init__(self, ident, name, n_agents = 4, max_coop_matrix = None, debit_max = 1.0,
                 algo_neg_source = None, algo_neg_coop = None, minCost = True):
        super().__init__(ident, name, n_agents, max_coop_matrix)
        self.algo_tft_coop = algo_neg_coop # TFT function to update current coop matrix
        self.algo_tft_source = algo_neg_source # TFT function of the max flow source
        self.source_max = debit_max
        self.current_source = debit_max
        self.minCost = minCost

    def reset(self):
        self.algo_tft_source.reset()
        self.algo_tft_coop.reset()
        self.current_source = self.source_max
        self.current_coop_matrix = self.max_coop_matrix  # elements-wise between max_coop_matrix and coope
        self.coop_sub_graph = np.ones(
            [self.n_agents, self.n_agents])  # a flow sub graph of max_coop_matrix defining chosen cooperation
        self.step_t = 0

    def compute_coop_sub_graph(self):

        if self.minCost:
            self.coop_sub_graph = subgraph_coop_flowMax_minCost(
                                    self.current_coop_matrix,
                                    agent_i = self.ident,
                                    source_max=self.current_source,
                                    cost_matrix=None
                                    )
        else:
            self.coop_sub_graph = subgraph_coop_flowMax(
                                    self.current_coop_matrix,
                                    agent_i = self.ident,
                                    source_max=self.current_source)


    def update_cooperation_graph(self, coop_matrix):
        # update some cooperation degrees only from himself (ident self.ident)
        outgoing_flow = np.sum(coop_matrix, axis=1)
        outgoing_flow.clip(0,1)
        reaction = self.algo_tft_coop.act(outgoing_flow)
        factor = np.ones([self.n_agents, self.n_agents])
        factor[self.ident,:] = reaction
        self.current_coop_matrix = np.multiply(self.max_coop_matrix, factor)

    def update_source_max(self, coop_matrix):
        # adapt the source max of cooperation I can provide according to what I received
        # apply a TFT on my incoming flow (if 1 then 1, else: less)
        incoming_flow = np.sum(coop_matrix, axis=0)[self.ident]
        incoming_flow = np.clip(incoming_flow,0,1)
        self.current_source = self.source_max * np.round(self.algo_tft_source.act([incoming_flow])[0],3)

    def act(self, coop_matrix, step_t):
        self.update_cooperation_graph(coop_matrix)
        self.update_source_max(coop_matrix)
        self.compute_coop_sub_graph()
        return self.coop_sub_graph[self.ident]


# Agent without graph-structure
class Agent_No_Graph(Agent):
    def __init__(self, ident, name, n_agents = 4, max_coop_matrix = None, debit_max = 1.0,
                 algo_neg_coop = None):
        super().__init__(ident, name, n_agents, max_coop_matrix)
        self.algo_tft= algo_neg_coop

    def reset(self):
        self.algo_tft.reset()
        self.step_t = 0

    def act(self, coop_matrix, step_t):
        receiving_degrees = coop_matrix[:,self.ident]
        sending_degrees = self.algo_tft.act(receiving_degrees)
        return sending_degrees


class Agent_Optimal(Agent):
    def __init__(self, ident, name, n_agents = 4, max_coop_matrix = None, debit_max = 1.0,
                 optimal_subgraph = None):
        super().__init__(ident, name, n_agents, max_coop_matrix)
        self.optimal_subgraph = optimal_subgraph

    def reset(self):
        self.step_t = 0

    def act(self, coop_matrix, step_t):
        sending_degrees = self.optimal_subgraph[self.ident,:]
        return sending_degrees


# Agent with graph-structure and TFT_beta
class Agent_TFT_Beta(Agent_Graph):
    def __init__(self, ident, name, n_agents = 4, max_coop_matrix = None, debit_max = 1.0,
                 alpha_inertia = 0.6, r_incentive = 0.4, beta_adaptive = 0.2, minCost = True):

        algo_tft_source = TFT_improved(alpha_inertia, r_incentive, beta_adaptive, 1)
        algo_tft_agents = TFT_improved(alpha_inertia, r_incentive, beta_adaptive, n_agents)
        super().__init__(ident, name, n_agents, max_coop_matrix,
                         debit_max, algo_neg_source = algo_tft_source,
                         algo_neg_coop = algo_tft_agents, minCost = minCost)
        self.parameters_TFT = (alpha_inertia, r_incentive, beta_adaptive)

# Agent with graph-structure and TFT_gamma
class Agent_TFT_Gamma(Agent_Graph):
    def __init__(self, ident, name, n_agents = 4, max_coop_matrix = None, debit_max = 1.0,
                 alpha_inertia = 0.6, r_incentive = 0.3, beta_adaptive = 0.6, gamma_proba = 0.1, minCost=True):

        algo_tft_source = TFT_improved_beta(alpha_inertia, r_incentive, beta_adaptive, gamma_proba, 1)
        algo_tft_agents = TFT_improved_beta(alpha_inertia, r_incentive, beta_adaptive, gamma_proba, n_agents)
        super().__init__(ident, name, n_agents, max_coop_matrix,
                         debit_max, algo_neg_source=algo_tft_source,
                         algo_neg_coop=algo_tft_agents, minCost=minCost)
        self.parameters_TFT = (alpha_inertia, r_incentive, beta_adaptive, gamma_proba)

# Agent with graph-structure and TFT_alpha
class Agent_TFT(Agent_Graph):
    def __init__(self, ident, name, n_agents = 4, max_coop_matrix = None, debit_max = 1.0,
                 alpha_inertia = 0.5, r_incentive = 0.1, minCost=True):

        algo_tft_source = TFT_inertia(alpha_inertia, r_incentive, 1)
        algo_tft_agents = TFT_inertia(alpha_inertia, r_incentive, n_agents)
        super().__init__(ident, name, n_agents, max_coop_matrix,
                         debit_max, algo_neg_source=algo_tft_source,
                         algo_neg_coop=algo_tft_agents, minCost=minCost)

        self.parameters_TFT = (alpha_inertia, r_incentive)

# Agent without graph-structure and TFT_beta
class Agent_TFT_NoGraph_Beta(Agent_No_Graph):
    def __init__(self, ident, name, n_agents = 4, max_coop_matrix = None, debit_max = 1.0,
                 alpha_inertia = 0.6, r_incentive = 0.7, beta_adaptive = 0.6):

        algo_tft_agents = TFT_improved(alpha_inertia, r_incentive, beta_adaptive, n_agents)
        super().__init__(ident, name, n_agents, max_coop_matrix, debit_max, algo_neg_coop = algo_tft_agents)
        self.parameters_TFT = (alpha_inertia, r_incentive, beta_adaptive)

# Agent without graph-structure and TFT_gamma
class Agent_TFT_NoGraph_Gamma(Agent_No_Graph):
    def __init__(self, ident, name, n_agents = 4, max_coop_matrix = None, debit_max = 1.0,
                 alpha_inertia=0.6, r_incentive=0.3, beta_adaptive=0.6, gamma_proba=0.1):

        algo_tft_agents = TFT_improved_beta(alpha_inertia, r_incentive, beta_adaptive, gamma_proba, n_agents)
        super().__init__(ident, name, n_agents, max_coop_matrix, debit_max, algo_neg_coop = algo_tft_agents)
        self.parameters_TFT = (alpha_inertia, r_incentive, beta_adaptive, gamma_proba)

# Agent without graph-structure and TFT_alpha
class Agent_TFT_NoGraph(Agent_No_Graph):
    def __init__(self, ident, name, n_agents = 4, max_coop_matrix = None, debit_max = 1.0,
                 alpha_inertia = 0.6, r_incentive = 0.2):

        algo_tft_agents = TFT_inertia(alpha_inertia, r_incentive, n_agents)
        super().__init__(ident, name, n_agents, max_coop_matrix, debit_max, algo_neg_coop = algo_tft_agents)
        self.parameters_TFT = (alpha_inertia, r_incentive)

class Agent_Nice(Agent_Graph):
    def __init__(self, ident, name, n_agents = 4, max_coop_matrix = None, debit_max = 1.0):
        super().__init__(ident, name, n_agents, max_coop_matrix, debit_max, algo_neg_source = Nice_algo(), algo_neg_coop = Nice_algo())


class Agent_Egoist(Agent_Graph):
    def __init__(self, ident, name, n_agents = 4, max_coop_matrix = None, debit_max = 1.0):
        super().__init__(ident, name, n_agents, max_coop_matrix, debit_max, algo_neg_source = Egoist_algo(), algo_neg_coop = Egoist_algo())


class Agent_LateNice(Agent_Graph):
    def __init__(self, ident, name, n_agents = 4, max_coop_matrix = None, debit_max = 1.0, t_coop=0):
        super().__init__(ident, name, n_agents, max_coop_matrix, debit_max, algo_neg_source = LateNice_algo(t_coop), algo_neg_coop = LateNice_algo(t_coop))


class Agent_Traitor(Agent_TFT_Gamma):
    def __init__(self, ident, name, n_agents = 4, max_coop_matrix = None, debit_max = 1.0 , t_traitor = [50,80],
                 alpha_inertia = 0.6, r_incentive = 0.6, beta_adaptive = 0.6, gamma_proba = 0.3):
        super().__init__(ident, name, n_agents, max_coop_matrix, debit_max,
                         alpha_inertia=alpha_inertia, r_incentive=r_incentive, beta_adaptive=beta_adaptive, gamma_proba=gamma_proba)
        self.t_traitor = t_traitor
        self.parameters_TFT = (alpha_inertia, r_incentive, beta_adaptive, gamma_proba)

    def act(self, coop_matrix, step_t):
        self.update_cooperation_graph(coop_matrix)
        self.update_source_max(coop_matrix)
        self.compute_coop_sub_graph()

        if step_t < self.t_traitor[0] or step_t >= self.t_traitor[1]:
            print('essai Traitor Nice', step_t)
            return self.coop_sub_graph[self.ident]

        else:
            print('essai Traitor C', step_t)
            return 0.0 * self.coop_sub_graph[self.ident]
