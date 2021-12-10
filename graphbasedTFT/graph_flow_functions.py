import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
import networkx as nx


def coopgraph2flowgraph(graph_coop, agent_i, source_max=1.0, Q_norm=100):
    """
    Convert a cooperation graph of N agents into a integer flow graph with a source and a sink
    connected to the vertex of agent_i which is able to provide a max cooperation equal to source_max
    INPUT:
    - an (N,N) array reprenting a graph of cooperation of N agents with cooperation degrees in [0,1]
    - an ident agent_i from where the cooperation start
    - a cooperation max from agent_i
    - Q_norm : an integer for the transformation of the graph into an integer graph
    OUPUT:
    - an integer matrix (N+2, N+2) representing a flow network of cooperation with
    a source (index 0) and a sink (index N+1), the cooperations degrees are integers between 0 and Q_norm
    """

    (n, _) = np.shape(graph_coop)
    new_graph = np.zeros([n + 2, n + 2])
    new_agent_i = agent_i + 1  # with the addition of source vertex in index 0, the index of agent_i is modified
    new_graph[0, new_agent_i] = source_max  # the max capacity of cooperation : source_max in the edge source-agent_i

    for k in range(n):
        for p in range(n):
            # for all edge (k,p):

            if p == agent_i:  # k is an incoming neighbor of agent_i
                # then the edge from k (k+1 in new_graph) is put toward the sink (n+1)
                new_graph[k + 1, n + 1] = graph_coop[k, p]
            else:
                new_graph[k + 1, p + 1] = graph_coop[k, p]

    # normalisation and convertion to an integer matrix
    output = (Q_norm * new_graph).astype(int)

    return output



def coopgraph2flowDict(graph_coop, agent_i, source_max=1.0, Q_norm=100, cost_matrix=None):
    """
    Same function of coopgraph2flowgraph with dict representation
    Convert a cooperation graph of N agents into a integer flow graph with a source and a sink
    connected to the vertex of agent_i which is able to provide a max cooperation equal to source_max
    INPUT:
    - an (N,N) array reprenting a graph of cooperation of N agents with cooperation degrees in [0,1]
    - an ident agent_i from where the cooperation start
    - a cooperation max from agent_i
    - Q_norm : an integer for the transformation of the graph into an integer graph
    OUPUT:
    - an integer matrix (N+2, N+2) representing a flow network of cooperation with
    a source (index 0) and a sink (index N+1), the cooperations degrees are integers between 0 and Q_norm
    """

    (n, _) = np.shape(graph_coop)
    new_source_max = int(Q_norm * source_max)  # renormalisation into an integer
    new_graph_coop = (Q_norm * graph_coop).astype(int)  # renormalisation into an integer

    if cost_matrix is None:
        cost_matrix = -1.0 * np.ones([n, n])  # constant negative cost (for benefice)

    G = nx.DiGraph()  # new graph with dict representation from library networkx
    new_agent_i = agent_i + 1  # with the addition of source vertex in index 0, the index of agent_i is modified

    # the max capacity of cooperation : source_max in the edge source-agent_i
    G.add_edges_from([(0, new_agent_i, {"capacity": new_source_max, "weight": 0})])

    for k in range(n):
        for p in range(n):
            # for all edge (k,p):

            if p == agent_i:  # k is an incoming neighbor of agent_i
                # then the edge from k (k+1 in new_graph) is put toward the sink (n+1)
                G.add_edges_from([(k + 1, n + 1, {"capacity": new_graph_coop[k, p], "weight": cost_matrix[k, p]})])
            else:
                G.add_edges_from([(k + 1, p + 1, {"capacity": new_graph_coop[k, p], "weight": cost_matrix[k, p]})])

    return G


def flowDict2array(dict_graph):
    # converts a graph (dict) into a numpy graph (weighted adjacency matrix)
    n = len(dict_graph)
    numpy_graph = np.zeros([n,n])
    for i in dict_graph:
        if len(dict_graph[i]) > 0:
            for j in dict_graph[i]:
                numpy_graph[i,j] = dict_graph[i][j]
    return numpy_graph



def flowgraph2coopgraph(flow_graph, agent_i, Q_norm=100):
    """
    Convert a residual flow graph (array) associated to agent_i into a cooperation graph
    1. keep only non negative values (corresponding of the right way of the directed graph)
    2. Removes the source and sink and renormalise (with Q_norm) in a float graph (in [0,1])
    rows 0 and (n+1) useless because corresponding to source and sink
    column (n+1) of the sink has to be placed in column agent_i
    """

    output = np.maximum(0, flow_graph).astype(float)  # keep only right way edges of the directed graph
    output /= Q_norm  # normalise
    output[:, agent_i + 1] = output[:, -1]  # connects the sink to the vertex of agent_i
    output = output[1:-1, 1:-1]  # removes the vertices of source and sink
    return output


def flowgraphDict2coopgraph(flow_graph, agent_i, Q_norm = 100):
    """
    Convert a residual flow graph (array) associated to agent_i into a cooperation graph
    1. keep only non negative values (corresponding of the right way of the directed graph)
    2. Removes the source and sink and renormalise (with Q_norm) in a float graph (in [0,1])
    rows 0 and (n+1) useless because corresponding to source and sink
    column (n+1) of the sink has to be placed in column agent_i
    """

    output = np.maximum(0, flow_graph).astype(float) #keep only right way edges of the directed graph
    output /= Q_norm #normalise
    output[:,agent_i+1] = output[:,-1] #connects the sink to the vertex of agent_i
    output = output[1:-1,1:-1] #removes the vertices of source and sink
    return output


def extract_maxflow(flow_graph):
    """
    extracts from the flow graph the residual graph correponding to the maximum flow
    source (resp. sink) is the first (resp. last) vertex
    INPUT: flowgraph is a matrix (n,n)
    OUPUT: residual graph (matrix (n,n))
    Using Ford-Fulkerson algo with csgraph and csgraph.maximum_flow from scipy.sparse
    """
    (n,_) = np.shape(flow_graph)
    graph = csr_matrix(flow_graph)
    res_graph = maximum_flow(graph, 0, n-1).residual
    return res_graph.toarray()


def subgraph_coop_flowMax(coopgraph, agent_i, source_max=1.0, Q_norm=100):
    """
    Extracts from the max cooperation graph (NxN array of n agents with coop degrees in [0,1])
    a subgraph for agent_i who is able to provide in the graph a total of cooperation equal to source_max
    Uses Ford-Fulkerson algorithm to find the maximum flow
    """
    # converts into a flow graph with source and sink connected to vertex of agent_i
    flow_graph = coopgraph2flowgraph(coopgraph, agent_i, source_max, Q_norm)

    # extracts a max flow graph from the above flow graph
    res_graph = extract_maxflow(flow_graph)

    # converts into a cooperation graph
    sub_graph_coop = flowgraph2coopgraph(res_graph, agent_i, Q_norm)

    return sub_graph_coop



def subgraph_coop_flowMax_minCost(coopgraph, agent_i, source_max = 1.0, Q_norm = 100, cost_matrix = None):
    """
    Extracts from the max cooperation graph (NxN array of n agents with coop degrees in [0,1])
    a subgraph for agent_i who is able to provide in the graph a total of cooperation equal to source_max
    Uses a variant of Ford-Fulkerson algorithm to find the maximum flow with minimal cost (maximal benefice=cooperation)
    """
    (n,_) = np.shape(coopgraph)
    if cost_matrix is None:
        #cost_matrix = -1.0*coopgraph
        cost_matrix = -1.0*np.ones([n,n])
    flow_dict = coopgraph2flowDict(coopgraph, agent_i, source_max = source_max, Q_norm = Q_norm, cost_matrix = cost_matrix)
    mincostFlow = nx.max_flow_min_cost(flow_dict, 0, n+1)
    flow_numpy = flowDict2array(mincostFlow)
    sub_graph_coop = flowgraph2coopgraph(flow_numpy, agent_i)
    return sub_graph_coop