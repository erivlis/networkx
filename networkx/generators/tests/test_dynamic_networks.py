"""Unit tests for the :mod:`networkx.generators.dynamic_networks` module."""
import itertools
import random

import pytest

import networkx as nx

NODES = range(100)
EDGES = list(itertools.combinations(NODES, 2))
STATIC_SCALAR_FIELD_VALUES = {k: random.uniform(5, 25) for k in NODES}
RANDOM_SCALAR_FIELD_VALUES = {
    k: lambda: v + random.uniform(-5.0, 5.0)
    for k, v in STATIC_SCALAR_FIELD_VALUES.items()
}


def test_gradient_network_static_values():
    # arrange
    substrate_network = nx.Graph(EDGES)
    nx.set_node_attributes(substrate_network, STATIC_SCALAR_FIELD_VALUES, "value")

    # act
    actual_gradient_network_1st_run = nx.gradient_network(substrate_network)
    actual_gradient_network_2nd_run = nx.gradient_network(substrate_network)

    # assert
    assert actual_gradient_network_1st_run.number_of_nodes() == len(substrate_network)
    assert actual_gradient_network_2nd_run.number_of_nodes() == len(substrate_network)
    assert nx.utils.graphs_equal(
        actual_gradient_network_1st_run, actual_gradient_network_2nd_run
    )


def test_gradient_network_random_values():
    # arrange
    substrate_network = nx.Graph(EDGES)
    nx.set_node_attributes(substrate_network, RANDOM_SCALAR_FIELD_VALUES, "value")

    # act
    actual_gradient_network_1st_run = nx.gradient_network(substrate_network)
    actual_gradient_network_2nd_run = nx.gradient_network(substrate_network)

    # assert
    assert actual_gradient_network_1st_run.number_of_nodes() == len(substrate_network)
    assert actual_gradient_network_2nd_run.number_of_nodes() == len(substrate_network)
    assert not nx.utils.graphs_equal(
        actual_gradient_network_1st_run, actual_gradient_network_2nd_run
    )


@pytest.mark.parametrize("graph_class", (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
def test_gradient_network_raises_error(graph_class):
    # arrange
    substrate_network = graph_class(EDGES)
    nx.set_node_attributes(substrate_network, STATIC_SCALAR_FIELD_VALUES, "value")

    # act + assert
    with pytest.raises(nx.NetworkXNotImplemented):
        _ = nx.gradient_network(substrate_network)


def test_gradient_network_star_graph():
    """Test star graph gradient flow based on Toroczkai et al. (2008) Fig. 6a.

    In a star graph where the central node has the highest scalar potential,
    all outer nodes flow into the center, creating a maximal congestion funnel.
    """
    G = nx.star_graph(4)
    # Center node 0 with highest potential; leaves 1..4 with lower potentials
    potentials = {0: 10.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    nx.set_node_attributes(G, potentials, "value")

    # Ascending gradient: leaves point to center, center points to itself
    H_asc = nx.gradient_network(G, ascending=True)
    assert sorted(H_asc.edges()) == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    assert H_asc.in_degree(0) == 5
    for leaf in (1, 2, 3, 4):
        assert H_asc.in_degree(leaf) == 0
        assert H_asc[leaf][0]["size"] == 10.0 - potentials[leaf]
    assert H_asc[0][0]["size"] == 0.0

    # Descending gradient: center points to leaf 1 (minimum value), leaves point to themselves
    H_desc = nx.gradient_network(G, ascending=False)
    assert sorted(H_desc.edges()) == [(0, 1), (1, 1), (2, 2), (3, 3), (4, 4)]
    assert H_desc[0][1]["size"] == 9.0
    for node in (1, 2, 3, 4):
        assert H_desc[node][node]["size"] == 0.0


def test_gradient_network_path_graph_with_distance():
    """Test path graph where gradient flows lead to a local potential peak."""
    # Path: 0 - 1 - 2 - 3 - 4 with peak at node 2
    G = nx.path_graph(5)
    potentials = {0: 1.0, 1: 3.0, 2: 6.0, 3: 4.0, 4: 2.0}
    nx.set_node_attributes(G, potentials, "value")

    # Custom edge distances
    distances = {(0, 1): 2.0, (1, 2): 3.0, (2, 3): 1.0, (3, 4): 2.0}
    nx.set_edge_attributes(G, distances, "distance")

    H = nx.gradient_network(G, ascending=True)
    expected_edges = [(0, 1), (1, 2), (2, 2), (3, 2), (4, 3)]
    assert sorted(H.edges()) == sorted(expected_edges)

    assert H[0][1]["size"] == abs(1.0 - 3.0) / 2.0  # 1.0
    assert H[1][2]["size"] == abs(3.0 - 6.0) / 3.0  # 1.0
    assert H[2][2]["size"] == 0.0
    assert H[3][2]["size"] == abs(4.0 - 6.0) / 1.0  # 2.0
    assert H[4][3]["size"] == abs(2.0 - 4.0) / 2.0  # 1.0


def test_gradient_network_callables_as_parameters():
    """Test passing callables directly as scalar_field_value and scalar_field_distance."""
    G = nx.path_graph(3)

    # Node value supplier as callable
    value_func = lambda n: float(n * 10)
    # Edge distance supplier as callable
    dist_func = lambda u, v, d: 2.0

    H = nx.gradient_network(
        G,
        scalar_field_value=value_func,
        scalar_field_distance=dist_func,
        ascending=True,
    )
    # Nodes: 0 (val 0), 1 (val 10), 2 (val 20)
    # 0 -> 1, 1 -> 2, 2 -> 2
    assert sorted(H.edges()) == [(0, 1), (1, 2), (2, 2)]
    assert H[0][1]["size"] == 10.0 / 2.0  # 5.0
    assert H[1][2]["size"] == 10.0 / 2.0  # 5.0
    assert H[2][2]["size"] == 0.0


def test_gradient_network_isolated_nodes_and_empty():
    """Test graphs with isolated nodes and empty graph."""
    G_empty = nx.Graph()
    H_empty = nx.gradient_network(G_empty)
    assert H_empty.number_of_nodes() == 0
    assert H_empty.number_of_edges() == 0

    G_iso = nx.Graph()
    G_iso.add_nodes_from([1, 2, 3])
    H_iso = nx.gradient_network(G_iso)
    assert sorted(H_iso.edges()) == [(1, 1), (2, 2), (3, 3)]
    for n in (1, 2, 3):
        assert H_iso[n][n]["size"] == 0.0


def test_gradient_network_sequence_dict_attributes():
    """Test gradient_network_sequence with time-indexed dictionary attributes."""
    G = nx.star_graph(2)
    potentials = {
        0: {0: 10.0, 1: 0.0},
        1: {0: 1.0, 1: 5.0},
        2: {0: 2.0, 1: 2.0},
    }
    nx.set_node_attributes(G, potentials, "value")

    snapshots = list(nx.gradient_network_sequence(G, times=[0, 1]))
    assert len(snapshots) == 2

    # Snapshot at t = 0
    t0, H0 = snapshots[0]
    assert t0 == 0
    assert H0.graph["time"] == 0
    assert sorted(H0.edges()) == [(0, 0), (1, 0), (2, 0)]

    # Snapshot at t = 1
    t1, H1 = snapshots[1]
    assert t1 == 1
    assert H1.graph["time"] == 1
    assert sorted(H1.edges()) == [(0, 1), (1, 1), (2, 2)]


def test_gradient_network_sequence_callables():
    """Test gradient_network_sequence with time-dependent functions."""
    G = nx.path_graph(3)

    # Node potential as function of (node, t): at t=0, node 2 is max; at t=2, node 0 is max
    value_func = lambda n, t: float((2 - n) * t + n * (2 - t))
    # Edge distance as function of (u, v, t)
    dist_func = lambda u, v, t: 1.0 + t

    snapshots = list(
        nx.gradient_network_sequence(
            G,
            times=[0, 2],
            scalar_field_value=value_func,
            scalar_field_distance=dist_func,
        )
    )
    # At t = 0: values are {0: 0, 1: 2, 2: 4} -> gradient towards node 2
    t0, H0 = snapshots[0]
    assert t0 == 0
    assert sorted(H0.edges()) == [(0, 1), (1, 2), (2, 2)]

    # At t = 2: values are {0: 4, 1: 2, 2: 0} -> gradient towards node 0
    t1, H1 = snapshots[1]
    assert t1 == 2
    assert sorted(H1.edges()) == [(0, 0), (1, 0), (2, 1)]


@pytest.mark.parametrize("graph_class", (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
def test_gradient_network_sequence_raises_error(graph_class):
    """Test gradient_network_sequence raises NetworkXNotImplemented on unsupported graphs."""
    G = graph_class()
    with pytest.raises(nx.NetworkXNotImplemented):
        _ = list(nx.gradient_network_sequence(G, times=[0, 1]))

