#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 18:28:55 2025

@author: jack

sudo apt-get install graphviz graphviz-dev
pip install pygraphviz


"""

#%% plot_atlas.py


import random

import matplotlib.pyplot as plt
import networkx as nx


GraphMatcher = nx.isomorphism.vf2userfunc.GraphMatcher


def atlas6():
    """Return the atlas of all connected graphs with at most 6 nodes"""

    Atlas = nx.graph_atlas_g()[3:209]  # 0, 1, 2 => no edges. 208 is last 6 node graph
    U = nx.Graph()  # graph for union of all graphs in atlas
    for G in Atlas:
        # check if connected
        if nx.number_connected_components(G) == 1:
            # check if isomorphic to a previous graph
            if not GraphMatcher(U, G).subgraph_is_isomorphic():
                U = nx.disjoint_union(U, G)
    return U


G = atlas6()

print(G)
print(nx.number_connected_components(G), "connected components")

plt.figure(1, figsize=(8, 8))
# layout graphs with positions using graphviz neato
pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
# color nodes the same in each connected subgraph
C = (G.subgraph(c) for c in nx.connected_components(G))
for g in C:
    c = [random.random()] * nx.number_of_nodes(g)  # random color...
    nx.draw(g, pos, node_size=40, node_color=c, vmin=0.0, vmax=1.0, with_labels=False)
plt.show()

#%% plot_beam_search.py
"""
===========
Beam Search
===========

Beam search with dynamic beam width.

The progressive widening beam search repeatedly executes a beam search
with increasing beam width until the target node is found.
https://networkx.org/documentation/stable/auto_examples/algorithms/plot_beam_search.html#sphx-glr-auto-examples-algorithms-plot-beam-search-py
"""
import math

import matplotlib.pyplot as plt
import networkx as nx


def progressive_widening_search(G, source, value, condition, initial_width=1):
    """Progressive widening beam search to find a node.

    The progressive widening beam search involves a repeated beam
    search, starting with a small beam width then extending to
    progressively larger beam widths if the target node is not
    found. This implementation simply returns the first node found that
    matches the termination condition.

    `G` is a NetworkX graph.

    `source` is a node in the graph. The search for the node of interest
    begins here and extends only to those nodes in the (weakly)
    connected component of this node.

    `value` is a function that returns a real number indicating how good
    a potential neighbor node is when deciding which neighbor nodes to
    enqueue in the breadth-first search. Only the best nodes within the
    current beam width will be enqueued at each step.

    `condition` is the termination condition for the search. This is a
    function that takes a node as input and return a Boolean indicating
    whether the node is the target. If no node matches the termination
    condition, this function raises :exc:`NodeNotFound`.

    `initial_width` is the starting beam width for the beam search (the
    default is one). If no node matching the `condition` is found with
    this beam width, the beam search is restarted from the `source` node
    with a beam width that is twice as large (so the beam width
    increases exponentially). The search terminates after the beam width
    exceeds the number of nodes in the graph.

    """
    # Check for the special case in which the source node satisfies the
    # termination condition.
    if condition(source):
        return source
    # The largest possible value of `i` in this range yields a width at
    # least the number of nodes in the graph, so the final invocation of
    # `bfs_beam_edges` is equivalent to a plain old breadth-first
    # search. Therefore, all nodes will eventually be visited.
    log_m = math.ceil(math.log2(len(G)))
    for i in range(log_m):
        width = initial_width * pow(2, i)
        # Since we are always starting from the same source node, this
        # search may visit the same nodes many times (depending on the
        # implementation of the `value` function).
        for u, v in nx.bfs_beam_edges(G, source, value, width):
            if condition(v):
                return v
    # At this point, since all nodes have been visited, we know that
    # none of the nodes satisfied the termination condition.
    raise nx.NodeNotFound("no node satisfied the termination condition")


###############################################################################
# Search for a node with high centrality.
# ---------------------------------------
#
# We generate a random graph, compute the centrality of each node, then perform
# the progressive widening search in order to find a node of high centrality.

# Set a seed for random number generation so the example is reproducible
seed = 89

G = nx.gnp_random_graph(100, 0.5, seed=seed)
centrality = nx.eigenvector_centrality(G)
avg_centrality = sum(centrality.values()) / len(G)

def has_high_centrality(v):
    return centrality[v] >= avg_centrality

source = 0
value = centrality.get
condition = has_high_centrality

found_node = progressive_widening_search(G, source, value, condition)
c = centrality[found_node]
print(f"found node {found_node} with centrality {c}")


# Draw graph
pos = nx.spring_layout(G, seed=seed)
options = {
    "node_color": "blue",
    "node_size": 20,
    "edge_color": "grey",
    "linewidths": 0,
    "width": 0.1,
}
nx.draw(G, pos, **options)
# Draw node with high centrality as large and red
nx.draw_networkx_nodes(G, pos, nodelist=[found_node], node_size=100, node_color="r")
plt.show()



#%% plot_betweenness_centrality


"""
======================
Betweenness Centrality
======================

Betweenness centrality measures of positive gene functional associations
using WormNet v.3-GS.

Data from: https://www.inetbio.org/wormnet/downloadnetwork.php
https://networkx.org/documentation/stable/auto_examples/algorithms/plot_betweenness_centrality.html#sphx-glr-auto-examples-algorithms-plot-betweenness-centrality-py
"""

from random import sample
import networkx as nx
import matplotlib.pyplot as plt

# Gold standard data of positive gene functional associations
# from https://www.inetbio.org/wormnet/downloadnetwork.php
G = nx.read_edgelist("WormNet.v3.benchmark.txt")

# remove randomly selected nodes (to make example fast)
num_to_remove = int(len(G) / 1.5)
nodes = sample(list(G.nodes), num_to_remove)
G.remove_nodes_from(nodes)

# remove low-degree nodes
low_degree = [n for n, d in G.degree() if d < 10]
G.remove_nodes_from(low_degree)

# largest connected component
components = nx.connected_components(G)
largest_component = max(components, key=len)
H = G.subgraph(largest_component)

# compute centrality
centrality = nx.betweenness_centrality(H, k=10, endpoints=True)

# compute community structure
lpc = nx.community.label_propagation_communities(H)
community_index = {n: i for i, com in enumerate(lpc) for n in com}

#### draw graph ####
fig, ax = plt.subplots(figsize=(20, 15))
pos = nx.spring_layout(H, k=0.15, seed=4572321)
node_color = [community_index[n] for n in H]
node_size = [v * 20000 for v in centrality.values()]
nx.draw_networkx(
    H,
    pos=pos,
    with_labels=False,
    node_color=node_color,
    node_size=node_size,
    edge_color="grey",
    alpha=0.4,
)

# Title/legend
font = {"color": "k", "fontweight": "bold", "fontsize": 20}
ax.set_title("Gene functional association network (C. elegans)", font)
# Change font color for legend
font["color"] = "r"

ax.text(
    0.80,
    0.10,
    "node color = community structure",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)
ax.text(
    0.80,
    0.06,
    "node size = betweenness centrality",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)

# Resize figure for label readability
ax.margins(0.1, 0.05)
fig.tight_layout()
plt.axis("off")
plt.show()



#%% plot_chess_masters

"""
=============
Chess Masters
=============

An example of the MultiDiGraph class.

The function `chess_pgn_graph` reads a collection of chess matches stored in
the specified PGN file (PGN ="Portable Game Notation").  Here the (compressed)
default file::

    chess_masters_WCC.pgn.bz2

contains all 685 World Chess Championship matches from 1886--1985.
(data from http://chessproblem.my-free-games.com/chess/games/Download-PGN.php)

The `chess_pgn_graph()` function returns a `MultiDiGraph` with multiple edges.
Each node is the last name of a chess master. Each edge is directed from white
to black and contains selected game info.

The key statement in `chess_pgn_graph` below is::

    G.add_edge(white, black, game_info)

where `game_info` is a `dict` describing each game.

https://github.com/bjedwards/NetworkX_fork/blob/master/examples/multigraph/chess_masters_WCC.pgn.bz2
"""

import matplotlib.pyplot as plt
import networkx as nx

# tag names specifying what game info should be
# stored in the dict on each digraph edge
game_details = ["Event", "Date", "Result", "ECO", "Site"]


def chess_pgn_graph(pgn_file="chess_masters_WCC.pgn.bz2"):
    """Read chess games in pgn format in pgn_file.

    Filenames ending in .bz2 will be uncompressed.

    Return the MultiDiGraph of players connected by a chess game.
    Edges contain game data in a dict.

    """
    import bz2

    G = nx.MultiDiGraph()
    game = {}
    with bz2.BZ2File(pgn_file) as datafile:
        lines = [line.decode().rstrip("\r\n") for line in datafile]
    for line in lines:
        if line.startswith("["):
            tag, value = line[1:-1].split(" ", 1)
            game[str(tag)] = value.strip('"')
        else:
            # empty line after tag set indicates
            # we finished reading game info
            if game:
                white = game.pop("White")
                black = game.pop("Black")
                G.add_edge(white, black, **game)
                game = {}
    return G


G = chess_pgn_graph()

print(
    f"Loaded {G.number_of_edges()} chess games between {G.number_of_nodes()} players\n"
)

# identify connected components of the undirected version
H = G.to_undirected()
Gcc = [H.subgraph(c) for c in nx.connected_components(H)]
if len(Gcc) > 1:
    print(f"Note the disconnected component consisting of:\n{Gcc[1].nodes()}")

# find all games with B97 opening (as described in ECO)
openings = {game_info["ECO"] for (white, black, game_info) in G.edges(data=True)}
print(f"\nFrom a total of {len(openings)} different openings,")
print("the following games used the Sicilian opening")
print('with the Najdorff 7...Qb6 "Poisoned Pawn" variation.\n')

for white, black, game_info in G.edges(data=True):
    if game_info["ECO"] == "B97":
        summary = f"{white} vs {black}\n"
        for k, v in game_info.items():
            summary += f"   {k}: {v}\n"
        summary += "\n"
        print(summary)

# make new undirected graph H without multi-edges
H = nx.Graph(G)

# edge width is proportional number of games played
edgewidth = [len(G.get_edge_data(u, v)) for u, v in H.edges()]

# node size is proportional to number of games won
wins = dict.fromkeys(G.nodes(), 0.0)
for u, v, d in G.edges(data=True):
    r = d["Result"].split("-")
    if r[0] == "1":
        wins[u] += 1.0
    elif r[0] == "1/2":
        wins[u] += 0.5
        wins[v] += 0.5
    else:
        wins[v] += 1.0
nodesize = [wins[v] * 50 for v in H]

# Generate layout for visualization
pos = nx.kamada_kawai_layout(H)
# Manual tweaking to limit node label overlap in the visualization
pos["Reshevsky, Samuel H"] += (0.05, -0.10)
pos["Botvinnik, Mikhail M"] += (0.03, -0.06)
pos["Smyslov, Vassily V"] += (0.05, -0.03)

fig, ax = plt.subplots(figsize=(12, 12))
# Visualize graph components
nx.draw_networkx_edges(H, pos, alpha=0.3, width=edgewidth, edge_color="m")
nx.draw_networkx_nodes(H, pos, node_size=nodesize, node_color="#210070", alpha=0.9)
label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
nx.draw_networkx_labels(H, pos, font_size=14, bbox=label_options)

# Title/legend
font = {"fontname": "Helvetica", "color": "k", "fontweight": "bold", "fontsize": 14}
ax.set_title("World Chess Championship Games: 1886 - 1985", font)
# Change font color for legend
font["color"] = "r"

ax.text(
    0.80,
    0.10,
    "edge width = # games played",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)
ax.text(
    0.80,
    0.06,
    "node size = # games won",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)

# Resize figure for label readability
ax.margins(0.1, 0.05)
fig.tight_layout()
plt.axis("off")
plt.show()



#%% plot_circular_tree

"""
=============
Circular Tree
=============

This example needs Graphviz and PyGraphviz.
https://networkx.org/documentation/stable/auto_examples/graphviz_layout/plot_circular_tree.html#sphx-glr-auto-examples-graphviz-layout-plot-circular-tree-py
"""

import matplotlib.pyplot as plt
import networkx as nx

G = nx.balanced_tree(3, 5)
pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)
plt.axis("equal")
plt.show()





#%% plot_knuth_miles

"""
===========
Knuth Miles
===========

`miles_graph()` returns an undirected graph over 128 US cities. The
cities each have location and population data.  The edges are labeled with the
distance between the two cities.

This example is described in Section 1.1 of

    Donald E. Knuth, "The Stanford GraphBase: A Platform for Combinatorial
    Computing", ACM Press, New York, 1993.
    http://www-cs-faculty.stanford.edu/~knuth/sgb.html

The data file can be found at:

- https://github.com/networkx/networkx/blob/main/examples/drawing/knuth_miles.txt.gz
https://networkx.org/documentation/stable/auto_examples/drawing/plot_knuth_miles.html#sphx-glr-auto-examples-drawing-plot-knuth-miles-py
"""

import gzip
import re

# Ignore any warnings related to downloading shpfiles with cartopy
import warnings

warnings.simplefilter("ignore")

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def miles_graph():
    """Return the cites example graph in miles_dat.txt
    from the Stanford GraphBase.
    """
    # open file miles_dat.txt.gz (or miles_dat.txt)

    fh = gzip.open("knuth_miles.txt.gz", "r")

    G = nx.Graph()
    G.position = {}
    G.population = {}

    cities = []
    for line in fh.readlines():
        line = line.decode()
        if line.startswith("*"):  # skip comments
            continue

        numfind = re.compile(r"^\d+")

        if numfind.match(line):  # this line is distances
            dist = line.split()
            for d in dist:
                G.add_edge(city, cities[i], weight=int(d))
                i = i + 1
        else:  # this line is a city, position, population
            i = 1
            (city, coordpop) = line.split("[")
            cities.insert(0, city)
            (coord, pop) = coordpop.split("]")
            (y, x) = coord.split(",")

            G.add_node(city)
            # assign position - Convert string to lat/long
            G.position[city] = (-float(x) / 100, float(y) / 100)
            G.population[city] = float(pop) / 1000
    return G


G = miles_graph()

print("Loaded miles_dat.txt containing 128 cities.")
print(G)

# make new graph of cites, edge if less then 300 miles between them
H = nx.Graph()
for v in G:
    H.add_node(v)
for u, v, d in G.edges(data=True):
    if d["weight"] < 300:
        H.add_edge(u, v)

# draw with matplotlib/pylab
fig = plt.figure(figsize=(8, 6))

# nodes colored by degree sized by population
node_color = [float(H.degree(v)) for v in H]

# Use cartopy to provide a backdrop for the visualization
try:
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader

    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal(), frameon=False)
    ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())
    # Add map of countries & US states as a backdrop
    for shapename in ("admin_1_states_provinces_lakes_shp", "admin_0_countries"):
        shp = shpreader.natural_earth(
            resolution="110m", category="cultural", name=shapename
        )
        ax.add_geometries(
            shpreader.Reader(shp).geometries(),
            ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="k",
        )
    # NOTE: When using cartopy, use matplotlib directly rather than nx.draw
    # to take advantage of the cartopy transforms
    ax.scatter(
        *np.array(list(G.position.values())).T,
        s=[G.population[v] for v in H],
        c=node_color,
        transform=ccrs.PlateCarree(),
        zorder=100,  # Ensure nodes lie on top of edges/state lines
    )
    # Plot edges between the cities
    for edge in H.edges():
        edge_coords = np.array([G.position[v] for v in edge])
        ax.plot(
            edge_coords[:, 0],
            edge_coords[:, 1],
            transform=ccrs.PlateCarree(),
            linewidth=0.75,
            color="k",
        )

except ImportError:
    # If cartopy is unavailable, the backdrop for the plot will be blank;
    # though you should still be able to discern the general shape of the US
    # from graph nodes and edges!
    nx.draw(
        H,
        G.position,
        node_size=[G.population[v] for v in H],
        node_color=node_color,
        with_labels=False,
    )

plt.show()



#%% plot_multipartite_graph
"""
===================
Multipartite Layout
===================
https://networkx.org/documentation/stable/auto_examples/drawing/plot_multipartite_graph.html#sphx-glr-auto-examples-drawing-plot-multipartite-graph-py
"""

import itertools
import matplotlib.pyplot as plt
import networkx as nx

subset_sizes = [5, 5, 4, 3, 2, 4, 4, 3]
subset_color = [
    "gold",
    "violet",
    "violet",
    "violet",
    "violet",
    "limegreen",
    "limegreen",
    "darkorange",
]


def multilayered_graph(*subset_sizes):
    extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
    layers = [range(start, end) for start, end in extents]
    G = nx.Graph()
    for i, layer in enumerate(layers):
        G.add_nodes_from(layer, layer=i)
    for layer1, layer2 in nx.utils.pairwise(layers):
        G.add_edges_from(itertools.product(layer1, layer2))
    return G


G = multilayered_graph(*subset_sizes)
color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
pos = nx.multipartite_layout(G, subset_key="layer")
plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_color=color, with_labels=False)
plt.axis("equal")
plt.show()



#%% plot_lanl_routes

"""
===========
Lanl Routes
===========

Routes to LANL from 186 sites on the Internet.

The data file can be found at:

- https://github.com/networkx/networkx/blob/main/examples/graphviz_layout/lanl_routes.edgelist
https://networkx.org/documentation/stable/auto_examples/graphviz_layout/plot_lanl_routes.html#sphx-glr-auto-examples-graphviz-layout-plot-lanl-routes-py

This example needs Graphviz and PyGraphviz.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def lanl_graph():
    """Return the lanl internet view graph from lanl.edges"""
    try:
        fh = open("lanl_routes.edgelist")
    except OSError:
        print("lanl.edges not found")
        raise

    G = nx.Graph()

    time = {}
    time[0] = 0  # assign 0 to center node
    for line in fh.readlines():
        (head, tail, rtt) = line.split()
        G.add_edge(int(head), int(tail))
        time[int(head)] = float(rtt)

    # get largest component and assign ping times to G0time dictionary
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)[0]
    G0 = G.subgraph(Gcc)
    G0.rtt = {}
    for n in G0:
        G0.rtt[n] = time[n]

    return G0


G = lanl_graph()

print(G)
print(nx.number_connected_components(G), "connected components")

plt.figure(figsize=(8, 8))
# use graphviz to find radial layout
pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", root=0)
# draw nodes, coloring by rtt ping time
options = {"with_labels": False, "alpha": 0.5, "node_size": 15}
nx.draw(G, pos, node_color=[G.rtt[v] for v in G], **options)
# adjust the plot limits
xmax = 1.02 * max(xx for xx, yy in pos.values())
ymax = 1.02 * max(yy for xx, yy in pos.values())
max_frame = np.max([xmax, ymax])
plt.xlim(0, max_frame)
plt.ylim(0, max_frame)
plt.show()



#%% plot_rainbow_coloring
"""
================
Rainbow Coloring
================

Generate a complete graph with 13 nodes in a circular layout with the
edges colored by node distance. The node distance is given by the minimum
number of nodes traversed along an arc between any two nodes on the circle.

Such graphs are the subject of Ringel's conjecture, which states: any complete
graph with ``2n + 1`` nodes can be tiled by any tree with ``n + 1`` nodes
(i.e. copies of the tree can be placed over the complete graph such that each
edge in the complete graph is covered exactly once). The edge coloring is
helpful in determining how to place the tree copies.

References
----------
https://www.quantamagazine.org/mathematicians-prove-ringels-graph-theory-conjecture-20200219/
https://networkx.org/documentation/stable/auto_examples/drawing/plot_rainbow_coloring.html#sphx-glr-auto-examples-drawing-plot-rainbow-coloring-py
"""
import matplotlib.pyplot as plt
import networkx as nx

# A rainbow color mapping using matplotlib's tableau colors
node_dist_to_color = {
    1: "tab:red",
    2: "tab:orange",
    3: "tab:olive",
    4: "tab:green",
    5: "tab:blue",
    6: "tab:purple",
}

# Create a complete graph with an odd number of nodes
nnodes = 13
G = nx.complete_graph(nnodes)

# A graph with (2n + 1) nodes requires n colors for the edges
n = (nnodes - 1) // 2
ndist_iter = list(range(1, n + 1))

# Take advantage of circular symmetry in determining node distances
ndist_iter += ndist_iter[::-1]


def cycle(nlist, n):
    return nlist[-n:] + nlist[:-n]


# Rotate nodes around the circle and assign colors for each edge based on
# node distance
nodes = list(G.nodes())
for i, nd in enumerate(ndist_iter):
    for u, v in zip(nodes, cycle(nodes, i + 1)):
        G[u][v]["color"] = node_dist_to_color[nd]

pos = nx.circular_layout(G)
# Create a figure with 1:1 aspect ratio to preserve the circle.
fig, ax = plt.subplots(figsize=(8, 8))
node_opts = {"node_size": 500, "node_color": "w", "edgecolors": "k", "linewidths": 2.0}
nx.draw_networkx_nodes(G, pos, **node_opts)
nx.draw_networkx_labels(G, pos, font_size=14)
# Extract color from edge data
edge_colors = [edgedata["color"] for _, _, edgedata in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, width=2.0, edge_color=edge_colors)

ax.set_axis_off()
fig.tight_layout()
plt.show()


#%% plot_random_geometric_graph

"""
======================
Random Geometric Graph
======================

Example
https://networkx.org/documentation/stable/auto_examples/drawing/plot_random_geometric_graph.html#sphx-glr-auto-examples-drawing-plot-random-geometric-graph-py

"""

import matplotlib.pyplot as plt
import networkx as nx

# Use seed when creating the graph for reproducibility
G = nx.random_geometric_graph(200, 0.125, seed=896803)
# position is stored as node attribute data for random_geometric_graph
pos = nx.get_node_attributes(G, "pos")

# find node near center (0.5,0.5)
dmin = 1
ncenter = 0
for n in pos:
    x, y = pos[n]
    d = (x - 0.5) ** 2 + (y - 0.5) ** 2
    if d < dmin:
        ncenter = n
        dmin = d

# color by path length from node near center
p = dict(nx.single_source_shortest_path_length(G, ncenter))

plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G, pos, alpha=0.4)
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=list(p.keys()),
    node_size=80,
    node_color=list(p.values()),
    cmap=plt.cm.Reds_r,
)

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.axis("off")
plt.show()


#%% plot_triad_types
"""
======
Triads
======
According to the paper by Snijders, T. (2012). “Transitivity and triads.”
University of Oxford, there are 16 Triad Types possible. This plot shows
the 16 Triad Types that can be identified within directed networks.
Triadic relationships are especially useful when analysing Social Networks.
The first three digits refer to the number of mutual, asymmetric and null
dyads (bidirectional, unidirection and nonedges) and the letter gives
the Orientation as Up (U), Down (D) , Cyclical (C) or Transitive (T).
"""

import networkx as nx
import matplotlib.pyplot as plt

p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
triads = {
    "003": [],
    "012": [(1, 2)],
    "102": [(1, 2), (2, 1)],
    "021D": [(3, 1), (3, 2)],
    "021U": [(1, 3), (2, 3)],
    "021C": [(1, 3), (3, 2)],
    "111D": [(1, 2), (2, 1), (3, 1)],
    "111U": [(1, 2), (2, 1), (1, 3)],
    "030T": [(1, 2), (3, 2), (1, 3)],
    "030C": [(1, 3), (3, 2), (2, 1)],
    "201": [(1, 2), (2, 1), (3, 1), (1, 3)],
    "120D": [(1, 2), (2, 1), (3, 1), (3, 2)],
    "120U": [(1, 2), (2, 1), (1, 3), (2, 3)],
    "120C": [(1, 2), (2, 1), (1, 3), (3, 2)],
    "210": [(1, 2), (2, 1), (1, 3), (3, 2), (2, 3)],
    "300": [(1, 2), (2, 1), (2, 3), (3, 2), (1, 3), (3, 1)],
}

for (title, triad), ax in zip(triads.items(), axes.flatten()):
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3])
    G.add_edges_from(triad)
    nx.draw_networkx(
        G,
        ax=ax,
        with_labels=False,
        node_color=["green"],
        node_size=200,
        arrowsize=20,
        width=2,
        pos=nx.planar_layout(G),
    )
    ax.set_xlim(val * 1.2 for val in ax.get_xlim())
    ax.set_ylim(val * 1.2 for val in ax.get_ylim())
    ax.text(
        0,
        0,
        title,
        fontsize=15,
        fontweight="extra bold",
        horizontalalignment="center",
        bbox={"boxstyle": "round, pad=0.3", "fc": "none"},
    )
    ax.axis('off')
fig.tight_layout()
plt.show()


#%% plot_tsp
"""
==========================
Traveling Salesman Problem
==========================

This is an example of a drawing solution of the traveling salesman problem

The function is used to produce the solution is christofides,
where given a set of nodes, it calculates the route of the nodes
that the traveler has to follow in order to minimize the total cost.
"""

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.approximation as nx_app
import math

G = nx.random_geometric_graph(20, radius=0.4, seed=3)
pos = nx.get_node_attributes(G, "pos")

# Depot should be at (0,0)
pos[0] = (0.5, 0.5)

H = G.copy()


# Calculating the distances between the nodes as edge's weight.
for i in range(len(pos)):
    for j in range(i + 1, len(pos)):
        dist = math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])
        dist = dist
        G.add_edge(i, j, weight=dist)

cycle = nx_app.christofides(G, weight="weight")
edge_list = list(nx.utils.pairwise(cycle))

# Draw closest edges on each node only
nx.draw_networkx_edges(H, pos, edge_color="blue", width=0.5)

# Draw the route
nx.draw_networkx(
    G,
    pos,
    with_labels=True,
    edgelist=edge_list,
    edge_color="red",
    node_size=200,
    width=3,
)

print("The route of the traveller is:", cycle)
plt.grid('off')

#% plot_words

"""
==================
Words/Ladder Graph
==================

Generate  an undirected graph over the 5757 5-letter words in the datafile
`words_dat.txt.gz`.  Two words are connected by an edge if they differ in one
letter, resulting in 14,135 edges. This example is described in Section 1.1 of

    Donald E. Knuth, "The Stanford GraphBase: A Platform for Combinatorial
    Computing", ACM Press, New York, 1993.
    http://www-cs-faculty.stanford.edu/~knuth/sgb.html

The data file can be found at:

- https://github.com/networkx/networkx/blob/main/examples/graph/words_dat.txt.gz
"""

import gzip
from string import ascii_lowercase as lowercase

import matplotlib.pyplot as plt
import networkx as nx

p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

def generate_graph(words):
    G = nx.Graph(name="words")
    lookup = {c: lowercase.index(c) for c in lowercase}

    def edit_distance_one(word):
        for i in range(len(word)):
            left, c, right = word[0:i], word[i], word[i + 1 :]
            j = lookup[c]  # lowercase.index(c)
            for cc in lowercase[j + 1 :]:
                yield left + cc + right

    candgen = (
        (word, cand)
        for word in sorted(words)
        for cand in edit_distance_one(word)
        if cand in words
    )
    G.add_nodes_from(words)
    for word, cand in candgen:
        G.add_edge(word, cand)
    return G


def words_graph():
    """Return the words example graph from the Stanford GraphBase"""
    fh = gzip.open("words_dat.txt.gz", "r")
    words = set()
    for line in fh.readlines():
        line = line.decode()
        if line.startswith("*"):
            continue
        w = str(line[0:5])
        words.add(w)
    return generate_graph(words)


G = words_graph()
print("Loaded words_dat.txt containing 5757 five-letter English words.")
print("Two words are connected if they differ in one letter.")
print(G)
print(f"{nx.number_connected_components(G)} connected components")

for source, target in [("chaos", "order"), ("nodes", "graph"), ("pound", "marks")]:
    print(f"Shortest path between {source} and {target} is")
    try:
        shortest_path = nx.shortest_path(G, source, target)
        for n in shortest_path:
            print(n)
    except nx.NetworkXNoPath:
        print("None")


# draw a subset of the graph
boundary = list(nx.node_boundary(G, shortest_path))
G.add_nodes_from(shortest_path, color="red")
G.add_nodes_from(boundary, color="blue")
H = G.subgraph(shortest_path + boundary)
colors = nx.get_node_attributes(H, "color")
options = {"node_size": 1500, "alpha": 0.3, "node_color": colors.values()}
pos = nx.kamada_kawai_layout(H)
nx.draw(H, pos, **options)
nx.draw_networkx_labels(H, pos, font_weight="bold")
plt.show()











