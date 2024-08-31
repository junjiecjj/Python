#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:21:53 2024

@author: jack
"""


#%% >>>>>>>>>>>>>>>> Directed Acyclic Graph
import igraph as ig
import matplotlib.pyplot as plt
import random



random.seed(0)


g = ig.Graph.Erdos_Renyi(n=15, m=30, directed=False, loops=False)

g.to_directed(mode="acyclic")

ig.summary(g)


fig, ax = plt.subplots()
ig.plot(
    g,
    target=ax,
    layout="sugiyama",
    vertex_size=15,
    vertex_color="grey",
    edge_color="#222",
    edge_width=1,
)
plt.show()




#%% >>>>>>>>>>>>>>>> Connected Components
import igraph as ig
import matplotlib.pyplot as plt
import random

random.seed(0)
g = ig.Graph.GRG(50, 0.15)

components = g.connected_components(mode='weak')
fig, ax = plt.subplots()
ig.plot(
    components,
    target=ax,
    palette=ig.RainbowPalette(),
    vertex_size=7,
    vertex_color=list(map(int, ig.rescale(components.membership, (0, 200), clamp=True))),
    edge_width=0.7
)
plt.show()


#%% >>>>>>>>>>>>>>>> Configuration Instance
import igraph as ig
import matplotlib.pyplot as plt
import random

ig.config["plotting.backend"] = "matplotlib"
ig.config["plotting.layout"] = "fruchterman_reingold"
ig.config["plotting.palette"] = "rainbow"

ig.config.save()
random.seed(1)
g = ig.Graph.Barabasi(n=100, m=1)

betweenness = g.betweenness()
colors = [int(i * 200 / max(betweenness)) for i in betweenness]

ig.plot(g, vertex_color=colors, vertex_size=15, edge_width=0.3)
plt.show()



#%% >>>>>>>>>>>>>>>> Articulation Points

import igraph as ig
import matplotlib.pyplot as plt
g = ig.Graph.Formula(
    "0-1-2-0, 3:4:5:6 - 3:4:5:6, 2-3-7-8",
)
articulation_points = g.vs[g.articulation_points()]

fig, ax = plt.subplots()
ig.plot(
    g,
    target=ax,
    vertex_size=30,
    vertex_color="lightblue",
    vertex_label=range(g.vcount()),
    vertex_frame_color = ["red" if v in articulation_points else "black" for v in g.vs],
    vertex_frame_width = [3 if v in articulation_points else 1 for v in g.vs],
    edge_width=0.8,
    edge_color='gray'
)
plt.show()


#%% >>>>>>>>>>>>>>>> Maximum Flow

import igraph as ig
import matplotlib.pyplot as plt


g = ig.Graph(
    6,
    [(3, 2), (3, 4), (2, 1), (4,1), (4, 5), (1, 0), (5, 0)],
    directed=True
)
g.es["capacity"] = [7, 8, 1, 2, 3, 4, 5]

flow = g.maxflow(3, 0, capacity=g.es["capacity"])

print("Max flow:", flow.value)
print("Edge assignments:", flow.flow)

# Output:
# Max flow: 6.0
# Edge assignments [1.0, 5.0, 1.0, 2.0, 3.0, 3.0, 3.0]


fig, ax = plt.subplots()
ig.plot(
    g,
    target=ax,
    layout="circle",
    vertex_label=range(g.vcount()),
    vertex_color="lightblue"
)
plt.show()




#%% >>>>>>>>>>>>>>>> Minimum Spanning Trees

import random
import igraph as ig
import matplotlib.pyplot as plt

random.seed(0)
g = ig.Graph.Lattice([5, 5], circular=False)
g.es["weight"] = [random.randint(1, 20) for _ in g.es]


mst_edges = g.spanning_tree(weights=g.es["weight"], return_tree=False)


print("Minimum edge weight sum:", sum(g.es[mst_edges]["weight"]))

# Minimum edge weight sum: 136

g.es["color"] = "lightgray"
g.es[mst_edges]["color"] = "midnightblue"
g.es["width"] = 1.0
g.es[mst_edges]["width"] = 3.0

fig, ax = plt.subplots()
ig.plot(
    g,
    target=ax,
    layout="grid",
    vertex_color="lightblue",
    edge_width=g.es["width"],
    edge_label=g.es["weight"],
    edge_background="white",
)
plt.show()



#%% >>>>>>>>>>>>>>>> Spanning Trees

import igraph as ig
import matplotlib.pyplot as plt
import random
# First we create a two-dimensional, 6 by 6 lattice graph:

g = ig.Graph.Lattice([6, 6], circular=False)
# We can compute the 2D layout of the graph:

layout = g.layout("grid")
# To spice things up a little, we rearrange the vertex ids and compute a new layout. While not terribly useful in this context, it does make for a more interesting-looking spanning tree ;-)

random.seed(0)
permutation = list(range(g.vcount()))
random.shuffle(permutation)
g = g.permute_vertices(permutation)
new_layout = g.layout("grid")
for i in range(36):
    new_layout[permutation[i]] = layout[i]
layout = new_layout
# We can now generate a spanning tree:

spanning_tree = g.spanning_tree(weights=None, return_tree=False)
# Finally, we can plot the graph with a highlight color for the spanning tree. We follow the usual recipe: first we set a few aesthetic options and then we leverage igraph.plot() and matplotlib for the heavy lifting:

g.es["color"] = "lightgray"
g.es[spanning_tree]["color"] = "midnightblue"
g.es["width"] = 0.5
g.es[spanning_tree]["width"] = 3.0

fig, ax = plt.subplots()
ig.plot(
    g,
    target=ax,
    layout=layout,
    vertex_color="lightblue",
    edge_width=g.es["width"]
)
plt.show()



#%% >>>>>>>>>>>>>>>> Complement


import igraph as ig
import matplotlib.pyplot as plt
import random

random.seed(0)
g1 = ig.Graph.Erdos_Renyi(n=10, p=0.5)


g2 = g1.complementer(loops=False)

g_full = g1 | g2

g_empty = g_full.complementer(loops=False)

fig, axs = plt.subplots(2, 2)
ig.plot(
    g1,
    target=axs[0, 0],
    layout="circle",
    vertex_color="black",
)
axs[0, 0].set_title('Original graph')
ig.plot(
    g2,
    target=axs[0, 1],
    layout="circle",
    vertex_color="black",
)
axs[0, 1].set_title('Complement graph')

ig.plot(
    g_full,
    target=axs[1, 0],
    layout="circle",
    vertex_color="black",
)
axs[1, 0].set_title('Union graph')
ig.plot(
    g_empty,
    target=axs[1, 1],
    layout="circle",
    vertex_color="black",
)
axs[1, 1].set_title('Complement of union graph')
plt.show()


#%% >>>>>>>>>>>>>>>> Maximum Bipartite Matching


import igraph as ig
import matplotlib.pyplot as plt


g = ig.Graph.Bipartite(
    [0, 0, 0, 0, 0, 1, 1, 1, 1],
    [(0, 5), (1, 6), (1, 7), (2, 5), (2, 8), (3, 6), (4, 5), (4, 6)]
)

assert g.is_bipartite()

matching = g.maximum_bipartite_matching()

matching_size = 0
print("Matching is:")
for i in range(5):
    print(f"{i} - {matching.match_of(i)}")
    if matching.is_matched(i):
        matching_size += 1
print("Size of maximum matching is:", matching_size)
# Matching is:
# 0 - 5
# 1 - 7
# 2 - 8
# 3 - 6
# 4 - None
# Size of maximum matching is: 4

fig, ax = plt.subplots(figsize=(7, 3))
ig.plot(
    g,
    target=ax,
    layout=g.layout_bipartite(),
    vertex_size=30,
    vertex_label=range(g.vcount()),
    vertex_color="lightblue",
    edge_width=[3 if e.target == matching.match_of(e.source) else 1.0 for e in g.es],
    edge_color=["red" if e.target == matching.match_of(e.source) else "black" for e in g.es]
)



#%% >>>>>>>>>>>>>>>> Topological sorting

import igraph as ig
import matplotlib.pyplot as plt

g = ig.Graph(
    edges=[(0, 1), (0, 2), (1, 3), (2, 4), (4, 3), (3, 5), (4, 5)],
    directed=True,
)

assert g.is_dag
results = g.topological_sorting(mode='out')
print('Topological sort of g (out):', *results)

results = g.topological_sorting(mode='in')
print('Topological sort of g (in):', *results)
# Topological sort of g (in): 5 3 1 4 2 0

for i in range(g.vcount()):
    print('degree of {}: {}'.format(i, g.vs[i].indegree()))

# %
# Finally, we can plot the graph to make the situation a little clearer.
# Just to change things up a bit, we use the matplotlib visualization mode
# inspired by `xkcd <https://xkcd.com/>_:
with plt.xkcd():
    fig, ax = plt.subplots(figsize=(5, 5))
    ig.plot(
        g,
        target=ax,
        layout='kk',
        vertex_size=25,
        edge_width=4,
        vertex_label=range(g.vcount()),
        vertex_color="white",
    )



#%% >>>>>>>>>>>>>>>> Simplify

import igraph as ig
import matplotlib.pyplot as plt

g1 = ig.Graph([
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 0),
    (0, 0),
    (1, 4),
    (1, 4),
    (0, 2),
    (2, 4),
    (2, 4),
    (2, 4),
    (3, 3)],
)

g2 = g1.copy()
g2.simplify()

visual_style = {
    "vertex_color": "lightblue",
    "vertex_size": 20,
    "vertex_label": [0, 1, 2, 3, 4],
}

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
ig.plot(
    g1,
    layout="circle",
    target=axs[0],
    **visual_style,
)
ig.plot(
    g2,
    layout="circle",
    target=axs[1],
    **visual_style,
)
axs[0].set_title('Multigraph...')
axs[1].set_title('...simplified')
# Draw rectangles around axes
axs[0].add_patch(plt.Rectangle(
    (0, 0), 1, 1, fc='none', ec='k', lw=4, transform=axs[0].transAxes,
    ))
axs[1].add_patch(plt.Rectangle(
    (0, 0), 1, 1, fc='none', ec='k', lw=4, transform=axs[1].transAxes,
    ))
plt.show()



#%% >>>>>>>>>>>>>>>> Communities

import igraph as ig
import matplotlib.pyplot as plt

g = ig.Graph.Famous("Zachary")

communities = g.community_edge_betweenness()
communities = communities.as_clustering()

num_communities = len(communities)
palette = ig.RainbowPalette(n=num_communities)
for i, community in enumerate(communities):
    g.vs[community]["color"] = i
    community_edges = g.es.select(_within=community)
    community_edges["color"] = i

fig, ax = plt.subplots()
ig.plot(
    communities,
    palette=palette,
    edge_width=1,
    target=ax,
    vertex_size=20,
)

# Create a custom color legend
legend_handles = []
for i in range(num_communities):
    handle = ax.scatter(
        [], [],
        s=100,
        facecolor=palette.get(i),
        edgecolor="k",
        label=i,
    )
    legend_handles.append(handle)
ax.legend(
    handles=legend_handles,
    title='Community:',
    bbox_to_anchor=(0, 1.0),
    bbox_transform=ax.transAxes,
)
plt.show()



#%% >>>>>>>>>>>>>>>> Erdős-Rényi Graph

import igraph as ig
import matplotlib.pyplot as plt
import random

random.seed(0)

g1 = ig.Graph.Erdos_Renyi(n=15, p=0.2, directed=False, loops=False)
g2 = ig.Graph.Erdos_Renyi(n=15, p=0.2, directed=False, loops=False)

g3 = ig.Graph.Erdos_Renyi(n=20, m=35, directed=False, loops=False)
g4 = ig.Graph.Erdos_Renyi(n=20, m=35, directed=False, loops=False)

ig.summary(g1)
ig.summary(g2)
ig.summary(g3)
ig.summary(g4)

# IGRAPH U--- 15 18 --
# IGRAPH U--- 15 21 --
# IGRAPH U--- 20 35 --
# IGRAPH U--- 20 35 --
# IGRAPH U--- 15 23 --
# IGRAPH U--- 15 28 --
# IGRAPH U--- 20 35 --
# IGRAPH U--- 20 35 --

fig, axs = plt.subplots(2, 2)
# Probability
ig.plot(
    g1,
    target=axs[0, 0],
    layout="circle",
    vertex_color="lightblue"
)
ig.plot(
    g2,
    target=axs[0, 1],
    layout="circle",
    vertex_color="lightblue"
)
axs[0, 0].set_ylabel('Probability')
# N edges
ig.plot(
    g3,
    target=axs[1, 0],
    layout="circle",
    vertex_color="lightblue",
    vertex_size=15
)
ig.plot(
    g4,
    target=axs[1, 1],
    layout="circle",
    vertex_color="lightblue",
    vertex_size=15
)
axs[1, 0].set_ylabel('N. edges')
plt.show()





#%% >>>>>>>>>>>>>>>> Visual styling

import igraph as ig
import matplotlib.pyplot as plt
import random

visual_style = {
    "edge_width": 0.3,
    "vertex_size": 15,
    "palette": "heat",
    "layout": "fruchterman_reingold"
}

random.seed(1)
gs = [ig.Graph.Barabasi(n=30, m=1) for i in range(4)]

betweenness = [g.betweenness() for g in gs]
colors = [[int(i * 255 / max(btw)) for i in btw] for btw in betweenness]

fig, axs = plt.subplots(2, 2)
axs = axs.ravel()
for g, color, ax in zip(gs, colors, axs):
    ig.plot(g, target=ax, vertex_color=color, **visual_style)
plt.show()

g = ig.Graph.Barabasi(n=30, m=1)
fig, ax = plt.subplots()
ig.plot(g, target=ax)
artist = ax.get_children()[0]
# Option 1:
artist.set(vertex_color="blue")
# Option 2:
artist.set_vertex_color("blue")
plt.show()

g = ig.Graph(n=5)
g.add_edge(2, 3)
g.add_edge(0, 0)
g.add_edge(1, 1)
fig, ax = plt.subplots()
ig.plot(
    g,
    target=ax,
    vertex_size=20,
    edge_loop_size=[
        0,  # ignored, the first edge is not a loop
        30,  # loop for vertex 0
        80,  # loop for vertex 1
    ],
)
plt.show()

#%% >>>>>>>>>>>>>>>> Cliques

import igraph as ig
import matplotlib.pyplot as plt

g = ig.Graph.Famous('Zachary')

cliques = g.cliques(4, 4)

fig, axs = plt.subplots(3, 4)
axs = axs.ravel()
for clique, ax in zip(cliques, axs):
    ig.plot(
        ig.VertexCover(g, [clique]),
        mark_groups=True, palette=ig.RainbowPalette(),
        vertex_size=5,
        edge_width=0.5,
        target=ax,
    )
plt.axis('off')
plt.show()

fig, axs = plt.subplots(3, 4)
axs = axs.ravel()
for clique, ax in zip(cliques, axs):
    # Color vertices yellow/red based on whether they are in this clique
    g.vs['color'] = 'yellow'
    g.vs[clique]['color'] = 'red'

    # Color edges black/red based on whether they are in this clique
    clique_edges = g.es.select(_within=clique)
    g.es['color'] = 'black'
    clique_edges['color'] = 'red'
    # also increase thickness of clique edges
    g.es['width'] = 0.3
    clique_edges['width'] = 1

    ig.plot(
        ig.VertexCover(g, [clique]),
        mark_groups=True,
        palette=ig.RainbowPalette(),
        vertex_size=5,
        target=ax,
    )
plt.axis('off')
plt.show()



#%% >>>>>>>>>>>>>>>> Maximum Bipartite Matching by Maximum Flow

import igraph as ig
import matplotlib.pyplot as plt

g = ig.Graph(
    9,
    [(0, 4), (0, 5), (1, 4), (1, 6), (1, 7), (2, 5), (2, 7), (2, 8), (3, 6), (3, 7)],
    directed=True
)

g.vs[range(4)]["type"] = True
g.vs[range(4, 9)]["type"] = False

g.add_vertices(2)
g.add_edges([(9, 0), (9, 1), (9, 2), (9, 3)])  # connect source to one side
g.add_edges([(4, 10), (5, 10), (6, 10), (7, 10), (8, 10)])  # ... and sinks to the other

flow = g.maxflow(9, 10)
print("Size of maximum matching (maxflow) is:", flow.value)
# Size of maximum matching (maxflow) is: 4.0

# delete the source and sink, which are unneeded for this function.
g2 = g.copy()
g2.delete_vertices([9, 10])
matching = g2.maximum_bipartite_matching()
matching_size = sum(1 for i in range(4) if matching.is_matched(i))
print("Size of maximum matching (maximum_bipartite_matching) is:", matching_size)
# Size of maximum matching (maximum_bipartite_matching) is: 4

layout = g.layout_bipartite()
layout[9] = (2, -1)
layout[10] = (2, 2)

fig, ax = plt.subplots()
ig.plot(
    g,
    target=ax,
    layout=layout,
    vertex_size=30,
    vertex_label=range(g.vcount()),
    vertex_color=["lightblue" if i < 9 else "orange" for i in range(11)],
    edge_width=[1.0 + flow.flow[i] for i in range(g.ecount())]
)
plt.show()



#%% >>>>>>>>>>>>>>>> Shortest Paths

import igraph as ig
import matplotlib.pyplot as plt

g = ig.Graph(
    6,
    [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]
)
results = g.get_shortest_paths(1, to=4, output="vpath")

# results = [[1, 0, 2, 4]]

if len(results[0]) > 0:
    # The distance is the number of vertices in the shortest path minus one.
    print("Shortest distance is: ", len(results[0])-1)
else:
    print("End node could not be reached!")
# Shortest distance is:  3

g.es["weight"] = [2, 1, 5, 4, 7, 3, 2]

results = g.get_shortest_paths(0, to=5, weights=g.es["weight"], output="epath")

# results = [[1, 3, 5]]

if len(results[0]) > 0:
    # Add up the weights across all edges on the shortest path
    distance = 0
    for e in results[0]:
        distance += g.es[e]["weight"]
    print("Shortest weighted distance is: ", distance)
else:
    print("End node could not be reached!")
# Shortest weighted distance is:  8


g.es['width'] = 0.5
g.es[results[0]]['width'] = 2.5

fig, ax = plt.subplots()
ig.plot(
    g,
    target=ax,
    layout='circle',
    vertex_color='steelblue',
    vertex_label=range(g.vcount()),
    edge_width=g.es['width'],
    edge_label=g.es["weight"],
    edge_color='#666',
    edge_align_label=True,
    edge_background='white'
)
plt.show()



#%% >>>>>>>>>>>>>>>> Delaunay Triangulation

import numpy as np
from scipy.spatial import Delaunay
import igraph as ig
import matplotlib.pyplot as plt

np.random.seed(0)
x, y = np.random.rand(2, 30)
g = ig.Graph(30)
g.vs['x'] = x
g.vs['y'] = y

layout = g.layout_auto()

delaunay = Delaunay(layout.coords)

for tri in delaunay.simplices:
    g.add_edges([
        (tri[0], tri[1]),
        (tri[1], tri[2]),
        (tri[0], tri[2]),
    ])

g.simplify()

fig, ax = plt.subplots()
ig.plot(
    g,
    layout=layout,
    target=ax,
    vertex_size=4,
    vertex_color="lightblue",
    edge_width=0.8
)
plt.show()

fig, ax = plt.subplots()
for tri in delaunay.simplices:
    # get the points of the triangle
    tri_points = [delaunay.points[tri[i]] for i in range(3)]

    # calculate the vertical center of the triangle
    center = (tri_points[0][1] + tri_points[1][1] + tri_points[2][1]) / 3

    # draw triangle onto axes
    poly = plt.Polygon(tri_points, color=palette.get(int(center * 100)))
    ax.add_patch(poly)

ig.plot(
    g,
    layout=layout,
    target=ax,
    vertex_size=0,
    edge_width=0.2,
    edge_color="white",
)
ax.set(xlim=(0, 1), ylim=(0, 1))
plt.show()



#%% >>>>>>>>>>>>>>>> Bridges

import igraph as ig
import matplotlib.pyplot as plt

g = ig.Graph(14, [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3), (3, 4),
        (4, 5), (5, 6), (6, 4), (6, 7), (7, 8), (7, 9), (9, 10), (10 ,11),
        (11 ,7), (7, 10), (8, 9), (8, 10), (5, 12), (12, 13)])

bridges = g.bridges()

g.es["color"] = "gray"
g.es[bridges]["color"] = "red"
g.es["width"] = 0.8
g.es[bridges]["width"] = 1.2

fig, ax = plt.subplots()
ig.plot(
    g,
    target=ax,
    vertex_size=30,
    vertex_color="lightblue",
    vertex_label=range(g.vcount())
)
plt.show()

g = ig.Graph(14, [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3), (3, 4),
        (4, 5), (5, 6), (6, 4), (6, 7), (7, 8), (7, 9), (9, 10), (10 ,11),
        (11 ,7), (7, 10), (8, 9), (8, 10), (5, 12), (12, 13)])

bridges = g.bridges()
g.es["color"] = "gray"
g.es[bridges]["color"] = "red"
g.es["width"] = 0.8
g.es[bridges]["width"] = 1.2
g.es["label"] = ""
g.es[bridges]["label"] = "x"


fig, ax = plt.subplots()
ig.plot(
    g,
    target=ax,
    vertex_size=30,
    vertex_color="lightblue",
    vertex_label=range(g.vcount()),
    edge_background="#FFF0",    # transparent background color
    edge_align_label=True,      # make sure labels are aligned with the edge
    edge_label=g.es["label"],
    edge_label_color="red"
)
plt.show()

#%% >>>>>>>>>>>>>>>> Isomorphism

import igraph as ig
import matplotlib.pyplot as plt

g1 = ig.Graph([(0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
g2 = ig.Graph([(4, 2), (4, 3), (4, 0), (2, 3), (2, 1), (3, 1), (3, 0), (1, 0)])
g3 = ig.Graph([(4, 1), (4, 3), (4, 0), (2, 3), (2, 1), (3, 1), (3, 0), (1, 0)])

print("Are the graphs g1 and g2 isomorphic?")
print(g1.isomorphic(g2))
print("Are the graphs g1 and g3 isomorphic?")
print(g1.isomorphic(g3))
print("Are the graphs g2 and g3 isomorphic?")
print(g2.isomorphic(g3))


visual_style = {
    "vertex_color": "lightblue",
    "vertex_label": [0, 1, 2, 3, 4],
    "vertex_size": 25,
}

fig, axs = plt.subplots(1, 3)
ig.plot(
    g1,
    layout=g1.layout("circle"),
    target=axs[0],
    **visual_style,
)
ig.plot(
    g2,
    layout=g1.layout("circle"),
    target=axs[1],
    **visual_style,
)
ig.plot(
    g3,
    layout=g1.layout("circle"),
    target=axs[2],
    **visual_style,
)
fig.text(0.38, 0.5, '$\\simeq$' if g1.isomorphic(g2) else '$\\neq$', fontsize=15, ha='center', va='center')
fig.text(0.65, 0.5, '$\\simeq$' if g2.isomorphic(g3) else '$\\neq$', fontsize=15, ha='center', va='center')
plt.show()


#%% >>>>>>>>>>>>>>>> Ring Graph Animation

import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.animation as animation

g = ig.Graph.Ring(10, directed=True)

layout = g.layout_circle()

def _update_graph(frame):
    # Remove plot elements from the previous frame
    ax.clear()

    # Fix limits (unless you want a zoom-out effect)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    if frame < 10:
        # Plot subgraph
        gd = g.subgraph(range(frame))
    elif frame == 10:
        # In the second-to-last frame, plot all vertices but skip the last
        # edge, which will only be shown in the last frame
        gd = g.copy()
        gd.delete_edges(9)
    else:
        # Last frame
        gd = g

    ig.plot(gd, target=ax, layout=layout[:frame], vertex_color="yellow")

    # Capture handles for blitting
    if frame == 0:
        nhandles = 0
    elif frame == 1:
        nhandles = 1
    elif frame < 11:
        # vertex, 2 for each edge
        nhandles = 3 * frame
    else:
        # The final edge closing the circle
        nhandles = 3 * (frame - 1) + 2

    handles = ax.get_children()[:nhandles]
    return handles

fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, _update_graph, 12, interval=500, blit=True)
plt.ion()
plt.show()





#%% >>>>>>>>>>>>>>>>











#%% >>>>>>>>>>>>>>>>











#%% >>>>>>>>>>>>>>>>




















