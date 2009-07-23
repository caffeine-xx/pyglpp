import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def matrix_to_graph(matrix):
  G = nx.DiGraph()
  E = np.atleast_2d(matrix)
  for x in range(E.shape[0]):
    for y in range(E.shape[1]):
      udo.add_edge(x,y,E[x,y])
  return G

def draw_graph(G,savefile=None):
  pos = nx.circular_layout(G)
  colors = range(G.number_of_edges())
  nx.draw(G,pos,color='#336699',edge_color=colors,width=4,
          edge_cmap=plt.cm.Blues,with_labels=True)
  savefile!=None and plt.savefig(savefile)
  plt.show()
