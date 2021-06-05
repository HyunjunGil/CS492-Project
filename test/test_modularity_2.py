from utils import calc_modularity, calc_modularity_small
from louvain import apply_louvain

nodes_ex1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
edges_ex1 = [
  [0,1],[1, 0],[1, 2],[2,1],[2,3],[3,2],[3,4],[4,3],[4,0],[0,4],[1,3],[3,1],
  [5,6],[6,5],[7,8],[8,7],[8,9],[9,8],[5,9],[9,5],[5,7],[7,5],[6,9],[9,6],[7,9],[9,7],
  [10,11],[11,10],[12,13],[13,12],[10,13],[13,10],[10,12],[12,10],[11,13],[13,11],
  [0,11],[11,0],[3,5],[5,3],[6,13],[13,6]
]
edge_weights_ex1 = {(x[0], x[1]) : 1 for x in edges_ex1}
cmty_ex1 = {0 : [0, 1, 2, 3, 4], 1 : [5, 6, 7, 8, 9], 2 : [10, 11, 12, 13]}
cmty_ex1_node = {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 1, 6 : 1, 7 : 1, 8 : 1, 9 : 1, 10: 2, 11 : 2, 12: 2, 13 : 2}
# print(calc_modularity(nodes_ex1, edges_ex1, edge_weights_ex1, cmty_ex1))
# print(calc_modularity_small(nodes_ex1, edges_ex1, edge_weights_ex1, cmty_ex1_node))
print(apply_louvain(edges_ex1, (14, 14)))