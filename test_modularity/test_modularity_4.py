from utils import calc_modularity, calc_modularity_small, change_cmty_structure
from louvain import apply_louvain
import numpy as np

nodes_karate = [i for i in range(34)]
edges_karate = [
  [2, 1],
  [3, 1], [3, 2], 
  [4, 1], [4, 2], [4, 3],
  [5, 1],
  [6, 1],
  [7, 1], [7, 5], [7, 6],
  [8, 1], [8, 2], [8, 3], [8, 4],
  [9, 1], [9, 3],
  [10, 3],
  [11, 1], [11, 5], [11, 6],
  [12, 1],
  [13, 1], [13, 4],
  [14, 1], [14, 2], [14, 3], [14, 4],
  [17, 6], [17, 7],
  [18, 1], [18, 2],
  [20, 1], [20, 2],
  [22, 1], [22, 2],
  [26, 24], [26, 25],
  [28, 3], [28, 24], [28, 25],
  [29, 3],
  [30, 24], [30, 27],
  [31, 2], [31, 9],
  [32, 1], [32, 25], [32, 26], [32, 29],
  [33, 3], [33 ,9], [33 ,15], [33 ,16], [33 ,19], [33 ,21], [33, 23],[33 ,24],[33, 30],[33,31], [33, 32],
  [34, 9], [34, 10], [34 ,14], [34, 15], [34 ,16], [34, 19], [34 ,20], [34, 21], [34, 23] ,[34, 24] ,[34, 27], [34, 28] ,[34, 29] ,[34 ,30], [34, 31], [34, 32] ,[34, 33]  
]
edges_karate = list(map(lambda x : [x[0]-1, x[1] - 1], edges_karate))
edges_karate = edges_karate + list(map(lambda x : [x[1], x[0]], edges_karate))
edge_weights_karate = {(x[0], x[1]) : 1 for x in edges_karate}
print(len(edges_karate), len(edge_weights_karate))
cmty_info_karate = {i : i for i in range(len(nodes_karate))}
cmty_info_karate_sol = {
  0 : [0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21],
  1 : [4, 5, 6, 10, 16],
  2 : [8, 9, 14, 15, 18, 20, 22, 30, 32, 33],
  3 : [23, 24, 25, 26, 27, 28, 29, 31]
}

cmty_info_karate_node = {
  0:0,
  1:0,
  2:0,
  3:0,
  4:1,
  5:1,
  6:1,
  7:0,
  8:2,
  9:2,
  10:1,
  11:0,
  12:0,
  13:0,
  14:2,
  15:2,
  16:1,
  17:0,
  18:2,
  19:0,
  20:2,
  21:0,
  22:2,
  23:3,
  24:3,
  25:3,
  26:3,
  27:3,
  28:3,
  29:3,
  30:2,
  31:3,
  32:2,
  33:2
}

# print(calc_modularity(nodes_karate, edges_karate, edge_weights_karate, cmty_info_karate_sol))
# print(calc_modularity_small(nodes_karate, edges_karate, edge_weights_karate, cmty_info_karate_node))

labels, q = apply_louvain(edges_karate, size = (34, 34))
print(labels)
print(np.array([1, 2, 3, 4, 5, 6,7]))