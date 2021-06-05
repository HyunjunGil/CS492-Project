from utils import calc_modularity, calc_modularity_small, change_cmty_structure

### test calc_modularity function for simple graph 

nodes_wiki = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
edges_wiki = [
  [0,1],[1,0],[0,2],[2,0],[0,9],[9,0],
  [1,2],[2,1],
  [3,4],[4,3],[3,5],[5,3],[3,9],[9,3],
  [4,5],[5,4],
  [6,7],[7,6],[6,8],[8,6],[6,9],[9,6],[7,8],[8,7]
]
edge_weights_wiki = {(x[0], x[1]) : 1 for x in edges_wiki}
cmty_info = {
  0 : 0,
  1 : 0,
  2 : 0,
  3 : 1,
  4 : 1,
  5 : 1,
  6 : 2,
  7 : 2,
  8 : 2,
  9 : 0,
}

print(calc_modularity(nodes_wiki, edges_wiki, edge_weights_wiki, change_cmty_structure(cmty_info)))
print(calc_modularity_small(nodes_wiki, edges_wiki, edge_weights_wiki, cmty_info))