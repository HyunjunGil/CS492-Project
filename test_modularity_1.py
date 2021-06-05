from utils import calc_modularity, calc_modularity_small

### test calc_modularity function for simple graph 

nodes_simple = [0, 1, 2, 3, 4]
edges_simple = [[0,1],[1,0],[3,4],[4,3],[0,2],[2,0],[1,4],[4,1],[2,3],[3,2]]
# edge_weights_simple = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
edge_weights_simple = {(x[0], x[1]) : 1 for x in edges_simple}
cmty_simple = {0 : [0, 2, 3], 1 : [1, 4]}
cmty_simple_node = {0 : 0, 1 : 1, 2 : 0, 3 : 0, 4 : 1}
print(calc_modularity(nodes_simple, edges_simple, edge_weights_simple, cmty_simple))
print(calc_modularity_small(nodes_simple, edges_simple, edge_weights_simple, cmty_simple_node))