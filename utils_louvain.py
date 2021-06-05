import time
import numpy as np
import torch

def decompose_edge_list(edges, no_adj = False):
  edge_from = []
  edge_to = []
  adj_list = {}
  for edge in edges:
    edge_from.append(edge[0])
    edge_to.append(edge[1])
    if no_adj:
      continue
    if (edge[0] in adj_list.keys()):
      adj_list[edge[0]].append(edge[1])
    else:
      adj_list[edge[0]] = [edge[1]]

  return edge_from, edge_to, adj_list

def change_cmty_structure(cmty_info):

  # input dict) 
  # key: node index         value: community index for the node
  # [ex] {1 : 2, 2 : 3 ...}
  # output dict)
  # key: community index    value: nodes indices that contained in the community
  # [ex] {1 : [1, 4, 5], 2 : [2, 3, 7] ...}

  new_dict = {}
  for i, j in cmty_info.items():
    if (j in new_dict.keys()):
      new_dict[j].append(i)
    else:
      new_dict[j] = [i]
  return new_dict




def calc_modularity(node_list, edge_list, edge_weights, cmty_info, quite = False):
  print('++++++++++++++++++++++==')
  # print(node_list)
  # print(edge_list)
  # print(edge_weights)
  # print(cmty_info)
  edge_from, edge_to, _ = decompose_edge_list(edge_list, no_adj = True)
  assert len(edge_from) == len(edge_weights)
  edge_weights_list = [edge_weights[(edge_from[i], edge_to[i])] for i in range(len(edge_weights))]
  # cuda
  adj = torch.sparse_coo_tensor(indices = [edge_from, edge_to], values = edge_weights_list, size = (len(node_list), len(node_list)), dtype = torch.float).cuda() #cuda
  modularity = 0
  m_2 = len(edge_list)
  t_tot = time.time()
  for c_idx in cmty_info.keys():
    if not quite:
      if (c_idx % 100 == 0):
        print('(' + str(c_idx) + ') curent modularity : ', modularity)
        if (c_idx != 0):
          t = time.time()
    
    temp = torch.Tensor([(1 if i in cmty_info[c_idx] else 0) for i in range(len(node_list))]).type(torch.float).reshape(len(node_list), 1).cuda() # cuda
    temp = torch.mm(adj, temp).reshape(len(node_list), 1)
    A = temp[cmty_info[c_idx], :].sum()
    K = temp.sum() ** 2

    modularity += (A - K / m_2) / m_2

  if not quite:
    print('========================================')
    print('modularity of given graph : ', modularity)
    print('total elapsed time: ', time.time() - t_tot)

  return modularity, time.time() - t_tot


def calc_modularity_small(node_list, edge_list, edge_weights_list, cmty_info):
  num_cmty = len(set(cmty_info.values()))
  encode_cmty = {x : i for i, x in enumerate(set(cmty_info.values()))}
  E = torch.zeros(num_cmty, num_cmty)
  for edge in edge_list:
    cmty_from = encode_cmty[cmty_info[edge[0]]]
    cmty_to = encode_cmty[cmty_info[edge[1]]]
    E[cmty_from, cmty_to] += edge_weights_list[(edge[0], edge[1])]

  E = E/len(edge_list)

  print('E.sum() : ', E.sum())

  return E.trace() - torch.mm(E, E).sum()

## Define Louvain method
def louvain_step1(node_list, edge_list, edge_weights, cmty_info, miss = 5):

  print('Step 1 : Find local maxima.')

  # node_list     : array of nodes.  Need to be evenly distributed(For example, [0, 1, 2, 3, 4, ..])
  # edge_list     : array of edges([u, v]). IT CONTAINS BOTH (u, v) and (v, u)
  # edge_weights  : dictionary of weights of edges
  # cmty_info     : dictionary with community information
  # miss          : acceptable consecutive fails to update maximum del_Q.

  # print(node_list)
  # print(edge_list)

  node_list_shuffle = node_list.copy()
  
  edge_from, edge_to, _ = decompose_edge_list(edge_list, no_adj = True)

  edge_from = list(map(lambda x : x[0], edge_list))
  edge_to = list(map(lambda x : x[1], edge_list))

  edge_weights_list = [edge_weights[(edge_from[i], edge_to[i])] for i in range(len(edge_weights))]
  m_2 = len(edge_list)

  adj = torch.sparse_coo_tensor(indices = [edge_from, edge_to], values = edge_weights_list, size = (len(node_list), len(node_list)), dtype = torch.float).cuda() # cuda

  
  updated = True
  while updated:
    updated = False
    np.random.shuffle(node_list_shuffle)
    for i in range(len(node_list_shuffle)):
      temp_node = node_list_shuffle[i]
      temp_cmty = cmty_info[temp_node]
      visited = [temp_cmty]
      max_dq = 0
      max_cmty = temp_cmty
      for cmty_idx in set(cmty_info.values()):
        if (cmty_idx in visited):
          continue
        visited.append(cmty_idx)
        nodes_in_cmty = list(filter(lambda x : cmty_info[x] == cmty_idx, cmty_info.keys()))

        cmty_vec = torch.Tensor([(1 if node_list[i] in nodes_in_cmty else 0) for i in range(len(node_list))]).type(torch.float).reshape(len(node_list), 1).cuda()
        cmty_vec_sum = torch.mm(adj, cmty_vec).reshape(len(node_list), 1)

        s_tot = cmty_vec_sum.sum()

        k_in = cmty_vec_sum[temp_node]
        k_tot = torch.mm(adj, torch.ones(len(node_list)).reshape(len(node_list, 1)).cuda())[temp_node]

        dq = k_in / m_2 - ((s_tot + k_tot) / m_2) ** 2 + (s_tot / m_2) ** 2 + (k_tot / m_2) ** 2

        if (max_dq < dq):
          max_dq = dq
          max_cmty = cmty_idx

      if (max_dq > 0):
        cmty_info[temp_node] = max_cmty
        updated = True

  print('Done.')

  return

def louvain_step2(edge_list, edge_weights, cmty_info):
  print('Step 2 : Aggregate vertices in same community')
  new_node_list = list(set(list(cmty_info.values())))

  new_edge_weights = {}

  for edge in edge_list:
    x_c = cmty_info[edge[0]]
    y_c = cmty_info[edge[1]]
    if (x_c, y_c) in new_edge_weights.keys():
      new_edge_weights[(x_c, y_c)] += edge_weights[(edge[0], edge[1])]
    else:
      new_edge_weights[(x_c, y_c)] = edge_weights[(edge[0], edge[1])]

  new_edge_list = list(map(lambda x : [x[0], x[1]], list(new_edge_weights.keys())))
  new_cmty_info = {i : i for i in new_node_list}

  print('Done.')
  print('Number of community : ', len(set(list(new_cmty_info.values()))))
  
  return new_node_list, new_edge_list, new_edge_weights, new_cmty_info

def louvain(node_list, edge_list, edge_weights, cmty_info, miss = 5, iter = 10):
  saved_cmty_info = cmty_info.copy()

  for i in range(iter):
    print('====================== Iter {} =============================='.format(i))
    encode_map = {node_list[i] : i for i in range(len(node_list))}
    decode_map = {i : node_list[i] for i in range(len(node_list))}
    # print(cmty_info)
    # print(encode_map)
    # print(decode_map)
    # encode
    node_list_en = list(map(lambda x : encode_map[x], node_list))
    edge_list_en = list(map(lambda x : [encode_map[x[0]], encode_map[x[1]]], edge_list))
    edge_weights_en = {(encode_map[x[0]], encode_map[x[1]]) : y for x, y in edge_weights.items()}
    cmty_info_en = {encode_map[x] : encode_map[y] for x, y in cmty_info.items()}
    louvain_step1(node_list_en, edge_list_en, edge_weights_en, cmty_info_en, miss = miss)

    print(cmty_info_en)

    cmty_info_de = {decode_map[x] : decode_map[y] for x, y in cmty_info_en.items()}

    for k in saved_cmty_info.keys():
      saved_cmty_info[k] = cmty_info_de[saved_cmty_info[k]]
    node_list, edge_list, edge_weights, cmty_info = louvain_step2(edge_list, edge_weights, cmty_info_de)
    

  return saved_cmty_info
    



  # while not local_maxima:
  #   print('(update{}) acc_dq = {}'.format(update_cnt, acc_dq))

  #   update_cnt += 1
  #   node = np.random.choice(node_list)
  #   max_dq = 0
  #   max_node = cmty_info[node]
  #   visited = [cmty_info[node]]
  #   prev_cmty = cmty_info[node]
  #   for v in adj_list[node]:
  #     # print(v, cmty_info[v])
  #     temp_cmty_idx = cmty_info[v]
  #     if (temp_cmty_idx in visited):
  #       continue
  #     visited.append(temp_cmty_idx)
      


  #     temp_cmty = list(filter(lambda x : cmty_info[x] == temp_cmty_idx, cmty_info.keys()))

  #     cmty_vec = torch.Tensor([(1 if i in temp_cmty else 0) for i in node_list]).type(torch.float).reshape(len(node_list), 1).cuda() # cuda
  #     cmty_sum_vec = torch.mm(adj, cmty_vec).reshape(len(node_list), 1) # already dense tensor
      
  #     s_tot = cmty_sum_vec.sum()

  #     k_i_in = cmty_sum_vec[node_list.index(node), 0]
  #     k_i_tot = torch.mm(adj, torch.ones(len(node_list)).reshape(len(node_list), 1).cuda() )[node_list.index(node)] # cuda

  #     dq = k_i_in / m_2 - ((s_tot + k_i_tot) / m_2) ** 2 + (s_tot / m_2) ** 2 + (k_i_tot / m_2) ** 2

  #     if (max_dq < dq):
  #       max_dq = dq
  #       max_node = v

  #   if (prev_cmty == cmty_info[max_node]):
  #     miss_cnt += 1
  #     print('Miss ',miss_cnt)
  #     if (miss_cnt >= miss):
  #       print('Five consecutive failure in updating. Break.')
  #       break
  #   else:
  #     miss_cnt = 0
  #     cmty_info[node] = cmty_info[max_node]
  #     acc_dq += max_dq