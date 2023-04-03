from __future__ import absolute_import

import torch
import numpy as np
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


def adj_mx_from_skeleton(skeleton):
    num_joints = skeleton.num_joints()
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), skeleton.parents())))
    return adj_mx_from_edges(num_joints, edges, sparse=False)

'''------------yyf---------------'''
def get_adj():
    num_node=16    
    
    neighbor_link = [(0,1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
                             (0, 7), (7, 8), (8, 9), (8, 13), (13, 14),(14,15),
                             (8, 10), (10, 11), (11, 12),
                             (1, 4), (10, 13)]
    
    A = np.zeros((num_node, num_node))     
    for i, j in neighbor_link:
        A[j, i] = 1
        A[i, j] = 1
    for i in range(num_node):
        A[i,i]=1
    
    adj=normalize_digraph(A)
    
    return adj
def get_adj2():
    num_node=16    
    neighbor_link = [(6, 4), (4, 0), (0, 1),  (6, 5), (5, 2), (2, 3),
                     (6, 7), (7, 8), (8, 9),(8, 14), (14, 12),(12,13), 
                     (8, 15), (15, 10), (10, 11),
                     (4, 5), (14, 15), (6,15), (6,14)
                     ] 
#    neighbor_link = [(0,1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
#                             (0, 7), (7, 8), (8, 9), (8, 13), (13, 14),(14,15),
#                             (8, 10), (10, 11), (11, 12),
#                             (1, 4), (10, 13)]
    
    A = np.zeros((num_node, num_node))     
    for i, j in neighbor_link:
        A[j, i] = 1
        A[i, j] = 1
    for i in range(num_node):
        A[i,i]=1
    
    adj=normalize_digraph(A)
    
    return adj

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(Dn,A)    
    return AD