from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
import math


def scaled_laplacian(adj):
    """
    Return the Laplacian of the weight matrix.
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    """
    # d ->  diagonal degree matrix
    n, d = np.shape(adj)[0], np.sum(adj, axis=1)
    # L -> graph Laplacian
    L = -adj
    #L=D-adj,把对角的值换成d里的值
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    return L
 

#构建邻接矩阵（利用公式）是0/1矩阵？？？
def weight_matrix(adj,adj_scale,sigma2=10, epsilon=0.5,scaling=True):
    '''
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    # check whether W is a 0/1 matrix.
    if set(np.unique(adj)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = adj.shape[0]
        if adj_scale:
            adj= adj/ 10000.
            adj2, adj_mask = adj*adj, np.ones([n, n]) - np.identity(n)
            # refer to Eq.10
            adj=np.exp(-adj2 / sigma2) * (np.exp(-adj2 / sigma2) >= epsilon) * adj_mask
            return adj+np.identity(n)
        else:
            adj2, adj_mask = adj*adj, np.ones([n, n]) - np.identity(n)
            # refer to Eq.10
            adj=np.exp(-adj2 / sigma2) * (np.exp(-adj2 / sigma2) >= epsilon) * adj_mask
            return adj
    else:
        return adj

def fourier(L, algo='eigh', k=100):
    """Return the Fourier basis, i.e. the EVD of the Laplacian."""
    # print "eigen decomposition:"
    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]
    if algo is 'eig':
        lamb, U = np.linalg.eig(L)
        lamb, U = sort(lamb, U)
    elif algo is 'eigh':
        lamb, U = np.linalg.eigh(L)
        lamb, U = sort(lamb, U)
    elif algo is 'eigs':
        lamb, U = scipy.sparse.linalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo is 'eigsh':
        lamb, U = scipy.sparse.linalg.eigsh(L, k=k, which='SM')
    return lamb, U

#从这里入手，怎么样将s变成多个变量，然后矩阵相加
def weight_wavelet(s,lamb,U):
    s = s
    for i in range(len(lamb)):
        lamb[i] = math.pow(math.e,-lamb[i]*s)

    Weight = np.dot(np.dot(U,np.diag(lamb)),np.transpose(U))

    return Weight

def weight_wavelet_inverse(s,lamb,U):
    s = s
    for i in range(len(lamb)):
        lamb[i] = math.pow(math.e, lamb[i] * s)

    Weight = np.dot(np.dot(U,np.diag(lamb)), np.transpose(U))

    return Weight

def wavelet_basis(adj,sparse_ness,threshold,weight_normalize,s):
    L = scaled_laplacian(adj)
    lamb, U = fourier(L)
    Weight=weight_wavelet(s,lamb,U)
    inverse_Weight=weight_wavelet_inverse(s,lamb,U)
    del U,lamb 
    #这一步的操作是什么？？？
    if (sparse_ness):
        Weight[Weight < threshold] = 0.0
        inverse_Weight[inverse_Weight < threshold] = 0.0
    if (weight_normalize == True):
        Weight=normalize(Weight, norm='l1', axis=1)
        inverse_Weight=normalize(inverse_Weight, norm='l1', axis=1)
    t_k=[inverse_Weight,Weight]
    return t_k  