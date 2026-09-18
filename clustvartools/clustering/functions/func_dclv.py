# -*- coding: utf-8 -*-
import random

# intern functions
from .gsvd import gSVD

def gPCA(X,row_w,col_w,tol=1e-7):
    """
    Generalized Principal Component Analysis (gPCA)
    
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Standardized data.
    row_w : 1d array-like of shape (n_samples,), default = None
        An optional individuals weights.
    col_w : 1d array-like of shape (n_columns,), default = None
        An optional variables weights.
    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger 
        than ``-tol*lambda1`` where ``lambda1`` is the largest eigenvalue.
    
    Returns
    -------
    eigvals : 1d numpy array of shape (2,)
        The first and second eigenvalues
    eigprops : 1d numpy array of shape (2,)
        The proportion of first and second eigenvalues
    V : 2d array of shape (n_columns, 2)
        The eigenvectors.
    princomps : 2d numpy array of shape (n_samples, 2)
        The principal components.
    """
    # singular values decomposition
    svd = gSVD(X=X,ncp=2,row_w=row_w,col_w=col_w,tol=tol)
    # eigen values
    eigvals = svd.d
    # proportion
    eigprops = 100*eigvals/sum(eigvals)
    # principal components    
    princomps = svd.U[:,:2]*svd.vs[:2]
    return eigvals[:2], eigprops[:2], svd.V[:,:2], princomps
    
def vartot(X,row_w,col_w,tol,*cls):
    """
    Total of Variance and Proportion
    
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Standardized data
    row_w : 1d array-like of shape (n_rows,), default = None
        The rows weights.
    col_w : 1d array-like of shape (n_columns,), default = None
        The columns weights.
    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger 
        than ``-tol*lambda1`` where ``lambda1`` is the largest eigenvalue.
    *cls: 
        Additionals parameters

    Returns
    -------
    tot_var : float
        Total of variance
    tot_pro : float
        Total of proportion
    """
    # initialization
    nb_tot, tot_var, tot_prop = (0,) * 3
    for cl in cls:
        if cl == []:
            continue
        nb_elt = len(cl)
        # generalized singular values decomposition
        svd = gSVD(X=X.loc[:,cl],ncp=2,row_w=row_w,col_w=col_w.loc[cl],tol=tol)
        # eigen values
        eigvals = svd.d
        # proportions
        eigprops = (100*eigvals/sum(eigvals))[0]
        tot_var += eigvals[0]
        tot_prop = (tot_prop * nb_tot + eigprops * nb_elt) / (nb_tot + nb_elt)
        nb_tot += nb_elt
    return tot_var, tot_prop

def assigncl(X,row_w,col_w,tol,cl1,cl2,cls=None):
    """
    Assign Cluster
    
    """
    # concatenate
    if cls is None:
        cls = cl1 + cl2
    # init variance
    init_var = vartot(X,row_w,col_w,tol,cl1,cl2)[0]
    fin_cl1, fin_cl2 = cl1[:], cl2[:]
    check_var, max_var = (init_var,) * 2

    while True:
        for k in cls:
            new_cl1, new_cl2 = fin_cl1[:], fin_cl2[:]
            if k in new_cl1:
                new_cl1.remove(k)
                new_cl2.append(k)
            elif k in new_cl2:
                new_cl1.append(k)
                new_cl2.remove(k)
            else:
                continue

            new_var = vartot(X,row_w,col_w,tol,new_cl1,new_cl2)[0]
            if new_var > check_var:
                check_var = new_var
                fin_cl1, fin_cl2 = new_cl1[:], new_cl2[:]

        if max_var == check_var:
            break
        else:
            max_var = check_var
    return fin_cl1, fin_cl2, max_var

def stabilitycl(X,row_w,col_w,tol,cl1,cl2,cls=None,max_iter=10):
    # concatenate
    if cls is None:
        cls = cl1 + cl2
    fin_cl1, fin_cl2, max_var = assigncl(X,row_w,col_w,tol,cl1,cl2)

    for _ in range(max_iter):
        random.shuffle(cls)
        init_cl1, init_cl2, init_var = assigncl(X,row_w,col_w,tol,cl1,cl2,cls)
        if init_var > max_var:
            max_var = init_var
            fin_cl1, fin_cl2 = init_cl1, init_cl2
    return fin_cl1, fin_cl2, max_var