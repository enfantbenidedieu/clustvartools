# -*- coding: utf-8 -*-
from numpy import sqrt, linalg
from pandas import Series
from collections import namedtuple

def clust_score(X,tol=1e-7):
    """
    Calculates the synthetic variable of a cluster of variables.
    
    Calculates the synthetic variable of a cluster of variables. The variables can be quantitative or qualitative. 
    The synthetic variable is the first principal component of PCAmix or FAMD. The variance of the synthetic variable is the first eigenvalue. 
    It is equal to the sum of squared correlations or correlation ratios to the synthetic variable. It measures the homogeneity of the cluster.
    
    Parameters
    ----------
    X : Series of shape (n_samples,) or DataFrame of shape (n_samples, n_columns)
        Standardized data.
        
    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger 
        than ``-tol*lambda1`` where ``lambda1`` is the largest eigenvalue.
        
    Returns
    -------
    result : clust_score
        An object with the following attributes:
        
        vs : float
            The first singular value
        d : float
            The firsty eigen values, ie.e square of singular value
        u : 1d numpy array of shape (n_samples,)
            The synthetic variables i.e. the scores
        v : 1d numpy array of shape (n_columns,)
            The standardized loadings
    """
    # convert X to DataFrame if Series
    if isinstance(X,Series):
        X = X.to_frame()
    
    # number of rows
    n_rows = X.shape[0]
    # divide by number of samples
    Z = X/sqrt(n_rows)
    # singular value decomposition
    svd = linalg.svd(Z)
    # set maximum number of components
    rank = sum(((svd[1]/svd[1][0])**2)>tol)
    # compute separate element
    vs = svd[1][0]
    d = vs**2
    p = d/sum(svd[1][:rank]**2)
    u = svd[0][:,0]*svd[1][0]*sqrt(n_rows)
    v = svd[2].T[:,0]
    return namedtuple("clust_score",["vs","d","p","u","v"])(vs,d,p,u,v)