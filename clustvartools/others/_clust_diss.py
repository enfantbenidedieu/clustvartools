# -*- coding: utf-8 -*-
from numpy import linalg
from pandas import concat

# intern function
from ..clustering.functions.get_sup_label import get_sup_label
from ._clust_score import clust_score

def clust_diss(X,Y,tol=1e-7):
    """
    Calculates the aggregation criterion for two clusters of variables

    Calculates the measure of aggregation of two clusters of variables. 
    This measure of aggregation is equal to the decrease in homogeneity for the clusters being merged.
    
    Parameters
    ----------
    X : Series of shape (n_samples,) or DataFrame of shape (n_samples, n_xcolumns)
        First group of variables.
    Y : Series of shape (n_samples,) or DataFrame of shape (n_samples, n_xyolumns)
        Second group of variables.
    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger 
        than ``-tol*lambda1`` where ``lambda1`` is the largest eigenvalue.

    Return
    ------
    crit : float
        The aggregation measure between the two clusters.
    """
    # check if convenient row length
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Inconvenient row length")
    # concatenate
    Z = concat((X,Y),axis=1)
    # synthetic variables
    x = clust_score(X)
    y = clust_score(Y)
    z = clust_score(Z)
    # criteria (eq )
    crit = x.d + y.d - z.d
    if crit < tol:
        crit = 0
    return crit

def clust_diss2(R,id1,id2,tol=1e-7):
    """
    Dissimilarity between two clusters of variables

    Dissimilarity between two clusters of variables when only the covariance/correlation matrix is known.

    Parameters
    ----------
    R : DataFrame of shape (n_columns, n_columns)
        Covariance/Correlation matrix.
    id1 : list, tuple or range
        The indexes or names of cluster X.
    id2 : list, tuple or range
        The indexes or names of cluster Y.
    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger 
        than ``-tol*lambda1`` where ``lambda1`` is the largest eigenvalue.

    Return
    ------
    crit : float
        The dissimilarity between the two clusters.
    """
    # labels 
    id1_label = get_sup_label(X=R,indexes=id1,axis=1)
    id2_label = get_sup_label(X=R,indexes=id2,axis=1)
    # concatenate label
    id12_label = [*id1_label, *id2_label]
    # eigen values
    xeigval = linalg.svd(R.loc[id1_label,id1_label],hermitian=True)[1][0]
    yeigval = linalg.svd(R.loc[id2_label,id2_label],hermitian=True)[1][0]
    zeigval = linalg.svd(R.loc[id12_label,id12_label],hermitian=True)[1][0]
    # criteria
    crit = xeigval + yeigval - zeigval
    if crit < tol:
        crit = 0
    return crit