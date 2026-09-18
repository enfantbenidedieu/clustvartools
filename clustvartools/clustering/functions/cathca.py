# -*- coding: utf-8 -*-
from numpy import array, sqrt, dot, sum
from pandas import crosstab
from scipy.stats.contingency import association
from sklearn.metrics import rand_score, adjusted_rand_score

def catsimdissim(x,y,method="dice"):
    """
    Compute similarity/dissimilarity between two dummies vectors x and y

    Parameters
    ----------
    x : 1D array-like of shape (n_samples,)
        First vector.
    y : 1D array-like of shape (n_samples,)
        Second vector.
    method : str, default = "dice"
        The method.
    
    Returns
    -------
    value : float
        Similarity/dissimilarity between x and y
    """
    x, y = array(x), array(y)
    value = sqrt(sum((x-y)**2)/2) if method == "dice" else dot(x,y)/len(x)
    return value

def simqual(x,y,method="cramer",normalize=True):
    """
    Compute similarity between two categorical vectors x and y

    Parameters
    ----------
    x : 1D array-like of shape (n_samples,)
        First vector.
    y : 1D array-like of shape (n_samples,)
        Second vector.
    method : str, default = "cramer"
        Metric.
    normalize : bool, default = True
        If True, then tschuprow and pearson metrics are normalize to range between 0 and 1.

    Returns
    -------
    value : float
        Similarity between x and y
    """
    if method in ("cramer","tschuprow","pearson"):
        value = association(crosstab(x,y),method=method,correction=False)
        if normalize and method in ("tschuprow","pearson"):
            r , c = len(x.unique()), len(y.unique())
            if method == "tschuprow":
                value_max = sqrt(min(r-1,c-1)/max(r-1,c-1))
            if method == "pearson":
                value_max = sqrt((max(r,c)-1)/min(r,c))
            value = value/value_max
    elif method == "rand":
        value = rand_score(x,y)
    elif method == "rand.adj":
        value = adjusted_rand_score(x,y)
    else:
        value = None
    return value