# -*- coding: utf-8 -*-
from numpy import delete, vstack
from collections import namedtuple

def getnnsvar(D,crit):
    """
    Nearest neighbor of variables
    
    Performns nearest neighbor of variables.
    
    Parameters
    ----------
    D : DataFrame of shape (n_columns, n_columns)
        Dissimilarity matrix.
        
    crit : 1d array-like of shape (n_columns,)
        a vector of size \\emph{p} which indicates if we want to compute nearest neighbor of variable j (crit[j]=1) or not (crit[j]=0)
        
    Returns
    -------
    result : getnnsvar
        An object with the following attributes:
        
        nn : 1d numpy array of shape (n_columns,)
            list of nearest neighbor of each row.
        nndiss : 1d numpy array of shap e(n_columns,)
            nearest neigbbor distance of each row.
    """
    def nnsvar(crit):
        # flag criterion
        n_cols = len(crit)
        # extract last criterion
        last_crit = crit[n_cols-1]
        # remove last criterion
        crit2 = delete(crit,n_cols-1)
        if last_crit == 0:
            res = [0,0]
        else:
            minobs = [i for i,x in enumerate(crit2) if x == min(crit2[crit2 > 0])][0]
            mindis = crit2[minobs]
            res = [minobs,mindis]
        return [res]
    
    # make a copy of dissimilarity matrix
    Dprim = D.copy()
    # add criterion to dissimilarity matrix
    Dprim["crit"] = crit
    res = vstack([nnsvar(Dprim.iloc[i,:].to_numpy()) for i in range(Dprim.shape[0])])
    # convert to namedtuple
    return namedtuple("getnnsvar",["nn","nndiss"])(res[:,0],res[:,1])