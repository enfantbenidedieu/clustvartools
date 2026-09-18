# -*- coding: utf-8 -*-
from pandas import DataFrame, Series

def clust_dist(D,cluster,linkage="average"):
    """
    Distance between a variable and a cluster
    
    Parameters
    ----------
    D : DataFrame of shape (n_columns, n_columns) or (n_columns_sup, n_columns)
        Matrix of distance between columns.
    cluster : Series of shape (n_columns,)
        Labels.
    linkage : {"average", "complete", "single", "ward"}, default = "average"
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        * "average" uses the average of the distances of each observation of
            the two sets.
        * "complete" linkage uses the maximum distances between
            all observations of the two sets.
        * "single" uses the minimum of the distances between all observations
            of the two sets.
        * "ward" minimizes the variance of the clusters being merged.
        
    Return
    ------
    dist : DataFrame of shape (n_columns, ncl)
        Distance between a variable and a cluster    
    """
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if D is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(D,DataFrame):
        raise TypeError(f"{type(D)} is not supported. Please convert to a DataFrame with pd.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if cluster is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(cluster,Series):
        raise TypeError(f"{type(cluster)} is not supported. Please convert to a Series with pd.Series.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.Series.html")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if linkage method is valid
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (linkage in ("average","complete","single","ward")):
        raise ValueError("linkage should be one of 'average', 'complete', 'single', 'ward'")
    
    # cluster labels
    cluster_name = sorted(cluster.unique())
    # distance for levels to the cluster
    dist = DataFrame(index=D.index,columns=cluster_name).astype("float")
    for l in D.index:
        for k in cluster_name:
            clus = cluster[cluster==k].index
            if len(clus) > 1:
                if linkage == "average":
                    d = sum(D.loc[l,clus])/len(clus)
                elif linkage == "complete":
                    d = max(D.loc[l,clus])
                elif linkage == "single":
                    d = min(D.loc[l,clus])
                else:
                    d = (1/(len(clus) + 1))*sum(D.loc[l,clus])
            else:
                d = D.loc[l,clus].values[0]
            dist.loc[l,k] = d
    return dist