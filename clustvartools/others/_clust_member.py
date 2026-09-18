# -*- coding: utf-8 -*-
from pandas import DataFrame, Series, concat

def clust_member(D,cluster,method="min"):
    """
    Cluster Members
    
    Parameters
    ----------
    D : DataFrame of shape (n_columns, ncl)
        Input data.
    cluster : Series of shape (n_columns,)
        Labels.
    method : {"min","max"}, default = "min"
        Method.
        
    Return
    ------
    table : DataFrame of shape (n_columns, 5)
        Cluster members
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
    # check if valid kind
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not (method in ("min","max")):
        raise ValueError("Invalid method.")
    
    # cluster's members
    colnames = ["Variable","Cluster","Own Cluster","Next Closest"]
    if method == "min":
        colnames += ["Ratio (Own/Next)"]
    else:
        colnames += ["1 - R**2 Ratio"]
    
    # members
    def member(k,D,cluster,method="min"):
        i = cluster.loc[k]
        own = D.loc[k,i]
        others = []
        for j in list(D.columns):
            if i == j:
                continue
            others.append(D.loc[k,j])
        if method== "min":
            next = min(others) if len(others) > 0 else 0
            ratio = own/next
        else:
            next = max(others) if len(others) > 0 else 0
            ratio = (1 - own)/(1 - next)
        return Series([k, i, own, next, ratio])
    # concatenate
    res = concat((member(k=k,D=D,cluster=cluster,method=method).to_frame(i) for i,k in enumerate(D.index)),axis=1).T
    res.columns = colnames
    return res