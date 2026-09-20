# -*- coding: utf-8 -*-
from sklearn.utils.validation import check_is_fitted

def summary(obj, 
            digits = 4, 
            nbelt = 10,
            detailed=False, 
            to_markdown=False, 
            tablefmt = "simple", 
            **kwargs):
    """
    Printing summaries for clustering of variables model

    Parameters
    ----------
    obj : class
        A fitted clustering of variables model.

    digits : int, default = 4
        The number of decimal printed.

    nbelt : int, default = 10
        The number of element.

    detailed : bool, default = False
        To print detailed summaries.

    to_markdown : bool, default = False
        To print summaries in `markdown <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html>`_-friendly format. Requires the `tabulate <https://pypi.org/project/tabulate/>`_. package.

    tablefmt : str, default = "simple"
        The table format.

    **kwargs: Any
        Additionals parameters. These parameters will be passed to `tabulate <https://pypi.org/project/tabulate/>`_.

    Returns
    -------
    NoneType

    Examples
    --------
    >>> from clustvartools.datasets import decathlon
    >>> from clustvartools import CLV, summary
    >>> clf = CLV(ncl=3,sup_var=(10,11,12))
    >>> clf.fit(decathlon.data)
    CLV(ncl=3,sup_var=(10,11,12))
    >>> summary(clf)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)
    name = obj.__class__.__name__
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if valid method
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    clvmethod = ("CatCLV","CatHCAV","CatVARHCA","CLV","CLVmix","CorCLV","DCLV","HCAV","HCAVmix")
    if not (name in clvmethod):
        raise TypeError(f"Not allowed to an object of class {name}")
    
    # number of elements
    nbelt = min(nbelt,obj.call_.X.shape[1])

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print(f"                     Clustering of Variables - {name} Results                     ")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # cluster summary for ncl clusters
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if hasattr(obj,"cluster_"):
        print(f"\nCluster Summary for {obj.call_.ncl} Clusters:")
        cluster_summary = obj.cluster_.summary
        if to_markdown: 
            cluster_summary = cluster_summary.to_markdown(tablefmt=tablefmt,**kwargs)
        print(cluster_summary)
        
        print("\nInter-cluster Correlations:")
        icor = obj.cluster_.cor
        if to_markdown:
            icor = icor.to_markdown(tablefmt=tablefmt,**kwargs)
        print(icor)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # coefficients
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if detailed and name in ("CatCLV","CLV","DCLV","CLVmix"):
        print("\nLinear coefficients:")
        coef = obj.coef_.coef
        if to_markdown: 
            coef = coef.to_markdown(tablefmt=tablefmt,**kwargs)
        print(coef)
        
        print("\nStandardized scoring coefficients:")
        coef_std = obj.coef_.coef_std
        if to_markdown: 
            coef_std = coef_std.to_markdown(tablefmt=tablefmt,**kwargs)
        print(coef_std)
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ressults for clustering of variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # data
    if name == "CatCLV":
        data, member = obj.quali_var_.sqload, obj.quali_var_.member
    elif name == "CatHCAV":
        data, member = obj.quali_var_.dist, obj.quali_var_.member
    elif name == "CatVARHCA":
        data, member = obj.levels_.dist, obj.levels_.member
    elif name in ("CLV","DCLV"):
        data, member = obj.quanti_var_.sqload, obj.quanti_var_.member
    elif name == "CLVmix":
        data, member = obj.var_.sqload, obj.var_.member
    elif name == "CorCLV":
        data, member = obj.quanti_var_.diss, obj.quanti_var_.member
    elif name == "HCAV":
        data, member = obj.quanti_var_.dist, obj.quanti_var_.member
    else:
        data, member = obj.var_.dist, obj.var_.member
        
    # text of first data
    if name in ("CatHCAV","CatVARHCA","HCAV","HCAVmix"):
        text1 = "Cluster Distance"
    elif name in ("CLV","DCLV"):
        text1 = "Cluster R-square"
    elif name == "CorCLV":
        text1 = "Cluster Dissimilarities"
    else:
        text1 = "Cluster Squared Loadings"
        
    # text of second data
    if name in ("CatHCAV","CatVARHCA","HCAV","HCAVmix"):
        text2 = "Cluster Members and Distance values"
    elif name in ("CLV","DCLV"):
        text2 = "Cluster Members and R-square values"
    elif name == "CorCLV":
        text2 = "Cluster Members and Dissimilarities values"
    else:
        text2 = "Cluster Members and Squared Loadings values"
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # distance, dissimilarities, squared loadings
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print(f"\n{text1}:")
    data = data.iloc[:nbelt,:].round(decimals=digits)
    if to_markdown: 
        data = data.to_markdown(tablefmt=tablefmt,**kwargs)
    print(data)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # cluster members
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print(f"\n{text2}:")
    member = member.iloc[:nbelt,:].round(decimals=digits)
    if to_markdown: 
        member = member.to_markdown(tablefmt=tablefmt,**kwargs)
    print(member)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # informations for supplementary variables
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.call_.sup_var is not None:
        if name == "CatVARHCA":
            data2, member2 = obj.levels_sup_.dist, obj.levels_sup_.member
        elif name == "CatHCAV":
            data2, member2 = obj.quali_var_sup_.dist, obj.quali_var_sup_.member
        elif name == "HCAV":
            data2, member2 = obj.quanti_var_sup_.dist, obj.quanti_var_sup_.member
        elif name == "HCAVmix":
            data2, member2 = obj.var_sup_.dist, obj.var_sup_.member
        elif name in ("CatCLV","CLV","CLVmix","DCLV"):
            data2, member2 = obj.var_sup_.sqload, obj.var_sup_.member
        else:
            data2, member2 = obj.quanti_var_sup_.diss, obj.quanti_var_sup_.member
            
        print(f"\n{text1} - Supplementary variables:")
        data2 = data2.iloc[:nbelt,:].round(decimals=digits)
        if to_markdown: 
            data2 = data2.to_markdown(tablefmt=tablefmt,**kwargs)
        print(data2)
        
        print(f"\n{text2} - Supplementary variables:")
        member2 = member2.iloc[:nbelt,:].round(decimals=digits)
        if to_markdown: 
            member2 = member2.to_markdown(tablefmt=tablefmt,**kwargs)
        print(member2)