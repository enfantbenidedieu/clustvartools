# -*- coding: utf-8 -*-
from pandas import concat, DataFrame
from sklearn.utils.validation import check_is_fitted

def sprintf(obj,**kwargs):
    """
    Print the analysis results
    
    Parameters
    ----------
    obj : class
        A fitted model.

    **kwargs :
        Additionals parameters. These parameters will be passed to `tabulate <https://pypi.org/project/tabulate/>`_.

    Returns
    -------
    NoneType
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if the estimator is fitted by verifying the presence of fitted attributes
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_fitted(obj)
    name, obj_attr = obj.__class__.__name__, [s for s in dir(obj) if s[0].isalpha() and s.endswith("_")]
    
    def attr_desc(attr):
        """
        Attributes description

        Parameters
        ----------
        attr : str
            The model attribute name.

        Returns
        -------
        desc : DataFrame of shape (1,2)
            Attribute long description
        """
        match attr:
            case "call_": desc = "summary called parameters"
            case "coef_": desc = "Coefficients"
            case "cluster_": desc = "results for clusters"
            case "levels_": desc = "results for the levels"
            case "levels_sup_": desc =  "results for the supplementary levels"
            case "quali_var_": desc = "results for the categorical variables"
            case "quali_var_sup_": desc =  "results for the supplementary categorical variables"
            case "quanti_var_": desc = "results for the continuous variables"
            case "quanti_var_sup_": desc = "results for the supplementary continuous variables"
            case "var_": desc = "results for variables"
            case "var_sup_": desc = "results for supplementary variables"
        return DataFrame([[f".{attr}", desc]],columns=["name","description"])
    
    # extract 
    res = concat((attr_desc(x) for x in obj_attr),axis=0,ignore_index=True).to_markdown(tablefmt="simple",index=False,**kwargs)
    
    text  = """
**Clustering of Variables - {}**\n
*The results are available in the following objects:\n
{}
""".format(name,res)
    print(text)