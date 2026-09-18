# -*- coding: utf-8 -*-
from numpy import ones, ndarray, array, sqrt,average,cov, linalg, real
from pandas import Series, get_dummies
from pandas.api.types import is_numeric_dtype

def coeffsim(X,Y,w=None):
    """
    Coefficient of Similarity
    
    Performns similarity between two variables. Variables
    
    Parameters
    ----------
    X : Series of shape (n_samples,)
        First variable.
    Y : Series of shape (n_samples,)
        Second variable.
    w : 1d array-like of shape (n_samples,), default = None
        The rows weights.
        
    Return
    ------
    value : float
        The similarity between X and Y.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an object of class Series
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,Series):
        raise TypeError(f"{type(X)} is not supported. Please convert to a Series with pd.Series.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.Series.html")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if Y is an object of class Series
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(Y,Series):
        raise TypeError(f"{type(Y)} is not supported. Please convert to a Series with pd.Series.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.Series.html")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if convenient length
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------    
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Inconvenient row length")
    
    n_rows = X.shape[0]
    # set individuals weights
    if w is None: 
        w = Series(ones(n_rows)/n_rows,index=X.index,name="weight")
    elif not isinstance(w,(list,tuple,ndarray,Series)): 
        raise TypeError("row_w must be a 1d array-like of individuals weights.")
    elif len(w) != n_rows: 
        raise ValueError(f"row_w must be a 1d array-like of shape ({n_rows},).")
    else:
        w = Series(array(w)/sum(w),index=X.index,name="weight")
    
    # check if numerics
    is_num1 = is_numeric_dtype(X)
    is_num2 = is_numeric_dtype(Y)
    
    # weighted and standard deviation
    def scale_unit(X,w):
        center = average(a=X,weights=w)
        scale = sqrt(cov(m=X,ddof=0,aweights=w))
        return (X - center)/scale 
    if is_num1:
        Zx = scale_unit(X=X,w=w)
    if is_num2:
        Zy = scale_unit(X=Y,w=w)
    # disjunctive data
    if not is_num1:
        Dx = get_dummies(data=X)
    if not is_num2:
        Dy = get_dummies(data=Y)
    
    if is_num1 and is_num2:
        return (sum(Zx*Zy)/n_rows)**2
    elif is_num1 and not is_num2:
        p_k = (Dy.T * w).sum(axis=1)
        n_k = n_rows * p_k
        A = Dy.T.dot(Zx)/n_k
        return sum((A**2)*p_k)
    elif not is_num1 and is_num2:
        p_k = (Dx.T * w).sum(axis=1)
        n_k = n_rows * p_k
        A = Dx.T.dot(Zy)/n_k
        return sum((A**2)*p_k)
    elif not is_num1 and not is_num2:
        px = (Dx.T * w).sum(axis=1)
        py = (Dy.T * w).sum(axis=1)
        X1, X2 = Dx/sqrt(px), Dy/sqrt(py)
        n_cols1 = Dx.shape[1]
        n_cols2 = Dy.shape[1]
        mincp = int(min(n_rows, n_cols1, n_cols2))
        if mincp == n_rows:
            A1 = X1.dot(X2.T)/n_rows
            A2 = X2.dot(X1.T)/n_rows
            A = A1.dot(A2)
        else:
            A1 = X1.T.dot(X2)/n_rows
            A2 = X2.T.dot(X1)/n_rows
            if mincp == n_cols1:
                A = A1.dot(A2)
            if mincp == n_cols2:
                A = A2.dot(A1)
        d = linalg.eigh(A)[0][::-1]
        return real(d[1])
        