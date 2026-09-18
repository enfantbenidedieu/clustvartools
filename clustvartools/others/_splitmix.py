# -*- coding: utf-8 -*-
from numpy import number
from pandas import DataFrame, Series
from collections import namedtuple
from typing import NamedTuple

def splitmix(
        X
) -> NamedTuple:
    """
    Split Mixed Data

    Splits a mixed data matrix in two data sets: one with the quantitative variables and one with the qualitative variables.
    
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data.

    Returns
    -------
    result : splitmixResult
        An object with the following attributes:

        quanti: DataFrame of shape (n_samples, k1) default = None
            Continuous variables.

        quali: DataFrame of shape (n_samples, k2), default = None
            Categorical variables.

        n: int, default = n_samples
            Number of rows.

        k1: int, default = 0
            Number of continuous variables.

        k2: int, default = 0
            Number of categorical variables.

    Examples
    --------
    >>> from scientisttools.datasets import wine
    >>> from scientisttools import splitmix
    >>> split_X = splitmix(wine)
    >>> X_quanti, X_quali, n_quanti, n_quali = split_X.quanti, split_X.quali, split_X.k1, split_X.k2
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. X must be an object of class pd.DataFrame")

    #initialisation
    X_quanti, X_quali, n, k1, k2 = None, None, X.shape[0], 0, 0

    #select all numerics columns
    is_quanti = X.select_dtypes(include=number)
    if not is_quanti.empty:
        X_quanti = is_quanti.to_frame() if isinstance(is_quanti,Series) else is_quanti
        k1 = X_quanti.shape[1]

    #select object or category
    is_quali = X.select_dtypes(include=["object","category"])
    if not is_quali.empty:
        X_quali = is_quali.to_frame() if isinstance(is_quali,Series) else is_quali
        k2 = X_quali.shape[1]

    #convert to namedtuple
    return namedtuple("splitmixResult",["quanti","quali","n","k1","k2"])(X_quanti,X_quali,n,k1,k2)