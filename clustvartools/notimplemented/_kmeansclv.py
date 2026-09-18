# -*- coding: utf-8 -*-
from numpy import zeros,ones, ndarray, repeat, array, where, c_, sqrt, diag
from pandas import Series, DataFrame, concat
from pandas.api.types import is_numeric_dtype
from collections import namedtuple
from scipy.cluster.hierarchy import fcluster
from scipy.stats import t as sst
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#interns functions
from .functions.preprocessing import preprocessing
from .functions.get_sup_label import get_sup_label
from .functions.statistics import wmean, wstd, wcorr
from .functions.tests import wpearsonr, wcorrtest
from ..others._clust_diss import clust_diss
from ..others._getnnsvar import getnnsvar
from ..others._clust_score import clust_score
from ..others._coeffsim import coeffsim
from ..others._clust_member import clust_member

class KMeansCLV(BaseEstimator,TransformerMixin):
    """
    K-Means Clustering of Variables around Latent Variables (KMeansCLV)
    
    
    
    
    """