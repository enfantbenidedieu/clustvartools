# -*- coding: utf-8 -*-
from numpy import c_, diag, ones
from pandas import DataFrame, Series
from collections import namedtuple
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#intern functions
from .functions.preprocessing import preprocessing
from .functions.get_sup_label import get_sup_label
from .functions.cathca import simqual
from ..others._clust_dist import clust_dist
from ..others._clust_member import clust_member

class CatHCAV(BaseEstimator,TransformerMixin):
    """
    Hierarchical Clustering Analysis of Categorical Variables (CatHCAV)
    
    Performs agglomerative clustering algorithm for categorical Variables. Supplementary categorical variables may be used.
    
    Parameters
    ----------
    ncl : int, default = 2
        If a (positive) integer, the tree is cut with ncl clusters. If None, then the ncl using optimal point.
    
    method : {"cramer", "tschuprow", "pearson", "rand", "rand.adj"}, default = "cramer"
        The similarity method.
        
    metric : callable, default = lambda x : 1 - x
        Callable with input a float. This refers to how similarity matrix :math:`S` is converting into dissimilarity matrix :math:`D`.
        
    linkage : {"average", "complete", "single", "ward"}, default = "ward"
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        * "average" uses the average of the distances of each observation of the two sets.
        * "complete" linkage uses the maximum distances between all observations of the two sets.
        * "single" uses the minimum of the distances between all observations of the two sets.
        * "ward" minimizes the variance of the clusters being merged.

    normalize : bool, default = True
        If True, then tschuprow and pearson metrics are normalize to range between 0 and 1.
    
    sup_var : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary categorical variables.

    Returns
    -------
    call_ : call
        An object containing the summary called parameters with the following attributes:

        Xtot : DataFrame of shape (n_samples, n_columns + n_columns_sup)
            Input data.
        X : DataFrame of shape (n_samples, n_columns)
            Active data.
        ncl : int
            The number of clusters.
        tree : tree
            An object containing the results for the hierarchical agglomerative clustering algorithm, with the following attributes:
            
            S : DataFrame of shape (n_columns, n_columns)
                Similarity matrix.
            D : DataFrame of shape (n_columns, n_columns)
                Dissimilarity matrix.
            Z : 2d numpy array of shape (n_columns - 1, 4)
                Linkage matrix.
            height: DataFrame of shape (n_columns - 1, 4)
                Reverse Height of aggregation between ``Z[:,0]`` and ``Z[:,1]``.
            merge : 1d numpy array of shape (n_columns - 1,)
                Height of aggregation between ``Z[:,0]`` and ``Z[:,1]``.
            size : 1d numpy array of shape (n_columns - 1,)
                Number of original observations in the newly formed cluster.

        sup_var : None, list
            The names of the supplementary categorical variables.

    quali_var_ : quali_var
        An object containing the description of the clusters by the categorical variables :

        cluster : Series of shape (n_columns,)
            The labels of each categorical variable.
        dist : DataFrame of shape (n_columns, ncl)
            The distance of each categorical variable to the cluster centers.
        member : DataFrame of shape (n_columns, 3)
            Cluster's members of each categorical variable (distance to own cluster, distance to next closest, ratio (own/next)).

    quali_var_sup_ : quali_var_sup, optional
        An object containing the description of the clusters by the supplementary categorical variables :

        cluster : Series of shape (n_columns_sup,)
            The labels of each supplementary categorical variable.
        dist : DataFrame of shape (n_columns_sup, ncl)
            The distance of each supplementary categorical variable to the cluster centers.
        member : DataFrame of shape (n_columns_sup, 3)
            Cluster's members of each supplementary categorical variable (distance to own cluster, distance to next closest, ratio (own/next)).

    References
    ----------
    [1] H. Abdallah, G. Saporta. (1998). `Classification d'un ensemble de variables qualitatives <https://www.numdam.org/item/RSA_1998__46_4_5_0.pdf>`_. in Revue de Statistique Appliquée, Tome 46, N°4, pp. 5-26
    
    [2] Rakotomalala, R. `Classification des variables qualitatives : Regroupement de variables, regroupement de modalités <https://eric.univ-lyon2.fr/ricco/cours/slides/classif_variables_quali.pdf>`_. Université Lumière Lyon 2.
    
    Examples
    --------
    >>> from clustvartools.datasets import voting
    >>> from clustvartools import CatHCAV
    >>> # cramer similarity
    >>> clf = CatHCAV(ncl=2,sup_var=0)
    >>> clf.fit(voting)
    CatHCAV(ncl=2,sup_var=0)
    """
    def __init__(
            self, 
            ncl = 2,
            method = "cramer", 
            metric = lambda x : 1 - x,
            linkage = "ward", 
            normalize = True, 
            sup_var = None
    ):
        self.ncl = ncl
        self.method = method
        self.metric = metric
        self.linkage = linkage
        self.normalize = normalize
        self.sup_var = sup_var

    def fit(self,X,y=None):
        """Compute CatHCAV

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            Training data, where ``n_samples`` in the number of samples 
            and ``n_columns`` is the number of columns.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if method option is valid
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (self.method in ("cramer","tschuprow", "pearson","rand","rand.adj")):
            raise ValueError("method should be one of 'cramer', 'tschuprow', 'pearson', 'rand', 'rand.adj'")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if linkage criterion is valid
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (self.linkage in ("average","complete","single","ward")):
            raise ValueError("linkage should be one of 'average', 'complete', 'single', 'ward'")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # preprocessing (drop level, fill NA with mean, convert to ordinal levels)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X)
    
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # get supplementary categprocal labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        sup_var_label = get_sup_label(X=X,indexes=self.sup_var,axis=1)

        # make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # drop supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # drop supplementary categorical variables
        if self.sup_var is not None:
            X_sup_var, X = X.loc[:,sup_var_label], X.drop(columns=sup_var_label)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # categorical variables HCA
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set number of categorical variables
        n_cols = X.shape[1]
        # similarity matrix
        S = DataFrame(diag(ones(n_cols)),index=X.columns,columns=X.columns).astype("float")
        for i in range(n_cols-1):
            for j in range(i+1,n_cols):
                S.iloc[i,j] = simqual(x=X.iloc[:,i],y=X.iloc[:,j],method=self.method,normalize=self.normalize)
                S.iloc[j,i] = S.iloc[i,j]
        # dissimilarity matrix
        D = S.transform(func=self.metric)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # agglomerative hierarchical clustering of categorical variables 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # linkage matrix
        Z = sch.linkage(squareform(D,checks=False), method=self.linkage)
        # height
        height = DataFrame(c_[list(range(1,Z.shape[0]+1)),Z[:,2][::-1]],columns=["cluster","height"])
        height["diff_1"] = -1*height["height"].diff(1)
        height["diff_2"] = height["diff_1"].diff(-1)
        height["cluster"] = height["cluster"].astype("int")

        # convert to dictionary
        tree_ = {"S":S,"D":D,"Z":Z,"height":height,"merge":Z[:,:2],"size":Z[:,3]}
        # convert to namedtuple
        tree = namedtuple("tree",tree_.keys())(*tree_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set numbers of clusters
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ncl is None:
            ncl = height[height["diff_2"]==height["diff_2"].max()]["cluster"].values[0]
        elif self.ncl < 0:
            raise TypeError("ncl should be a positive integer.")
        elif not isinstance(self.ncl,int):
            raise TypeError("ncl should be an integer")
        else:
            ncl = self.ncl

        # convert to ordered dictionary
        call_ = {"Xtot":Xtot,"X":X,"ncl":ncl,"tree":tree,"sup_var":sup_var_label}
        # convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Informations for categorical variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # assign cluster
        cluster = Series(sch.fcluster(Z=Z,t=ncl,criterion="maxclust"), index = D.index,name = "cluster",dtype="category")
        
        # distance for categorical variables to the cluster
        dist_cluster = clust_dist(D=D,cluster=cluster,linkage=self.linkage)
        
        # cluster's members : distance own cluster, distance next closest, ratio (own/next)
        cluster_member = clust_member(D=dist_cluster,cluster=cluster,method="min")
        
        #convert to dictionary
        quali_var_ = {"cluster":cluster,"dist":dist_cluster,"member":cluster_member}
        #convert to namedtuple
        self.quali_var_ = namedtuple("quali_var",quali_var_.keys())(*quali_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #clusters for supplementary categorical variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            # similarity matrix
            S_sup = DataFrame(index=sup_var_label,columns=X.columns).astype("float")
            for i in sup_var_label:
                for j in X.columns:
                    S_sup.loc[i,j] = simqual(x=X_sup_var[i],y=X[j],method=self.method,normalize=self.normalize) 
            # dissimilarity matrix
            D_sup = S_sup.transform(func=self.metric)
            
            # distance between supplementary categorical variables and clusters
            dist_sup_cluster = clust_dist(D=D_sup,cluster=cluster,linkage=self.linkage)
            
            # assign cluster to supplementary categorical variables
            quali_var_sup_cluster = dist_sup_cluster.idxmin(axis=1).astype("category")
            quali_var_sup_cluster.name = "cluster"
            
            # cluster's members : distance own cluster, distance nex closest, ratio (own/next)
            cluster_member_sup = clust_member(D=dist_sup_cluster,cluster=quali_var_sup_cluster,method="min")
            
            #convert to dictionary
            quali_var_sup_ = {"cluster":quali_var_sup_cluster,"dist":dist_sup_cluster,"member":cluster_member_sup}
            #convert to namedtuple
            self.quali_var_sup_ = namedtuple("quali_var_sup",quali_var_sup_.keys())(*quali_var_sup_.values())

        return self
    
    def fit_predict(self,X,y=None):
        """Fit and return the result of each column clustering assignment.

        In addition to fitting, this method also return the result of the clustering assignment for each column in the training set.
        Equivalent to fit(X).predict(X), but more efficiently implemented.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : Series of shape (n_columns,)
            Labels of the cluster each column belongs to.
        """
        self.fit(X)
        return self.quali_var_.cluster

    def fit_transform(self,X,y=None):
        """Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            Training data, where ``n_samples`` in the number of samples 
            and ``n_columns`` is the number of columns.

        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        X_new : DataFrame of shape (n_columns, ncl)
            X transformed in the new space.
        """
        self.fit(X)
        return self.quali_var_.dist
    
    def predict(self,X):
        """Predict the closest cluster each column in X belongs to.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            New data, where ``n_samples`` is the number of samples 
            and ``n_columns`` is the number of columns.

        Returns
        -------
        labels : Series of shape (n_columns,)
            Labels of the cluster each column belongs to.
        """
        # distance for new data points to cluster centers
        dist = self.transform(X)
        # assign cluster to new categorical variables
        cluster = dist.idxmin(axis=1).astype("category")
        cluster.name = "cluster"
        return cluster

    def transform(self,X):
        """
        Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            New data, where ``n_samples`` is the number of samples 
            and ``n_columns`` is the number of columns.

        Returns
        -------
        X_new : DataFrame of shape (n_columns, ncl) 
            X transformed in the new space.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # convert to pd.DataFrame if pd.Series
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if isinstance(X,Series):
            X = X.to_frame()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if X is an object of class pd.DataFrame
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not isinstance(X,DataFrame):
            raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame.",
                            "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set index name as None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X.index.name = None

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # drop level if ndim greater than 1 and reset columns name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if convient row shape
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.shape[0] != self.call_.X.shape[0]:
            raise ValueError("Inconvenient row length")
        
        # similarity between active variables and supplementay variables
        S = DataFrame(index=X.columns,columns=self.call_.X.columns).astype("float")
        for i in X.columns:
            for j in self.call_.X.columns:
                S.loc[i,j] = simqual(x=X[i],y=self.call_.X[j],method=self.method,normalize=self.normalize)
        # dissimilarity matrix
        D = S.transform(func=self.metric)
    
        # distance to cluster centers
        dist_cluster = clust_dist(D=D,cluster=self.quali_var_.cluster,linkage=self.linkage)
        return dist_cluster