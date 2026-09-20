# -*- coding: utf-8 -*-
from numpy import sqrt, c_
from pandas import Series, concat, DataFrame
from collections import namedtuple
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#interns functions
from .functions.preprocessing import preprocessing
from .functions.get_sup_label import get_sup_label
from ..others._clust_dist import clust_dist
from ..others._clust_member import clust_member

class HCAV(BaseEstimator,TransformerMixin):
    """
    Hierarchical Clustering Analysis for Variables (HCAV)
    
    Performns a standard hierarchical agglomerative clustering analysis on continuous variables using correlation matrix.
    
    Parameters
    ----------
    ncl : int, default = 2
        If a (positive) integer, the tree is cut with ncl clusters. If None, then the ncl is determine using using optimal point.

    method : {"pearson","kendall","spearman"} or callable, default = "pearson"
        Method of correlation :

        * "pearson" : standard correlation coefficient
        * "kendall" : Kendall Tau correlation coefficient
        * "spearman" : Spearman rank correlation
        * callable : callable with input two 1d ndarrays
            and returning a float. Note that the returned matrix from corr will have 1 along the diagonals 
            and will be symmetric regardless of the callable's behavior.

    metric : callable, default = lambda x : sqrt(1 - x**2)
        Callable with input a float. This refers to how similarity matrix :math:`S` is converting into dissimilarity matrix :math:`D`.

    linkage : {"average", "complete", "single", "ward"}, default = "ward"
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        * "average" uses the average of the distances of each observation of the two sets.
        * "complete" linkage uses the maximum distances between all observations of the two sets.
        * "single" uses the minimum of the distances between all observations of the two sets.
        * "ward" minimizes the variance of the clusters being merged.

    sup_var : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary continuous variables.
    
    Attributes
    ----------
    call_ : call
        An object containing the summary called parameters with the following attributes:

        Xtot : DataFrame of shape (n_samples, n_columns + n_columns_sup)
            Input data.
        X : DataFrame of shape (n_samples, n_columns)
            Active data.
        ncl : int
            The number of clusters kepted.
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
            The names of the supplementary continuous variables.

    quanti_var_ : quanti_var
        An object containing the description of the clusters by the continuous variables, with following attributes:

        cluster : Series of shape (n_columns,)
            The labels of each continuous variable.
        dist : DataFrame of shape (n_columns, ncl)
            The distance of each continuous variable to the cluster centers.
        member : DataFrame of shape (n_columns, 3)
            Cluster's members of each continuous variable (distance to own cluster, distance to next closest, ratio (own/next)).

    quanti_var_sup_ : quanti_var_sup, optional
        An object containing the description of the clusters by the supplementary continuous variables, with following attributes:

        cluster : Series of shape (n_columns_sup,)
            The labels of each supplementary continuous variable.
        dist : DataFrame of shape (n_columns_sup, ncl)
            The distance of each supplementary continuous variable to the cluster centers.
        member : DataFrame of shape (n_columns_sup, 3)
            Cluster's members of each supplementary continuous variable (distance to own cluster, distance to next closest, ratio (own/next)).

    References
    ----------
    [1] M. Qannari, E. Vigneau, P. Courcoux (1998). `Une nouvelle distance entre variables. Application en classification <https://www.numdam.org/item/RSA_1998__46_2_21_0.pdf>`_. Revue de Statistique Appliquée, vol. 46, n°2, pp. 21-32.
    
    [2] Rakotomalala, R. `Classification de variables : classification autour de variables latentes <https://eric.univ-lyon2.fr/ricco/cours/slides/classification_de_variables.pdf>`_. Tutoriel Tanagra pour le Data Mining
    
    [3] Rakotomalala, R. (2023). Classification de Variables Quantitatives (Python), `Tutoriel Youtube <https://www.youtube.com/watch?v=Wwq20qxS2E8>`_.
    
    Examples
    --------
    >>> from clustvartools.datasets import jobrate
    >>> from clustvartools import HCAV
    >>> clf = HCAV(ncl=4,sup_var=13)
    >>> clf.fit(jobrate)
    HCAV(ncl=4,sup_var=13)
    """
    def __init__(
            self, 
            ncl = 2, 
            method = "pearson", 
            metric = lambda x : sqrt(1 - x**2), 
            linkage = "ward", 
            sup_var = None
    ):
        self.ncl = ncl
        self.method = method
        self.metric = metric
        self.linkage = linkage
        self.sup_var = sup_var

    def fit(self,X,y=None):
        """Compute HCAV
        
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
        # check if linkage criterion is valid
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (self.linkage in ("average","complete","single","ward")):
            raise ValueError("linkage should be one of 'average', 'complete', 'single', 'ward'")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # preprocessing (drop level, fill NA with mean, convert to ordinal levels)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set sup_var_label
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # get supplementary continuous variables labels
        sup_var_label = get_sup_label(X=X, indexes=self.sup_var, axis=1)
            
        # make a copy of the original data
        Xtot = X.copy()

        # drop supplementary continuous variables columns
        if self.sup_var is not None:
            X_sup_var, X = X.loc[:,sup_var_label], X.drop(columns=sup_var_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # variables hierarchical clustering
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # compute the similarity matrix - correlation matrix
        S = X.corr(method=self.method)
        # compute dissimilary matrix
        D = S.transform(func=self.metric)
        # linkage matrix with vectorize dissimilarity matrix
        Z = sch.linkage(squareform(D,checks=False),method=self.linkage)

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
            raise TypeError("nclust should be a positive integer.")
        elif not isinstance(self.ncl,int):
            raise TypeError("ncl should be an integer")
        else:
            ncl = self.ncl

        #convert to dictionary
        call_ = {"Xtot":Xtot,"X":X,"ncl":ncl,"tree":tree,"sup_var":sup_var_label}
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Informations for levels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # assign cluster
        cluster = Series(sch.fcluster(Z,t=ncl,criterion="maxclust"), index = D.index, name = "cluster",dtype="category")
        
        # distance to cluster centers
        dist_cluster = clust_dist(D=D,cluster=cluster,linkage=self.linkage)
                
        # cluster's members : distance own cluster, distance next closest, ratio (own/next)
        cluster_member = clust_member(D=dist_cluster,cluster=cluster,method="min")
        
        # convert to dictionary
        quanti_var_ = {"cluster":cluster, "dist":dist_cluster,"member":cluster_member}
        # convert to namedtuple
        self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # clusters for supplementary continuous variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            # similarity matrix between active variables and supplementary variables
            if X.shape[1] > X_sup_var.shape[1]:
                S_sup = concat((X.corrwith(other=X_sup_var[k],method=self.method,axis=0).to_frame(k) for k in sup_var_label),axis=1).T
            else:
                S_sup = concat((X_sup_var.corrwith(other=X[k],method=self.method,axis=0).to_frame(k) for k in X.columns),axis=1)
            # dissimilarity matrix
            D_sup = S_sup.transform(func=self.metric) 
            # distance to cluster centers          
            dist_sup_cluster = clust_dist(D=D_sup,cluster=cluster,linkage=self.linkage) 
            
            # assign cluster
            quanti_var_sup_cluster = dist_sup_cluster.idxmin(axis=1).astype("category")
            quanti_var_sup_cluster.name = "cluster"
            
            # cluster's members : distance own cluster, distance nex closest, ratio (own/next)
            cluster_member_sup = clust_member(D=dist_sup_cluster,cluster=quanti_var_sup_cluster,method="min")
    
            #convert to ordered dictionary
            quanti_var_sup_ = {"cluster":quanti_var_sup_cluster,"dist":dist_sup_cluster,"member":cluster_member_sup}
            #convert to namedtuple
            self.quanti_var_sup_ = namedtuple("quanti_var_sup",quanti_var_sup_.keys())(*quanti_var_sup_.values())

        return self
    
    def fit_predict(self,X,y=None):
        """Compute cluster centers and predict cluster index for each column.

        Convenience method; equivalent to calling fit(X) followed by predict(X).

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : Series of shape (n_columns,)
            Index of the cluster each column belongs to.
        """
        self.fit(X)
        return self.quanti_var_.cluster

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
        return self.quanti_var_.dist
    
    def predict(self,X):
        """Predict the closest cluster each column in X belongs to.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            New data to predict, where ``n_samples`` is the number of samples 
            and ``n_columns`` is the number of columns.

        Returns
        -------
        labels : Series of shape (n_columns,)
            Labels of the cluster each column belongs to.
        """
        # distance for new data points to cluster centers
        dist = self.transform(X)
        # assign cluster to new data points
        cluster = dist.idxmin(axis=1).astype("category")
        cluster.name = "cluster"
        return cluster
        
    def transform(self,X):
        """Transform X to a cluster-distance space.

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
        # check if convenient row shape
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.shape[0] != self.call_.X.shape[0]:
            raise ValueError("Inconvenient row length")
        
        # similarity matrix
        if self.call_.X.shape[1] > X.shape[1]:
            S = concat((self.call_.X.corrwith(other=X[k],method=self.method,axis=0).to_frame(k) for k in X.columns),axis=1).T
        else:
            S = concat((X.corrwith(other=self.call_.X[k],method=self.method,axis=0).to_frame(k) for k in self.call_.X.columns),axis=1)
        # dissimilarity matrix
        D = S.transform(func=self.metric)
        # distance to cluster centers
        dist_cluster = clust_dist(D=D,cluster=self.quanti_var_.cluster,linkage=self.linkage)
        return dist_cluster