# -*- coding: utf-8 -*-
from numpy import sqrt, c_, sum
from pandas import concat, get_dummies, DataFrame, Series
from collections import namedtuple
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#intern functions
from .functions.preprocessing import preprocessing
from .functions.get_sup_label import get_sup_label
from .functions.cathca import catsimdissim
from .functions.revalue import revalue
from ..others._clust_dist import clust_dist
from ..others._clust_member import clust_member

class CatVARHCA(BaseEstimator,TransformerMixin):
    """
    Categorical Variables Hierarchical Clustering Analysis (CatVARHCA)
    
    Performs hierarchical agglomerative clustering algorithm for levels of categorical variables. 
    Supplementary categorical variables may be used.

    Parameters
    ----------
    ncl : int, default = 3
        If a (positive) integer, the tree is cut with ncl clusters. If None, then the ncl is determine using using optimal point.
    
    method : {"dice", "bothpos"}, default = "dice"
        The similarity method.
    
    linkage : {"average", "complete", "single", "ward"}, default = "average"
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        * "average" uses the average of the distances of each observation of the two sets.
        * "complete" linkage uses the maximum distances between all observations of the two sets.
        * "single" uses the minimum of the distances between all observations of the two sets.
        * "ward" minimizes the variance of the clusters being merged. 
    
    sup_var : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary categorical columns/variables.

    Attributes
    ----------
    call_ : call
        An object containing the summary called parameters with the following attributes:

        Xtot : DataFrame of shape (n_samples, n_columns + n_columns_sup)
            Input data.
        X : DataFrame of shape (n_samples, n_columns)
            Active data.
        M : DataFrame of shape (n_samples, n_levels)
            Disjunctive table.
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

    levels_ : levels
        An object containing the description of the clusters by the levels:

        cluster : Series of shape (n_levels,)
            The labels of each level.
        dist : DataFrame of shape (n_levels, ncl)
            The distance of each level to the cluster centers.
        member : DataFrame of shape (n_levels, 3)
            Cluster's members of each level (distance to own cluster, distance to next closest, ratio (own/next)).

    levels_sup_ : levels_sup, optional
        An object containing the description of the clusters by the supplementary levels:

        cluster : Series of shape (n_levels_sup,)
            The labels of each supplementary levels.
        dist : DataFrame of shape (n_levels_sup, ncl)
            The distance of each supplementary level to the cluster centers.
        member : DataFrame of shape (n_levels_sup, 3)
            Cluster's members of each supplementary level (distance to own cluster, distance to next closest, ratio (own/next)).

    References
    ----------
    [1] H. Abdallah, G. Saporta. (1998). `Classification d'un ensemble de variables qualitatives <https://www.numdam.org/item/RSA_1998__46_4_5_0.pdf>`_. in Revue de Statistique Appliquée, Tome 46, N°4, pp. 5-26
    
    [2] Rakotomalala, R. `Classification des variables qualitatives : Regroupement de variables, regroupement de modalités <https://eric.univ-lyon2.fr/ricco/cours/slides/classif_variables_quali.pdf>`_. Université Lumière Lyon 2.
    
    [3] Rakotomalala, R. (2013). `Classification automatique de variables (de modalités de variables) catégorielles <https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_Cat_Variable_Clustering.pdf>`_. Tutoriel Tanagra pour le Data Mining

    [4] Rakotomalala, R. (2023). Clustering Variables Qualitatives - Classification des modalités, `Tutoriel Youtube <https://www.youtube.com/watch?v=QVK3i44oGx8>`_.

    Examples
    --------
    >>> from clustvartools.datasets import voting
    >>> from clustvartools import CatVARHCA
    >>> clf = CatVARHCA(ncl=None,sup_var=0)
    >>> clf.fit(voting)
    CatVARHCA(ncl=None,sup_var=0)
    """
    def __init__(
            self, 
            ncl = 2, 
            method = "dice", 
            linkage = "average", 
            sup_var = None
    ):
        self.ncl = ncl
        self.method = method
        self.linkage = linkage
        self.sup_var = sup_var

    def fit(self,X,y=None):
        """Compute CatVARHCA

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
        if not (self.method in ("dice","bothpos")):
            raise ValueError("method should be one of 'dice', 'bothpos'")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if linkage method is valid
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (self.linkage in ("average","complete","single","ward")):
            raise ValueError("linkage should be one of 'average', 'complete', 'single', 'ward'")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # preprocessing (drop level, fill NA with mean, convert to ordinal levels)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X)
    
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # get supplementary categorical variables labels
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
        # set number of rows and number of columns
        n_rows = X.shape[0]
        # disjunctive table
        M =  concat((get_dummies(X[j],dtype=int) for j in X.columns),axis=1)
        # set number of categories
        n_levels = M.shape[1]
        # Compute similarity and dissimilarity matrices
        if self.method == "dice":
            # similarity matrix
            S = None
            # dissimilarity matrix
            D = DataFrame(sqrt(squareform(pdist(M.T,metric="sqeuclidean"))/2),index=M.columns,columns=M.columns)
        else:
            # similarity matrix
            S = DataFrame(index=M.columns,columns=M.columns).astype("float")
            for i in range(n_levels-1):
                S.iloc[i,i] = sum(M.iloc[:,i].values)/n_rows
                for j in range(i+1,n_levels):
                    S.iloc[i,j] = catsimdissim(x=M.iloc[:,i].values,y=M.iloc[:,j].values,method="bothpos")
                    S.iloc[j,i] = S.iloc[i,j]
            S.iloc[n_levels-1,n_levels-1] = sum(M.iloc[:,n_levels-1].values)/n_rows
            # dissimilarity matrix
            D = 1 - S

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # agglomerative hierarchical clustering (HCA)
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

        #convert to ordered dictionary
        call_ = {"Xtot":Xtot,"X":X,"M":M,"ncl":ncl,"tree":tree,"sup_var":sup_var_label}
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # informations for levels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # assign cluster
        cluster = Series(sch.fcluster(Z,t=ncl,criterion="maxclust"), index = D.index,name = "cluster",dtype="category")
    
        # distance for levels to the cluster
        dist_cluster = clust_dist(D=D,cluster=cluster,linkage=self.linkage)
    
        # cluster's members : distance own cluster, distance next closest, ratio (own/next)
        cluster_member = clust_member(D=dist_cluster,cluster=cluster,method="min")
        
        # convert to ordered dictionary
        levels_ = {"cluster":cluster,"dist":dist_cluster,"member":cluster_member}
        # convert to namedtuple
        self.levels_ = namedtuple("levels",levels_.keys())(*levels_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #clusters for supplementary levels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            # disjunctive table for supplementary variables
            M_sup = concat((get_dummies(X_sup_var[j],dtype=int) for j in sup_var_label),axis=1)
            # compute similarity/dissimilarity matrix
            D_sup = DataFrame(index=M_sup.columns,columns=M.columns).astype("float")
            for i in M_sup.columns:
                for j in M.columns:
                    D_sup.loc[i,j] = catsimdissim(x=M_sup[i].values,y=M[j].values,method=self.method)
            if self.method == "bothpos":
                D_sup = 1 - D_sup
            
            # distance for supplementary levels to the cluster centers
            dist_sup_cluster = clust_dist(D=D_sup,cluster=cluster,linkage=self.linkage)

            # assign cluster to supplementary levels
            levels_sup_cluster = dist_sup_cluster.idxmin(axis=1).astype("category")
            levels_sup_cluster.name = "cluster"
            
            # cluster's members : distance own cluster, distance next closest, ratio (own/next)
            cluster_member_sup = clust_member(D=dist_sup_cluster,cluster=levels_sup_cluster,method="min")
            
            # convert to dictionary
            levels_sup_ = {"cluster":levels_sup_cluster,"dist":dist_sup_cluster,"member":cluster_member_sup}
            # convert to namedtuple
            self.levels_sup_ = namedtuple("levels_sup",levels_sup_.keys())(*levels_sup_.values())

        return self
    
    def fit_predict(self,X,y=None):
        """Fit and return the result of each level clustering assignment.

        In addition to fitting, this method also return the result of the clustering assignment for each level in the training set.
        Equivalent to fit(X).predict(X), but more efficiently implemented.
        
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : Series of shape (n_levels,)
            Index of the cluster each level belongs to.
        """
        self.fit(X)
        return self.levels_.cluster

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
        X_new : DataFrame of shape (n_levels, ncl)
            X transformed in the new space.
        """
        self.fit(X)
        return self.levels_.dist
    
    def predict(self,X):
        """Predict the closest cluster each level in X belongs to.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            New data to predict, where ``n_samples`` is the number of samples 
            and ``n_columns`` is the number of columns.

        Returns
        -------
        labels : Series of shape (n_levels,)
            Labels of the cluster each level belongs to.
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
        X_new : DataFrame of shape (n_levels, ncl) 
            X transformed in the new space.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if the estimator is fitted by verifying the presence of fitted attributes
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
        #drop level if ndim greater than 1 and reset columns name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if convient row shape
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.shape[0] != self.call_.X.shape[0]:
            raise ValueError("Inconvenient row length")
        
        # revaluate columns
        X = revalue(X=X)

        # disjunctive table for new data
        M = concat((get_dummies(X[j],dtype=int) for j in X.columns),axis=1)
        # compute dimilarity matrix
        D = DataFrame(index=M.columns,columns=self.call_.M.columns).astype("float")
        for i in M.columns:
            for j in self.call_.M.columns:
                D.loc[i,j] = catsimdissim(x=M[i].values,y=self.call_.M[j].values,method=self.method)
        if self.method == "bothpos":
            D = 1 - D
        
        # distance from new data to cluster centers
        dist_cluster = clust_dist(D=D,cluster=self.levels_.cluster,linkage=self.linkage)
        return dist_cluster