# -*- coding: utf-8 -*-
from numpy import sqrt, c_, corrcoef, diag, ones
from pandas import Series,DataFrame, crosstab
from pandas.api.types import is_numeric_dtype
from collections import namedtuple
from scipy.stats.contingency import association
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#interns functions
from .functions.preprocessing import preprocessing
from .functions.get_sup_label import get_sup_label
from .functions.statistics import eta2
from ..others._clust_dist import clust_dist
from ..others._clust_member import clust_member

class HCAVmix(BaseEstimator,TransformerMixin):
    """
    Hierarchical Clustering Analysis of Variables for Mixed Data (HCAVmix)
        
    Performns a standard hierarchical agglomerative clustering analysis of variables for mixed data.
    The similarity matrix is the squared of the cramer's V (for categorical variables);
    the squared of pearson correlation (for continuous variables) and the squared correlation ratio (for mixture).
    Dissimilarity matrix is defined by the user with the parameter ``metric``.
    Missing values are replaced by means for continuous variables and by most frequent for categorical variables.
    Supplementary variables (continuous and/or categorical) may be used.
    
    Parameters
    ----------
    ncl : int, default = 2
        If a (positive) integer, the tree is cut with ncl clusters. If None, then the ncl is determine using using optimal point.
        
    metric : callable, default = lambda x : sqrt(1 - x)
        Callable with input a float. This refers to how similarity matrix :math:`S` is converting into dissimilarity matrix :math:`D`.

    linkage : {"average", "complete", "single", "ward"}, default = "average"
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
            The names of the supplementary variables.

    var_ : var
        An object containing the description of the clusters by the variables, with following attributes:

        cluster : Series of shape (n_columns,)
            The labels of each variable.
        dist : DataFrame of shape (n_columns, ncl)
            The distance of each variable to the cluster centers.
        member : DataFrame of shape (n_columns, 3)
            Cluster's members of each variable (distance to own cluster, distance to next closest, ratio (own/next)).

    var_sup_ : var_sup, optional
        An object containing the description of the clusters by the supplementary variables, with following attributes:

        cluster : Series of shape (n_columns_sup,)
            The labels of each supplementary variable.
        dist : DataFrame of shape (n_columns_sup, ncl)
            The distance of each supplementary variable to the cluster centers.
        member : DataFrame of shape (n_columns_sup, 3)
            Cluster's members of each supplementary variable (distance to own cluster, distance to next closest, ratio (own/next)).

    Examples
    --------
    >>> from clustvartools.datasets import wine
    >>> from clustvartools import HCAVmix
    >>> clf = HCAVmix(ncl=6)
    >>> clf.fit(wine.data)
    HCAVmix(ncl=6)
    """
    def __init__(
        self,
        ncl = 2,
        metric = lambda x : sqrt(1 - x),
        linkage = "average",
        sup_var = None
    ):
        self.ncl = ncl
        self.metric = metric
        self.linkage = linkage
        self.sup_var = sup_var
    
    def fit(self,X,y=None):
        """Compute HCAVmix
        
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
        # check if linkage method is valid
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
        # number of columns
        n_cols = X.shape[1]
        # compute the similarity matrix
        S = DataFrame(diag(ones(n_cols)),index=X.columns,columns=X.columns).astype("float")
        for i in range(n_cols-1):
            for j in range(i+1,n_cols):
                k, l = X.columns[i], X.columns[j]
                is_num1 = is_numeric_dtype(X[k])
                is_num2 = is_numeric_dtype(X[l])
                if is_num1 and is_num2:
                    S.iloc[i,j] = corrcoef(X[k],X[l])[0,1]**2
                elif not is_num1 and not is_num2:
                    S.iloc[i,j] = association(crosstab(X[k],X[l]),method="cramer",correction=False)**2
                elif is_num1 and not is_num2:
                    S.iloc[i,j] = eta2(categories=X[l],values=X[k])
                elif not is_num1 and is_num2:
                    S.iloc[i,j] = eta2(categories=X[k],values=X[l])
        # compute dissimilary matrix
        D = S.transform(func=self.metric)
        # replace diagonal with 0
        for c in D.columns:
            D.loc[c,c] = 0
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
            raise TypeError("ncl should be a positive integer.")
        elif not isinstance(self.ncl,int):
            raise TypeError("ncl should be an integer")
        else:
            ncl = self.ncl

        #convert to ordered dictionary
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
        
        #convert to dictionary
        var_ = {"cluster":cluster, "dist":dist_cluster,"member":cluster_member}
        #convert to namedtuple
        self.var_ = namedtuple("var",var_.keys())(*var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # clusters for supplementary continuous variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            # similarity matrix between active variables and supplementary variables
            S_sup = DataFrame(index=sup_var_label,columns=X.columns).astype("float")
            for k in sup_var_label:
                for l in X.columns:
                    is_num1 = is_numeric_dtype(X_sup_var[k])
                    is_num2 = is_numeric_dtype(X[l])
                    if is_num1 and is_num2:
                        S_sup.loc[k,l] = corrcoef(X_sup_var[k],X[l])[0,1]**2
                    elif not is_num1 and not is_num2:
                        S_sup.loc[k,l] = association(crosstab(X_sup_var[k],X[l]),method="cramer",correction=False)**2
                    elif is_num1 and not is_num2:
                        S_sup.loc[k,l] = eta2(categories=X[l],values=X_sup_var[k])
                    elif not is_num1 and is_num2:
                        S_sup.loc[k,l] = eta2(categories=X_sup_var[k],values=X[l])
            # dissimilarity matrix
            D_sup = S_sup.transform(func=self.metric)
            # distance to cluster centers          
            dist_sup_cluster = clust_dist(D=D_sup,cluster=cluster,linkage=self.linkage)
            # assign cluster to supplementary variables
            var_sup_cluster = dist_sup_cluster.idxmin(axis=1).astype("category")
            var_sup_cluster.name = "cluster"
            # cluster's members : distance own cluster, distance nex closest, ratio (own/next)
            cluster_member_sup = clust_member(D=dist_sup_cluster,cluster=var_sup_cluster,method="min")
            
            #convert to dictionary
            var_sup_ = {"cluster":var_sup_cluster,"dist":dist_sup_cluster,"member":cluster_member_sup}
            #convert to namedtuple
            self.var_sup_ = namedtuple("var_sup",var_sup_.keys())(*var_sup_.values())

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
        return self.var_.cluster

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
        return self.var_.dist
    
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
        # check if convient row shape
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.shape[0] != self.call_.X.shape[0]:
            raise ValueError("Inconvenient row length")

        # similarity matrix
        S = DataFrame(index=X.columns,columns=self.call_.X.columns).astype("float")
        for k in X.columns:
            for l in self.call_.X.columns:
                is_num1 = is_numeric_dtype(X[k])
                is_num2 = is_numeric_dtype(self.call_.X[l])
                if is_num1 and is_num2:
                    S.loc[k,l] = corrcoef(X[k],self.call_.X[l])[0,1]**2
                elif not is_num1 and not is_num2:
                    S.loc[k,l] = association(crosstab(X[k],self.call_.X[l]),method="cramer",correction=False)**2
                elif is_num1 and not is_num2:
                    S.loc[k,l] = eta2(categories=self.call_.X[l],values=X[k])
                elif not is_num1 and is_num2:
                    S.loc[k,l] = eta2(categories=X[k],values=self.call_.X[l])
        # dissimilarity matrix
        D = S.transform(func=self.metric)
        # distance to cluster centers
        dist = clust_dist(D=D,cluster=self.var_.cluster,linkage=self.linkage)
        return dist