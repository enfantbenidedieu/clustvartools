# -*- coding: utf-8 -*-
from numpy import zeros,ones, repeat, array, where, c_
from pandas import Series, DataFrame, concat
from pandas.api.types import is_numeric_dtype
from collections import namedtuple
from scipy.cluster.hierarchy import fcluster
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# interns functions
from .functions.get_sup_label import get_sup_label
from ..others._clust_diss import clust_diss2
from ..others._getnnsvar import getnnsvar
from ..others._clust_member import clust_member

class CorCLV(BaseEstimator,TransformerMixin):
    """
    Hierarchical Clustering of Variables from a covariance/correlation matrix (CorCLV)
    
    Performs ascendant hierarchical clustering analysis of a set of continuous variables from a covariance/correlation matrix.
    Supplementary continuous variables may be used.
    
    Parameters
    ----------
    ncl : int, default = 2
        If a (positive) integer, the tree is cut with ncl clusters. If None, then the ncl is determine using optimal point.
    
    kind : {'cor', 'cov'}, default = 'cor'
        Type of matrix:
        
        * "cor" : correlation matrix
        * "cov" : covariance matrix

    method : {"pearson","kendall","spearman"} or callable, default = "pearson"
        Method of correlation :

        * "pearson" : standard correlation coefficient
        * "kendall" : Kendall Tau correlation coefficient
        * "spearman" : Spearman rank correlation
        * callable : callable with input two 1d ndarrays
            and returning a float. Note that the returned matrix from corr will have 1 along the diagonals 
            and will be symmetric regardless of the callable's behavior.

    precomputed : bool, default = True
        If True, pre-computed similarities matrices are passed directly to ``fit`` and ``fit_transform``.

    sup_var : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary continuous variables.
        
    Attributes
    ----------
    call_ : call
        An object containing the summary called parameters with the following attributes:
        
        Xtot : DataFrame of shape (n_samples, n_columns + n_columns_sup) or (n_columns + n_columns_sup , n_columns + n_columns_sup)
            Input data.
        X : DataFrame of shape (n_samples, n_columns) or (n_columns, n_columns)
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
            The names of the supplementary variables (continuous and/or categorical).

    quanti_var_ : quanti_var
        An object with the following attributes:
        
        cluster : Series of shape (n_columns,)
            Index of the cluster each column belongs to.
        diss : DataFrame of shape (n_columns, ncl)
            Dissimilarities from variables to clusters.
        member : DataFrame of shape (n_columns, 4)
            Cluster members and dissimilarities values.
            
    quanti_var_sup_ : quanti_var_sup, optional
        An object with the following attributes:
        
        cluster : Series of shape (n_columns_sup,)
            Index of the cluster each supplementary column belongs to.
        diss : DataFrame of shape (n_columns_sup, ncl)
            Dissimilarities from supplementary variables to cluster.
        member : DataFrame of shape (n_columns_sup, 4)
            Cluster members and dissimilarities values.

    References
    ----------
    [1] E. Vigneau, M. Qannari. `Clustering of variables around latent components <https://www.researchgate.net/publication/243044470_Clustering_of_Variables_Around_Latent_Components>`_. in Statistics, Simulation and Computation, 32(4), pp.1131-1150, 2003.

    [2] M. Chavent, V. Kuentz Simonet, B. Liquet, J. Saracco. `ClustOfVar: An R package for the Clustering of Variables <https://www.jstatsoft.org/article/view/v050i13>`_. in Journal of Statistical Software, 50(13), september 2012.

    [3] Rakotomalala, R. `Classification de variables : classification autour de variables latentes <https://eric.univ-lyon2.fr/ricco/cours/slides/classification_de_variables.pdf>`_. Tutoriel Tanagra pour le Data Mining
    
    Examples
    --------
    >>> from clustvartools.datasets import decathlon
    >>> from clustvartools import CorCLV
    >>> clf = CorCLV(ncl=3)
    >>> clf.fit(decathlon.actif)
    CorCLV(ncl=3)
    """
    def __init__(
        self,
        ncl = 2,
        kind = "cor",
        method = "pearson",
        precomputed = False,
        sup_var = None
    ):
        self.ncl = ncl
        self.kind = kind
        self.method = method
        self.precomputed = precomputed
        self.sup_var = sup_var
        
    def fit(self,X,y=None):
        """
        Compute CorCLV
        
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
        # check if kind is valid
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (self.kind in ("cor","cov")):
            raise ValueError("kind should be one of 'cor', 'cov'")
        
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
        # check if X contains only numerics columns
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not all(is_numeric_dtype(X[c]) for c in X.columns):
            raise TypeError("Columns in X must be numerics.")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if squared matrix for precomputed
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.precomputed and X.shape[0] != X.shape[1]:
            raise TypeError("If precomputed, X must be a square matrix.")
                
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # get supplementary variables labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # get supplementary variables labels
        sup_var_label = get_sup_label(X=X, indexes=self.sup_var, axis=1)
            
        # make a copy of the original data
        Xtot = X.copy()

        # drop supplementary variables columns - columns axis
        if self.sup_var is not None:
            X = X.drop(columns=sup_var_label)
            
        # drop supplementary columns - rows axis
        if self.precomputed:
            X = X.drop(index=sup_var_label)
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # data preparation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set number of columns
        n_cols = X.shape[1]
        # compute similarity matrix - correlation/covariance matrix
        if self.precomputed:
            S = X.copy()
        else:
            if self.kind == "cor":
                S = X.corr(method=self.method)
            else:
                S = X.cov(ddof=0)
            
        # compute dissimilarity matrix
        D = DataFrame(zeros((n_cols,n_cols)),index=X.columns,columns=X.columns)
        for i in range(n_cols-1):
            for j in range(i+1,n_cols):
                D.iloc[i,j] = clust_diss2(R=S,id1=i,id2=j)
                D.iloc[j,i] = D.iloc[i,j]
        
        # make a copy of dissimilarity matrix
        Dprim = D.copy()
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ascendant hierarchical cluster analysis from covariance/correlation matrix
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # initial values
        init = list(range(n_cols))      # initial partition
        init_crit = repeat(1,n_cols)    # initial criteria
        maxv = 1e12                     # maximum
        a = zeros((n_cols-1))           
        b = zeros((n_cols-1))           
        ia = zeros((n_cols-1))          # Le premier cluster ou point d'origine sélectionné pour la fusion.
        ib = zeros((n_cols-1))          # Le deuxième cluster ou point d'origine sélectionné pour la fusion.
        lev = zeros((n_cols-1))         # la hauteur entre les deux clusters fusionnés
        card = ones((n_cols))           # cardinalities
        size = zeros((n_cols-1))        # Le nombre total d'observations (points initiaux) contenues dans ce nouveau cluster ainsi formé.
        
        # compute nearest neighbor of variables
        nnsvar = getnnsvar(D=D,crit=init_crit)
        # clust matrix
        clustmat = zeros((n_cols,n_cols))
        for i in range(n_cols):
            clustmat[i,n_cols-1] = i
        for ncl in range(n_cols-2,-1,-1):
            # check for agglomerable pair
            minobs = -1
            mindis = maxv
            for i in range(n_cols):
                if init_crit[i] == 1:
                    if nnsvar.nndiss[i] < mindis:
                        mindis = nnsvar.nndiss[i]
                        minobs = i      
            # find agglomerands clus1 and clus2, with former < latter
            if minobs < nnsvar.nn[minobs]:
                cl1 = minobs
                cl2 = nnsvar.nn[minobs]
            elif minobs > nnsvar.nn[minobs]:
                cl2 = minobs
                cl1 = nnsvar.nn[minobs]
            # convert to integer
            cl1, cl2 = int(cl1), int(cl2)
            id1 = [i for i,x in enumerate(clustmat[:,ncl+1]) if x == cl1]
            id2 = [i for i,x in enumerate(clustmat[:,ncl+1]) if x == cl2]
            A = [i for i in init if i in id1]
            B = [i for i in init if i in id2]
            clus = [*A,*B]
            # assign to leaf
            a[ncl], b[ncl], size[ncl] = cl1, cl2, len(clus)
            if card[cl1] == 1:
                ia[ncl] = - cl1
            if card[cl2] == 1:
                ib[ncl] = - cl2
            # 
            if card[cl1] > 1:
                last_ind = 0
                for i2 in range(n_cols-2,ncl,-1):
                    if a[i2] == cl1:
                        last_ind = i2
                ia[ncl] = n_cols - last_ind - 1 
            if card[cl2] > 1:
                last_ind = 0
                for i2 in range(n_cols-2,ncl,-1):
                    if a[i2] == cl2:
                        last_ind = i2
                ib[ncl] = n_cols - last_ind - 1
            # 
            if ia[ncl] > 0 or ib[ncl] > 0:
                l = min(ia[ncl],ib[ncl])
                if l > 0:
                    l -= 1 
                r = max(ia[ncl],ib[ncl])
                if r > 0:
                    r -= 1 
                ia[ncl], ib[ncl] = l,r
            #
            lev[ncl] = mindis
            for i in range(n_cols):
                clustmat[i,ncl] = clustmat[i,ncl+1]
                if clustmat[i,ncl] == cl2:
                    clustmat[i,ncl] = cl1
            # update dissimilarity matrix cluster 1
            for i in range(n_cols):
                if (i != cl1) and (i != cl2) and (init_crit[i] == 1):
                    A = [j for j in init if j in [k for k,x in enumerate(clustmat[:,ncl+1]) if x == i]]
                    D.iloc[cl1,i] = clust_diss2(R=S,id1=clus,id2=A)
                    D.iloc[i,cl1] = D.iloc[cl1,i]
            card[cl1] = card[cl1] + card[cl2]
            init_crit[cl2] = 0
            nnsvar.nndiss[cl2] = maxv
            # update dissimilarity matrix cluster 2
            for i in range(n_cols):
                D.iloc[cl2,i] = maxv
                D.iloc[i,cl2] = D.iloc[cl2,i]
            nnsvar = getnnsvar(D=D,crit=init_crit)
            
        # reverse
        merge = array([ia,ib]).T[::-1]
        # convert to python scipy 
        Z = zeros((n_cols-1,4))
        Z[:,0] = where(merge[:,0] < 0,-merge[:,0],where(merge[:,0]==0,0,merge[:,0]+n_cols))
        Z[:,1] = where(merge[:,1] < 0,-merge[:,1],where(merge[:,1]==0,n_cols,merge[:,1]+n_cols))
        Z[:,2] = lev[::-1]
        Z[:,3] = size[::-1]
        
        # height
        height = DataFrame(c_[list(range(1,Z.shape[0]+1)),Z[:,2][::-1]],columns=["cluster","height"])
        height["diff_1"] = -1*height["height"].diff(1)
        height["diff_2"] = height["diff_1"].diff(-1)
        height["cluster"] = height["cluster"].astype("int")

        # convert to dictionary
        tree_ = {"S":S,"D":Dprim,"Z":Z,"height":height,"merge":Z[:,:2],"size":Z[:,3]}
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
            
        #convert to dictionary
        call_ = {"Xtot":Xtot,"X":X,"ncl":ncl,"tree":tree,"sup_var":sup_var_label}
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Informations for variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # assign cluster
        cluster = Series(fcluster(Z,t=ncl,criterion="maxclust"), index = D.index, name = "cluster",dtype="category")
        # unique cluster
        uq_cluster = sorted(cluster.unique())
        # dissimilarity from a variable to a cluster
        diss_cluster = DataFrame(index=D.index,columns=uq_cluster).astype("float")
        for l in D.index:
            for k in uq_cluster:
                clus = list(cluster[cluster==k].index)
                diss_cluster.loc[l,k] = clust_diss2(R=S,id1=l,id2=clus)
        
        # cluster's members : distance own cluster, distance next closest, ratio (own/next)
        cluster_member = clust_member(D=diss_cluster,cluster=cluster,method="min")
          
        #convert to dictionary
        quanti_var_ = {"cluster":cluster,"diss":diss_cluster,"member":cluster_member}
        #convert to namedtuple
        self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # clusters for supplementary continuous variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            # similarity matrix between all variables
            if self.precomputed:
                Stot = Xtot.copy()
            else:
                if self.kind == "cor":
                    Stot = Xtot.corr(method=self.method)
                else:
                    Stot = X.cov(ddof=0)
            # dissimilarity of supplementary variables to cluster centers          
            diss_cluster_sup = DataFrame(index=sup_var_label,columns=uq_cluster).astype("float")
            for l in sup_var_label:
                for k in uq_cluster:
                    clus = list(cluster[cluster==k].index)
                    diss_cluster_sup.loc[l,k] = clust_diss2(R=Stot,id1=l,id2=clus)
            
            # assign cluster to supplementary variables
            cluster_sup = diss_cluster_sup.idxmin(axis=1).astype("category")
            cluster_sup.name = "cluster"
            
            # cluster's members : ward distance own cluster, ward distance nex closest, ratio (own/next)
            cluster_member_sup = clust_member(D=diss_cluster_sup,cluster=cluster_sup,method="min")
            
            # convert to dictionary
            quanti_var_sup_ = {"cluster":cluster_sup,"diss":diss_cluster_sup,"member":cluster_member_sup}
            # convert to namedtuple
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
        """Compute clustering and transform X to cluster-dissimilarity space.

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
        return self.quanti_var_.diss
    
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
        # ward distance for new data points to cluster centers
        dist = self.transform(X)
        # assign cluster to new data points
        cluster = dist.idxmin(axis=1).astype("category")
        cluster.name = "cluster"
        return cluster
        
    def transform(self,X):
        """Transform X to a cluster-dissimilarity space.

        In the new space, each dimension is the dissimilarity to the cluster centers

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
        if not self.precomputed:
            if X.shape[0] != self.call_.X.shape[0]:
                raise ValueError("Inconvenient row length")
            # concatenate
            Xtot = concat((self.call_.X,X),axis=1)
            # similarity matrix between all variables
            if self.kind == "cor":
                Stot = Xtot.corr(method=self.method)
            else:
                Stot = Xtot.cov(ddof=0)
        else:
            # check if X contains correlation between new data
            if X.shape[0] != (self.call_.X.shape[1] + X.shape[1]):
                raise ValueError("Inconvenient row length")
            # data preparation
            A, B = self.call_.tree.S, X.loc[X.columns,:]
            C = X.drop(index=X.columns)
            # concatenate
            Stot = concat((A,B),axis=0)
            Stot.loc[self.call_.X.columns,X.columns] = C
            Stot.loc[X.columns,self.call_.X.columns] = C.T
        
        # cluster
        cluster = self.quanti_var_.cluster
        uq_cluster = sorted(cluster.unique())
        # dissimilarity of new data to cluster
        diss_cluster = DataFrame(index=X.columns,columns=uq_cluster).astype("float")
        for l in X.columns:
            for k in uq_cluster:
                clus = list(cluster[cluster==k].index)
                diss_cluster.loc[l,k] = clust_diss2(R=Stot,id1=l,id2=clus)
        return diss_cluster