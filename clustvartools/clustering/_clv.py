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

class CLV(BaseEstimator,TransformerMixin):
    """
    Hierarchical Clustering of Variables around Latent Variables (CLV)
    
    Performns ascendant hierarchical clustering of a set of continuous variables. The aggregation criterion is the decrease 
    in homogeneity for the clusters being merged. The homogeneity of a cluster is the sum of the squared correlation 
    between the variables and the center of the cluster which is the first principal component of principal components analysis (PCA).
    Supplementary variables (continuous and/or categorical) may be used.
    
    Parameters
    ----------
    ncl : int, default = 2
        If a (positive) integer, the tree is cut with ncl clusters. If None, then the ncl is determine using optimal point.
        
    row_w : 1d array-like of shape (n_rows,), default = None
        An optional rows weights. The weights are given only for the active rows.
        
    sup_var : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary variables (continuous and/or categorical).
    
    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger 
        than `-tol*lambda1` where `lambda1` is the largest eigenvalue.
        
    Attributes
    ----------
    call_ : call
        An object containing the summary called parameters with the following attributes:
        
        Xtot : DataFrame of shape (n_samples, n_columns + n_columns_sup)
            Input data.
        X : DataFrame of shape (n_samples, n_columns)
            Active data.
        Z : DataFrame of shape (n_samples, n_columns) 
            Standardized data.
        center : Series of shape (n_columns,)
            The columns weighted average.
        scale : Series of shape (n_columns)
            The columns weighted standard deviation.
        row_w : Series of shape (n_samples,)
            The rows weights.
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

    cluster_ : cluster
        An object with the following attributes:
        
        summary : DataFrame of shape (ncl, 4)
            Cluster summary.
        cor : DataFrame of shape (ncl, ncl)
            Inter-cluster correlations.
        princomps : DataFrame of shape (n_samples, ncl)
            Synthetic variables , i.e cluster centers or latent variables.

    coef_ : coef
        An object with the following attributes:
        
        coef : DataFrame of shape (n_columns + 1, ncl)
            Coefficients of linear combinations defining the synthetic variable of each cluster.
        coef_std : DataFrame of shape (n_columns, ncl)
            Standardized scoring coefficients.
    
    quanti_var_ : quanti_var
        An object with the following attributes:
        
        cluster : Series of shape (n_columns,)
            Index of the cluster each column belongs to.
        cor : DataFrame of shape (n_columns, ncl)
            Pearson correlation of variables and latent components.
        cortest : DataFrame of shape (n_columns*ncl, 6)
            Pearson correlation test of variables and latent variables
        sqload : DataFrame of shape (n_columns, ncl)
            Squared loadings of variables and latent variables.
        member : DataFrame of shape (n_columns, 4)
            Cluster members and R-square values.
        sim : DataFrame of shape (n_columns, n_columns)
            Similarities between variables for each cluster.
        loadings : DataFrame of shape (n_columns, ncl)
            Loadings of variables, i.e. the first eigen vector calculated by PCA.
            
    var_sup_ : var_sup, optional
        An object with the following attributes:
        
        cluster : Series of shape (n_columns_sup,)
            Index of the cluster each supplementary column belongs to.
        cor : DataFrame of shape (n_quanti_sup, ncl)
            Pearson correlation of supplementary variables and latent variables.
        cortest : DataFrame of shape (n_quanti_sup*ncl, 6)
            Pearson correlation test of supplementary variables and latent variables.
        sqload : DataFrame of shape (n_columns_sup, ncl)
            Squared loadings of supplementary variables and latent variables.
        member : DataFrame of shape (n_columns_sup, 4)
            Cluster members and R-square values.

    References
    ----------
    [1] E. Vigneau, M. Qannari. `Clustering of variables around latent components <https://www.researchgate.net/publication/243044470_Clustering_of_Variables_Around_Latent_Components>`_. in Statistics, Simulation and Computation, 32(4), pp.1131-1150, 2003.

    [2] M. Chavent, V. Kuentz Simonet, B. Liquet, J. Saracco. `ClustOfVar: An R package for the Clustering of Variables <https://www.jstatsoft.org/article/view/v050i13>`_. in Journal of Statistical Software, 50(13), september 2012.

    [3] Rakotomalala, R. `Classification de variables : classification autour de variables latentes <https://eric.univ-lyon2.fr/ricco/cours/slides/classification_de_variables.pdf>`_. Tutoriel Tanagra pour le Data Mining
    
    Examples
    --------
    >>> from clustvartools.datasets import decathlon
    >>> from clustvartools import CLV
    >>> clf = CLV(ncl=3,sup_var=(10,11,12))
    >>> clf.fit(decathlon.data)
    CLV(ncl=3,sup_var=(10,11,12))
    """
    def __init__(
            self,
            ncl = 2,
            row_w = None,
            sup_var = None,
            tol = 1e-7
        ):
            self.ncl = ncl
            self.row_w = row_w
            self.sup_var = sup_var
            self.tol = tol
    
    def fit(self,X,y=None):
        """
        Compute CLV
        
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
        # preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # get supplementary elements labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        sup_var_label = get_sup_label(X=X,indexes=self.sup_var,axis=1)
            
        # make a copy of the original data
        Xtot = X.copy()

        # drop supplementary variables columns
        if self.sup_var is not None:
            X_sup_var, X = X.loc[:,sup_var_label], X.drop(columns=sup_var_label)
                
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # hierarchical clustering analysis of mixed data
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if X contains only numerics columns
        if not all(is_numeric_dtype(X[c]) for c in X.columns):
            raise TypeError("Columns in X must be numerics.")
        
        # set number of rows and columns
        n_rows, n_cols = X.shape
        
        # set individuals weights
        if self.row_w is None:
            row_w = Series(ones(n_rows)/n_rows,index=X.index,name="weight")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)): 
            raise TypeError("row_w must be a 1d array-like of individuals weights.")
        elif len(self.row_w) != n_rows: 
            raise ValueError(f"row_w must be a 1d array-like of shape ({n_rows},).")
        else:
            row_w = Series(array(self.row_w)/sum(self.row_w),index=X.index,name="weight")
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization: z_ik = (x_ik - m_k)/s_k
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # compute weighted average and weighted standard deviation
        center = wmean(X=X,w=row_w)
        scale = wstd(X=X,w=row_w) 
        # standardization: z_ik = (x_ik - m_k)/s_k
        Z = (X - center)/scale
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # data preparation
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
        # compute similarity and dissimilarity matrices
        S = DataFrame(diag(ones(n_cols)),index=X.columns,columns=X.columns)
        D = DataFrame(zeros((n_cols,n_cols)),index=X.columns,columns=X.columns)
        for i in range(n_cols-1):
            for j in range(i+1,n_cols):
                # similarity
                S.iloc[i,j] = coeffsim(X=X.iloc[:,i],Y=X.iloc[:,j],w=row_w)
                # dissimilarity
                D.iloc[i,j] = clust_diss(X=Z.iloc[:,i],Y=Z.iloc[:,j],tol=self.tol)
                S.iloc[j,i], D.iloc[j,i] = S.iloc[i,j], D.iloc[i,j]
        
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
            # find agglomerands cl1 and cl2, with former < latter
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
                ia[ncl], ib[ncl] = l, r
            # add height of aggregation
            lev[ncl] = mindis
            # update cluster matrix
            for i in range(n_cols):
                clustmat[i,ncl] = clustmat[i,ncl+1]
                if clustmat[i,ncl] == cl2:
                    clustmat[i,ncl] = cl1
            # update dissimilarity matrix in cluster 1
            for i in range(n_cols):
                if (i != cl1) and (i != cl2) and (init_crit[i] == 1):
                    A = [j for j in init if j in [k for k,x in enumerate(clustmat[:,ncl+1]) if x == i]]
                    D.iloc[cl1,i] = clust_diss(X=Z.iloc[:,clus],Y=Z.iloc[:,A],tol=self.tol)
                    D.iloc[i,cl1] = D.iloc[cl1,i]
            card[cl1] = card[cl1] + card[cl2]
            init_crit[cl2] = 0
            nnsvar.nndiss[cl2] = maxv
            # update dissimilarity matrix in cluster 2
            for i in range(n_cols):
                D.iloc[cl2,i] = maxv
                D.iloc[i,cl2] = D.iloc[cl2,i]
            nnsvar = getnnsvar(D=D,crit=init_crit)
   
        # reverse
        merge = array([ia,ib]).T[::-1]
        # convert to python scipy 
        linkage = zeros((n_cols-1,4))
        linkage[:,0] = where(merge[:,0] < 0,-merge[:,0],where(merge[:,0]==0,0,merge[:,0]+n_cols))
        linkage[:,1] = where(merge[:,1] < 0,-merge[:,1],where(merge[:,1]==0,n_cols,merge[:,1]+n_cols))
        linkage[:,2] = lev[::-1]
        linkage[:,3] = size[::-1]
    
        # height of aggregation
        height = DataFrame(c_[list(range(1,linkage.shape[0]+1)),linkage[:,2][::-1]],columns=["cluster","height"])
        height["diff_1"] = -1*height["height"].diff(1)
        height["diff_2"] = height["diff_1"].diff(-1)
        height["cluster"] = height["cluster"].astype("int")

        # convert to dictionary
        tree_ = {"S":S,"D":Dprim,"Z":linkage,"height":height,"merge":linkage[:,:2],"size":linkage[:,3]}
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
        call_ = {"Xtot":Xtot,"X":X,"Z":Z,"center":center,"scale":scale,"row_w":row_w,
                 "ncl":ncl,"tree":tree,"sup_var":sup_var_label}
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Informations for variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # assign cluster
        cluster = Series(fcluster(linkage,t=ncl,criterion="maxclust"), index = D.index, name = "cluster",dtype="category")
        # unique cluster
        uq_cluster = sorted(cluster.unique())
        
        # cluster infos
        clust_infos = {}
        for k in uq_cluster:
            clus = cluster[cluster==k].index
            score = clust_score(X=Z.loc[:,clus],tol=self.tol)._asdict()
            cor2 = S.loc[clus,clus]
            score = {**score, **{"cluster" : clus, "ncl" : len(clus), "name" : clus, "cor2" : cor2}}
            clust_infos[k] = namedtuple("clust_score",score.keys())(*score.values())
            
        # principal components - latent components
        pcs = concat((Series(cl.u,index=X.index).to_frame(c) for c, cl in clust_infos.items()),axis=1)
        pcs.columns = pcs.columns.astype("category")
    
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # informations for clusters
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # cluster summary
        cluster_summary = DataFrame(columns=["Cluster","Members","Variation Explained","Proportion Explained"]).astype("float")
        i = 0
        for c, cl in clust_infos.items():
            cluster_summary.loc[i] = [c, cl.ncl, cl.d, cl.p]
            i += 1
        # convert to category and integer
        cluster_summary["Cluster"] = cluster_summary["Cluster"].astype("int").astype("category")
        cluster_summary["Members"] = cluster_summary["Members"].astype("int")
        # inter cluster correlations
        icluster_cor = wcorr(X=pcs,w=row_w,ddof=0)
        # correlation test between variables each principal components
        icluster_cortest = wcorrtest(X=pcs,w=row_w).drop(columns=["test"]).rename(columns={"statistic" : "r","pvalue" : "Pr(>|t|)"})
        icluster_cortest[["variable1","variable2"]] = icluster_cortest[["variable1","variable2"]].astype("int").astype("category")
        icluster_cortest["r**2"] = icluster_cortest["r"]**2
        icluster_cortest["t"] = icluster_cortest["r"]*sqrt(((n_rows-2)/(1- icluster_cortest["r**2"])))
        icluster_cortest = icluster_cortest[["variable1","variable2","r","r**2","t","Pr(>|t|)"]]

        # convert to dictionary
        cluster_ = {"summary" : cluster_summary,"princomps" : pcs, "cor" : icluster_cor, "cortest" : icluster_cortest}
        # convert to namedtuple
        self.cluster_ = namedtuple("cluster",cluster_.keys())(*cluster_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # linear coefficients and standard scoring coefficients
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # loadings 
        V = concat((Series(cl.v,index=cl.name).to_frame(c) for c, cl in clust_infos.items()),axis=1).astype("float").loc[Z.columns,:]
        V.columns = V.columns.astype("category")
        # coefficients of linear combinations
        coeffs = concat((-(V.T * center / scale).T.sum(axis=0).to_frame("const").T,(V.T / scale).T),axis=0)
        # standardized scoring coefficients
        coeffs_std = V/sqrt(cluster_summary["Variation Explained"].values)
        coeffs_std.columns = coeffs_std.columns.astype("category")
        # convert to dictionary
        coef_ = {"coef" : coeffs,"coef_std" : coeffs_std}
        # convert to namedtuple
        self.coef_ = namedtuple("coef",coef_.keys())(*coef_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # variables informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # cluster correlations
        cluster_cor = DataFrame(index=X.columns,columns=uq_cluster).astype("float")
        for k in X.columns:
            for i, l in enumerate(uq_cluster):
                cluster_cor.loc[k,l] = wpearsonr(x=Z[k].values,y=pcs.iloc[:,i],w=row_w).statistic
        # correlation test between variables each principal components
        cluster_cortest = cluster_cor.stack().reset_index()
        cluster_cortest.columns = ["variable","cluster","r"]
        cluster_cortest["r**2"] = cluster_cortest["r"]**2
        cluster_cortest["t"] = cluster_cortest["r"]*sqrt(((n_rows-2)/(1- cluster_cortest["r**2"])))
        cluster_cortest["Pr(>|t|)"] = 2*sst.sf(abs(cluster_cortest["t"]),n_rows - 2)
        
        # squared loadings
        cluster_sqload = cluster_cor ** 2
        # cluster's members : r**2 own cluster, r**2 next closest, ratio ((1-r**2 own)/(1 - r**2 next))
        cluster_member = clust_member(D=cluster_sqload,cluster=cluster,method="max")
        
        # inter similarity matrix
        isim = concat((cl.cor2 for _, cl in clust_infos.items()),axis=0).astype("float").loc[X.columns,X.columns]
        # convert to dictionary
        quanti_var_ = {"cluster" : cluster, "cor" : cluster_cor,"cortest" : cluster_cortest, "sqload" : cluster_sqload, 
                       "member" : cluster_member, "sim" : isim, "loadings" : V}
        # convert to namedtuple
        self.quanti_var_ = namedtuple("quanti_var",quanti_var_.keys())(*quanti_var_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # supplementary variables informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.sup_var is not None:
            # correlation of supplementary continuous variables to synthetic scores
            if any(is_numeric_dtype(X_sup_var[k]) for k in sup_var_label):
                var_sup_cor = DataFrame(index=sup_var_label,columns=uq_cluster).astype("float")
                for k in sup_var_label:
                    for i, l in enumerate(uq_cluster):
                        if is_numeric_dtype(X_sup_var[k]):
                            var_sup_cor.loc[k,l] = wpearsonr(x=X_sup_var[k].values,y=pcs.iloc[:,i],w=row_w).statistic
                # remove rows where some columns are missings
                var_sup_cor = var_sup_cor.dropna(subset=[1])
                # correlation test between supplementary variables each principal components
                var_sup_cortest = var_sup_cor.stack().reset_index()
                var_sup_cortest.columns = ["variable","cluster","r"]
                var_sup_cortest["r**2"] = var_sup_cortest["r"]**2
                var_sup_cortest["t"] = var_sup_cortest["r"]*sqrt(((n_rows-2)/(1- var_sup_cortest["r**2"])))
                var_sup_cortest["Pr(>|t|)"] = 2*sst.sf(abs(var_sup_cortest["t"]),n_rows - 2)
            else:
                var_sup_cor, var_sup_cortest = None, None
        
            # squared loadings for supplementary variables
            var_sup_sqload = DataFrame(index=sup_var_label,columns=uq_cluster).astype("float")
            for k in sup_var_label:
                for i, l in enumerate(uq_cluster):
                    var_sup_sqload.loc[k,l] = coeffsim(X=self.cluster_.princomps.iloc[:,i],Y=X_sup_var[k],w=row_w)
                    
            # assign cluster to supplementary variables
            var_sup_cluster = var_sup_sqload.idxmax(axis=1).astype("category")
            var_sup_cluster.name = "cluster"
            
            # cluster members and R-square values for supplementary variables
            var_sup_member = clust_member(D=var_sup_sqload,cluster=var_sup_cluster,method="max")
            
            # convert to dictionary
            var_sup_ = {"cluster" : var_sup_cluster, "cor" : var_sup_cor, "cortest" : var_sup_cortest,
                        "sqload" : var_sup_sqload, "member" : var_sup_member}
            # convert to namedtuple
            self.var_sup_ = namedtuple("var_sup",var_sup_.keys())(*var_sup_.values())
            
        return self
    
    def fit_predict(self,X,y=None):
        """Compute squared loadings and predict cluster index for each column.

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
        """Compute clustering and transform X to cluster-squared loadings space.

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
        return self.quanti_var_.sqload
    
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
        cluster = dist.idxmax(axis=1).astype("category")
        cluster.name = "cluster"
        return cluster
        
    def transform(self,X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the squared loadings to the cluster centers

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

        # cluster labels
        uq_cluster = self.cluster_.cor.columns
        # squared loadings for new variables
        sqload = DataFrame(index=X.columns,columns=uq_cluster).astype("float")
        for k in X.columns:
            for i, l in enumerate(uq_cluster):
                sqload.loc[k,l] = coeffsim(X=self.cluster_.princomps.iloc[:,i],Y=X[k],w=self.call_.row_w)
        return sqload