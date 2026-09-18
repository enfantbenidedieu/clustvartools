# -*- coding: utf-8 -*-
from numpy import ones, array, dot, ndarray,sqrt,repeat
from pandas import Series, concat,DataFrame
from pandas.api.types import is_numeric_dtype
from collections import namedtuple
from scipy.stats import t as sst
from statsmodels.multivariate.factor_rotation import rotate_factors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# interns functions
from .functions.preprocessing import preprocessing
from .functions.get_sup_label import get_sup_label
from .functions.statistics import wmean, wstd, wcorr
from .functions.gsvd import gSVD
from .functions.func_dclv import gPCA, stabilitycl, vartot
from .functions.tests import wpearsonr, wcorrtest
from .functions.utils import check_is_dataframe, convert_series_to_dataframe
from ..others._clust_member import clust_member
from ..others._coeffsim import coeffsim

class DCLV(BaseEstimator,TransformerMixin):
    """
    Divisive Clustering of Variables around Latent Variables (DCLV)
    
    Performns divisive clustering of a set of continuous variables into disjoint or hierarchical clusters.
    Supplementary variables (continuous and/or categorical) may be used.

    Parameters
    ----------
    threshold : int, default = 1
        A threshold value to test whether the second eigenvalue is greater than.
    
    maxcl : int, default = None
        The maximum number of clusters.

    method : {"varimax","quartimax"}, default = "quartimax"
        The factor rotation method. Options include:
        
        * "quartimax" for quartimax rotation
        * "varimax" for varimax rotation

    max_iter : int, default = 10
        The maximum number of itérations.

    row_w : 1d array-like of shape (n_rows,), default = None
        An optional rows weights. The weights are given only for the active rows.
    
    sup_var : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary continuous variables.
    
    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger 
        than ``-tol*lambda1`` where ``lambda1`` is the largest eigenvalue.

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
    
    [4] SAS. `The VARCLUS Procedure <https://support.sas.com/documentation/onlinedoc/stat/141/varclus.pdf>`_. SAS/STAT® 14.1 User's Guide.
    
    Examples
    --------
    >>> from clustvartools.datasets import decathlon
    >>> from clustvartools import DCLV
    >>> clf = DCLV(sup_var=(10,11,12))
    >>> clf.fit(decathlon.data)
    DCLV(sup_var=(10,11,12))
    """
    def __init__(
            self, 
            threshold = 1, 
            maxcl = None, 
            method = "quartimax",
            max_iter = 10, 
            row_w = None, 
            sup_var = None, 
            tol = 1e-7, 
    ):
        self.threshold = threshold
        self.maxcl = maxcl
        self.method = method
        self.max_iter = max_iter
        self.row_w = row_w
        self.sup_var = sup_var
        self.tol = tol
    
    def fit(self,X,y=None):
        """Compute DCLV

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
        if not (self.method in ("varimax","quartimax")):
            raise ValueError("method should be one of 'varimax', 'quartimax'")
        
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # preprocessing (drop level, fill NA with mean, convert to ordinal levels)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set supplementary quantitative variables labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # get supplementary variables labels
        sup_var_label = get_sup_label(X=X, indexes=self.sup_var, axis=1)
            
        # make a copy of the original data
        Xtot = X.copy()

        # drop supplementary variables columns
        if self.sup_var is not None:
            X_sup_var, X = X.loc[:,sup_var_label], X.drop(columns=sup_var_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # divisive clustering of variables around latent variables (DCLV)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if X contains only numerics columns
        if not all(is_numeric_dtype(X[c]) for c in X.columns):
            raise TypeError("Columns in X must be numerics.")
        
        # set number of rows and columns
        n_rows, n_cols = X.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set individuals and variables weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set individuals weights
        if self.row_w is None: 
            row_w = Series(ones(n_rows)/n_rows,index=X.index,name="weight")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)): 
            raise TypeError("row_w must be a 1d array-like of rows weights.")
        elif len(self.row_w) != n_rows: 
            raise ValueError(f"row_w must be 1d array-like of shape ({n_rows},).")
        else: 
            row_w = Series(array(self.row_w)/sum(self.row_w),index=X.index,name="weight")
        
        # set columns weights
        col_w = Series(ones(n_cols),index=X.columns,name="weight")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # standardization: z_ik = (x_ik - m_k)/s_k
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # compute weighted average and weighted standard deviation
        center = wmean(X=X,w=row_w)
        scale = wstd(X=X,w=row_w) 
        # standardization: z_ik = (x_ik - m_k)/s_k
        Z = (X - center)/scale

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # VARCLUS Procedure - top down procedure
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ClusterInfos = namedtuple("ClusterInfos", ["cluster","name","d1","d2","p","u","v","cor2"])
        # principal component analysis with all variables
        d0, p0, v0, u0 = gPCA(X=Z,row_w=row_w,col_w=col_w,tol=self.tol)
        # step 0 (initialization) - with all variables
        clus0 = ClusterInfos(
            cluster = X.columns,
            name = X.columns,
            d1 = d0[0],
            d2 = d0[1],
            p = p0[0],
            u = u0[:,0],
            v = v0[:,0],
            cor2 = wcorr(X=X,w=row_w)**2
        )
        clust_infos = {0 : clus0}
        
        # divisive and iterative steps
        while True:
            # check if maximum number of clusters reached
            if ((self.maxcl is not None) and 
                (len(clust_infos) >= self.maxcl)):
                break
            
            # find which cluster has the largest second eigenvalue
            idx = max(clust_infos, key=lambda x: clust_infos.get(x).d2)
            if clust_infos[idx].d2 > self.threshold:
                clus = clust_infos[idx].cluster
                gsvd = gSVD(X=Z.loc[:,clus],row_w=row_w,col_w=col_w.loc[clus],tol=self.tol)
                d, v = gsvd.d, gsvd.V[:,:2]
            else:
                break
            
            # check
            if d[1] > self.threshold:
                # eigenvectors after rotation
                rv, _ = rotate_factors(A=v,method=self.method)
                # principal components after rotation
                ru = dot(Z[clus],rv)

                clus1, clus2 = [], []
                for k in clus:
                    cor1 = wpearsonr(x=X[k].values,y=ru[:,0],w=row_w).statistic**2
                    cor2 = wpearsonr(x=X[k].values,y=ru[:,1],w=row_w).statistic**2
                    if cor1 > cor2:
                        clus1.append(k)
                    else:
                        clus2.append(k)
                # nearest component sorting
                fin_clus1, fin_clus2, _ = stabilitycl(Z,row_w,col_w,self.tol,clus1,clus2,X.columns.tolist(),self.max_iter)
                # generalized singular values decomposition for first cluster
                d1, p1, v1, u1 = gPCA(X=Z.loc[:,fin_clus1],row_w=row_w,col_w=col_w.loc[fin_clus1],tol=self.tol)
                # generalized singular values decomposition for second cluster
                d2, p2, v2, u2 = gPCA(X=Z.loc[:,fin_clus2],row_w=row_w,col_w=col_w.loc[fin_clus2],tol=self.tol)

                # assign to dictionary
                clust_infos[idx] = ClusterInfos(
                    cluster = fin_clus1,
                    name = fin_clus1,
                    d1 = d1[0],
                    d2 = d1[1] if len(fin_clus1) > 1 else 0,
                    p = p1[0],
                    u = u1[:,0],
                    v = v1[:,0],
                    cor2 = wcorr(X=X.loc[:,fin_clus1],w=row_w)**2 if len(fin_clus1) > 1 else DataFrame([[1]],index=fin_clus1,columns=fin_clus1)
                )
                clust_infos[len(clust_infos)] = ClusterInfos(
                    cluster = fin_clus2,
                    name = fin_clus2,
                    d1 = d2[0],
                    d2 = d2[1] if len(fin_clus2) > 1 else 0,
                    p = p2[0],
                    u = u2[:,0],
                    v = v2[:,0],
                    cor2 = wcorr(X=X.loc[:,fin_clus2],w=row_w)**2 if len(fin_clus2) > 1 else DataFrame([[1]],index=fin_clus2,columns=fin_clus2)
                )
            else:
                break
        
        # set number of cluster
        ncl = len(clust_infos.keys())
        #convert to dictionary
        call_ = {"Xtot":Xtot,"X":X,"Z":Z,"center":center,"scale":scale,"row_w":row_w,
                 "ncl":ncl,"sup_var":sup_var_label}
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # assign cluster to variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # clusters
        cluster = concat((Series(repeat(c+1,len(cl.cluster)),index=cl.cluster,name="cluster",dtype="category") for c, cl in clust_infos.items()),axis=0)
        # unique cluster
        uq_cluster = sorted(cluster.unique())
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # principal components and loadings
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # clusters principal components
        pcs = concat((Series(cl.u,index=X.index).to_frame(c+1) for c, cl in clust_infos.items()),axis=1)
        pcs.columns = pcs.columns.astype("category")
    
        # clusters eigenvectors
        V = concat((Series(cl.v,index=cl.name).to_frame(c+1) for c, cl in clust_infos.items()),axis=1).astype("float").loc[Z.columns,:]
        V.columns = V.columns.astype("category")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # informations for clusters
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # cluster summary
        cluster_summary = DataFrame(columns=["Cluster","Members","Variation Explained","Proportion Explained","Second Eigenvalue"]).astype("float")
        i = 0
        for c, cl in clust_infos.items():
            cluster_summary.loc[i] = [int(c+1), len(cl.cluster), cl.d1, cl.p, cl.d2]
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
                cluster_cor.loc[k,l] = wpearsonr(x=X[k].values,y=pcs.iloc[:,i],w=row_w).statistic
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
        """Transform X to a cluster-squared loadings space.

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
        #check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # convert to pd.DataFrame if pd.Series
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = convert_series_to_dataframe(X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X is an object of class pd.DataFrame
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_dataframe(X)

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
        # check if convevient row shape
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