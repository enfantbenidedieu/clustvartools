# -*- coding: utf-8 -*-
from numpy import ones, array, dot, ndarray, sqrt, c_, corrcoef, linalg
from pandas import Series, concat, get_dummies, DataFrame
from collections import namedtuple, OrderedDict
from itertools import chain, repeat
from factor_analyzer import Rotator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#interns functions
from .functions.preprocessing import preprocessing
from .functions.get_sup_label import get_sup_label
from .functions.concat_empty import concat_empty
from .functions.statistics import wmean, wstd
from .functions.varclusfuncs import gPCA, total_variance, stability
from .functions.utils import check_is_dataframe
from .functions.tests import wpearsonr, eta2test
from .functions.utils import check_is_dataframe, convert_series_to_dataframe
from ...others._splitmix import splitmix
from ...others._disjunctive import disjunctive

class CatDCLV(BaseEstimator,TransformerMixin):
    """
    Divisive Clustering of Categorical Variables around Latent Variables (CatDCLV)
    
    Performns divisive clustering analysis of a set of categorical variables into disjoint or hierarchical clusters.

    Parameters
    ----------
    maxeig : int, default = 1
        The maximum eigen values.
        
    prop : float, default = None
        The percentage of variation explained.

    maxcl : int, default = None
        The maximum number of clusters.

    rotation : {"varimax","quartimax"}, default = "quartimax"

    max_iter : int, default = 0
        The maximum number of itérations.

    row_w : 1d array-like of shape (n_rows,), default = None
        An optional rows weights. The weights are given only for the active rows.

    col_w : 1d array-like of shape (n_columns,), default = None
        An optional columns weights. The weights are given only for the active columns.

    sup_var : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary continuous variables.
    
    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger 
        than ``-tol*lambda1`` where ``lambda1`` is the largest eigenvalue.

    Attributes
    ----------
    
    """
    def __init__(
            self, 
            scale_unit = True, 
            maxeig = 1, 
            prop = None,
            maxcl = None, 
            rotation = "quartimax",
            max_iter = 0, 
            row_w = None, 
            col_w = None, 
            sup_var = None, 
            tol = 1e-7, 
    ):
        self.scale_unit = scale_unit
        self.maxeig = maxeig
        self.prop = prop
        self.maxcl = maxcl
        self.rotation = rotation
        self.max_iter = max_iter
        self.row_w = row_w
        self.col_w = col_w
        self.sup_var = sup_var
        self.tol = tol
    
    def fit(self,X,y=None):
        """Compute VARCLUS procedure

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
        if not isinstance(self.scale_unit,bool):
            raise TypeError("")
        
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.prop is not None:
            if self.prop < 0 and self.prop > 1:
                raise ValueError("")
        
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
        # factor analysis of mixed data (FAMD)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #et number of rows and columns
        n_rows, n_cols = X.shape

        # split X
        split_X = splitmix(X)
        X_quanti, X_quali = split_X.quanti, split_X.quali
        n_quanti, n_quali = split_X.k1, split_X.k2

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals and variables weights
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
        
        #set columns weights
        if self.col_w is None: 
            col_w = Series(ones(n_cols),index=X.columns,name="weight")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)): 
            raise TypeError("col_w must be a 1d array-like of columns weights.")
        elif len(self.col_w) != n_cols: 
            raise ValueError(f"col_w must be a 1d array-like of shape ({n_cols},).")
        else: 
            col_w = Series(array(self.col_w),index=X.columns,name="weight")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardization: z_ik = (x_ik - m_k)/s_k
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # compute weighted average and weighted standard deviation
        center = wmean(X=X,w=row_w)
        if self.scale_unit:
            scale = wstd(X=X,w=row_w) 
        else:
            scale = Series(ones(n_cols),index=X.columns,name="scale")
        # standardization: z_ik = (x_ik - m_k)/s_k
        Z = (X - center)/scale

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # VARCLUS Procedure
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ClusterInfos = namedtuple('ClusInfo', ['cluster','eigval1','eigval2','princomp1','eigprop1'])
        eigvals, eigprops, _, princomps = gPCA(Z=Z,row_w=row_w,col_w=col_w,tol=self.tol)
        # initialization - with all variables
        clus0 = ClusterInfos(
            cluster = X.columns,
            eigval1 = eigvals[0],
            eigval2 = eigvals[1],
            princomp1 = princomps[:,0],
            eigprop1 = eigprops[0]
        )
        clusters_dict = {[(0, clus0)]}

        while True:
            if ((self.maxcls is not None) and 
                (len(clusters_dict) >= self.maxcls)):
                break
            i = max(clusters_dict, key=lambda x: clusters_dict.get(x).eigval2)
            if clusters_dict[i].eigval2 > self.max_eigen:
                clusters = clusters_dict[i].cluster
                eigvals, _ , eigvects, _ = gsvd(Z=Z.loc[:,clusters],row_w=row_w,col_w=col_w.loc[clusters],tol=self.tol)
            else:
                break
            
            # check
            if eigvals[1] > self.max_eigen:
                clf = Rotator(method=self.rotation)
                reigvecs = clf.fit_transform(DataFrame(eigvects))
                r_princomps = dot(Z[clusters],reigvecs)

                cluster1, cluster2 = [], []
                for k in clusters:
                    corr_pc1 = wpearsonr(x=Z[k].values,y=r_princomps[:, 0],w=row_w).statistic**2
                    corr_pc2 = wpearsonr(x=Z[k].values,y=r_princomps[:, 1],w=row_w).statistic**2
                    if corr_pc1 > corr_pc2:
                        cluster1.append(k)
                    else:
                        cluster2.append(k)
                tol = self.tol
                init_variance = total_variance(Z,X,X_quanti,X_quali,n_quanti,n_quali,row_w,col_w,tol,cluster1,cluster2)
                print(init_variance)
                fin_cluster1, fin_cluster2, _ = stability(Z,X,X_quanti,X_quali,n_quanti,n_quali,row_w,col_w,self.tol,cluster1,cluster2,self.max_iter)
                
                # generalized singular values decomposition for first cluster
                eigvals1, eigprops1, _ , princomps1  = gsvd(X=Z.loc[:,fin_cluster1],row_w=row_w,col_w=col_w.loc[fin_cluster1],tol=self.tol)

                # generalized singular values decomposition for second cluster
                eigvals2, eigprops2, _ , princomps2  = gsvd(X=Z.loc[:,fin_cluster2],row_w=row_w,col_w=col_w.loc[fin_cluster2],tol=self.tol)

                # 
                clusters_dict[i] = ClusterInfos(
                    cluster = fin_cluster1,
                    eigval1 = eigvals1[0],
                    eigval2 = eigvals1[1],
                    princomp1 = princomps1[:, 0],
                    eigprop1 = eigprops1[0]
                )
                clusters_dict[len(clusters_dict)] = ClusterInfos(
                    cluster = fin_cluster2,
                    eigval1 = eigvals2[0],
                    eigval2 = eigvals2[1],
                    princomp1 = princomps2[:, 0],
                    eigprop1 = eigprops2[0]
                )
            else:
                break
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # clusters informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # cluster summary
        cluster_infos = DataFrame(columns=["Cluster","Members","Variation Explained","Proportion Explained","Second Explained"]).astype("float")
        i = 0
        for c, clusters in clusters_dict.items():
            cluster_infos.loc[i] = [repr(c+1), repr(len(clusters.cluster)), clusters.eigval1, clusters.eigprop1, clusters.eigval2]
            i += 1
        # convert to category and integer
        cluster_infos["Cluster"] = cluster_infos["Cluster"].astype("category")
        cluster_infos["Members"] = cluster_infos["Members"].astype("int")
        cluster_infos = cluster_infos.set_index("Cluster")
        cluster_infos.index.name = None
        # clusters principal components
        princomps = concat((Series(clusters.princomp1).to_frame(c+1) for c, clusters in clusters_dict.items()),axis=1)
        princomps.columns = princomps.columns.astype("category")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # variables informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # cluster R-squared
        cluster_rsquared = DataFrame(index=X.columns,columns=cluster_infos.index).astype("float")
        for k in X.columns:
            for i, l in enumerate(cluster_infos.index):
                rsq = wpearsonr(x=Z[k].values,y=princomps.iloc[:,i],w=row_w).statistic**2
                cluster_rsquared.loc[k,l] = rsq
    
        # cluster's members : r**2 own cluster, r**2 next closest, ratio ((1-r**2 own)/(1 - r**2 next))
        cluster_member = DataFrame(index=X.columns,columns=["Own Cluster","Next Closest"]).astype("float")
        cluster_member["Own Cluster"] = cluster_rsquared.max(axis=1)
        cluster_member["Next Closest"] = cluster_rsquared.apply(lambda x: x.nlargest(2).iloc[-1], axis=1)
        cluster_member["1 - R**2 Ratio"] = (1 - cluster_member["Own Cluster"])/(1 - cluster_member["Next Closest"])
        
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
        X_new : DataFrame of shape (n_columns, n_clusters)
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
        X_new : DataFrame of shape (n_columns, n_clusters) 
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
        #set index name as None
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

        cluster = self.quanti_var_.cluster
        uq_cluster = sorted(list(cluster.unique()))
        # # similarity matrix
        # S = concat((self.call_.X.corrwith(other=X[k],method=self.similarity,axis=0).to_frame(k) for k in X.columns),axis=1)
        # # dissimilarity matrix
        # D = S.apply(lambda x : self.dissimilarity(x),axis=0)
        # # distance to cluster centers
        # dist_cluster_center = DataFrame(index=X.columns,columns=uq_cluster).astype(float)
        # for i, l in enumerate(X.columns):
        #     for j, k in enumerate(uq_cluster):
        #         index = list(cluster[cluster==k].index)
        #         dist = D.T.loc[l,index]
        #         # remove l distance
        #         if len(dist.index) > 1:
        #             if l in dist.index:
        #                 dist = dist.drop(index=[l])
        #         if self.method in ("ward","average"):
        #             d = dist.sum()/len(index)
        #         elif self.method == "single":
        #             d = dist.min()
        #         elif self.method == "complete":
        #             d = dist.max()
        #         dist_cluster_center.iloc[i,j] = d
        # return dist_cluster_center