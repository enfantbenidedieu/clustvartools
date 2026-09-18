.. _getting_started:

===============
Getting started
===============

`clustvartools <https://github.com/enfantbenidedieu/clustvartools>`_ is an open source python library for the clustering of variables distributed under the MIT Licence.

The purpose of this guide is to illustrate some of the main features of ``clustvartools``. It assumes basic working knowledge of `scikit-learn <https://scikit-learn.org/stable/>`_ practices.

Fitting : estimator basics
--------------------------

As scikit-learn, clustvartools provides models called `estimators <https://scikit-learn.org/stable/glossary.html#term-estimators>`_. 
Each estimator can be fitted to some data using its `fit <https://scikit-learn.org/stable/glossary.html#term-fit>`_ method.

Here is a simple example where we fit a :class:`~clustvartools.CLV` to :class:`~scientisttools.datasets.decathlon` data:

.. code:: python
  
  >>> from clustvartools.datasets import decathlon
  >>> from clustvartools import CLV
  >>> clf = CLV(ncl=3,sup_var=(10,11,12))
  >>> clf.fit(decathlon.data)
  CLV(ncl=3,sup_var=(10,11,12))
  
The ``fit`` method generally accepts 2 inputs:

- The samples `DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ (or design matrix) ``X``. The size of ``X``
  is typically ``(n_samples, n_features)``, which means that samples are
  represented as rows and features are represented as columns.
- The target values ``y`` which are true lables for ``X``. ``y`` is a pandas Series with categorical terms, but for unsupervised learning tasks, ``y`` does not need to be specified.

Transform
---------

Once the estimator is fitted, it can be used to predict the closest cluster each variable in new data belongs to.. 
You don't need to re-train the estimator:

.. code:: python

  >>> # new data
  >>> newdata = decathlon.sup_var
  >>> # squared loadings for new data and clusters
  >>> clf.transform(newdata).head()
  >>> # labels for new data
  >>> clf.predict(newdata)