# -*- coding: utf-8 -*-
from __future__ import annotations
import pathlib
from pandas import DataFrame, Series, read_excel, read_csv
from pyreadr import read_r
from collections import OrderedDict, namedtuple

def namedtupledocstring(docstring, *ntargs):
    nt = namedtuple(*ntargs)
    class NT(nt):
        __doc__ = docstring
    return NT

DATASETS_DIR = pathlib.Path(__file__).parent / "data"

#------------------------------------------ burger dataset ----------------------------------------------------- 
burger = read_excel(DATASETS_DIR/"burger.xlsx",sheet_name="Feuil1",index_col=0,header=0)
burger.__doc__ = """
Burger King Dataset

Examples
--------
>>> from clustvartools.datasets import burger
>>> from clustvartools import VarClus
>>> clf = VarClus(sup_var=13)
>>> clf.fit(burger)
"""

#------------------------------------------ canines dataset ----------------------------------------------------- 
canines = OrderedDict(
    actif = read_excel(DATASETS_DIR/"canines.xlsx",sheet_name="Feuil1",header=0,index_col=0),
    ind_sup = read_excel(DATASETS_DIR/"canines.xlsx",sheet_name="Feuil2",header=0,index_col=0),
    sup_var = read_excel(DATASETS_DIR/"canines.xlsx",sheet_name="Feuil3",header=0,index_col=0),
    data = read_excel(DATASETS_DIR/"canines.xlsx",sheet_name="Feuil4",header=0,index_col=0)
)
__doc__ = """
Canines Dataset

The data contains 32 individuals

Returns
-------
canines : canines
    An object with the following attributes:

    actif: DataFrame of shape (27,6)
        Input data for actifs elements.
    ind_sup: DataFrame of shape (5,6)
        for supplementary individuals dataset.
    sup_var: DataFrame of shape (27, 2)
        Input data for supplementary variables.
    data: DataFrame of shape (32,8)
        Overall dataset.

Examples
--------
>>> from scientisttools.datasets import canines
>>> from scientisttools import MCA
>>> clf = MCA(ind_sup=range(27,32),sup_var=(6,7))
>>> clf.fit(canines.data)
MCA(ind_sup=(27,28,29,30,31),sup_var=(6,7))
"""
canines = namedtupledocstring(__doc__,"canines",canines.keys())(*canines.values())

#------------------------------------------ jobrate dataset ----------------------------------------------------- 
jobrate = read_excel(DATASETS_DIR/"jobrate.xlsx",index_col=None,header=0)
jobrate.__doc__ = """
Jobrate Dataset

Examples
--------
>>> from clustvartools.datasets import jobrate
>>> from clustvartools import VARHCA
>>> clf = VARHCA(n_clusters=4,sup_var=13)
>>> clf.fit(jobrate)
VARHCA(n_clusters=4,var_sup=13,matrix_type="completed",metric="euclidean",method="ward")
"""

#------------------------------------------ uscrime dataset ----------------------------------------------------- 
uscrime = read_excel(DATASETS_DIR/"uscrime.xlsx",sheet_name="Feuil1",header=0,index_col=0)
uscrime.__doc__ = """
US Crime Dataset

These data are crime-related and demographic statistics for 47 US states in 1960. The data were collected from the FBI's Uniform Crime Report and other government agencies to determine how the variable crime rate depends on the other variables measured in the study.

1. Crime.rate: # of offenses reported to police per million population
2. Male14_24: The number of males of age 14-24 per 1000 population
3. Southern.states: Indicator variable for Southern states (Yes, No)
4. Education: Mean # of years of schooling x 10 for persons of age 25 or older
5. Expend60: 1960 per capita expenditure on police by state and local government
6. Expend59: 1959 per capita expenditure on police by state and local government
7. Labor.force: Labor force participation rate per 1000 civilian urban males age 14-24
8. Male: The number of males per 1000 females
9. Pop.size: State population size in hundred thousands
10. Non.white: The number of non-whites per 1000 population
11. Unemp14_24: Unemployment rate of urban males per 1000 of age 14-24
12. Unemp35_39: Unemployment rate of urban males per 1000 of age 35-39
13. Family.income: Median value of transferable goods and assets or family income in tens of $
14. Under.median: The number of families per 1000 earning below 1/2 the median income

References
----------
[1] see https://lib.stat.cmu.edu/DASL/Datafiles/USCrime.html

Vandaele, W. (1978) Participation in illegitimate activities: Erlich revisited. In Deterrence and incapacitation, Blumstein, A., Cohen, J. and Nagin, D., eds., Washington, D.C.: National Academy of Sciences, 270-335. Methods: A Primer, New York: Chapman & Hall, 11. Also found in: Hand, D.J., et al. (1994) A Handbook of Small Data Sets, London: Chapman & Hall, 101-103.

Examples
--------
>>> from clustvartools.datasets import uscrime
>>> from clustvartools import VARHCA
>>> clf = VARHCA(n_clusters=3,similarity="pearson",method="ward",quanti_sup=0)
>>> clf.fit(uscrime)
VARHCA(method="ward",n_clusters=3,quanti_sup=0,similarity="pearson")
"""

#------------------------------------------ voting dataset -----------------------------------------------------
voting = read_excel(DATASETS_DIR/"congressvotingrecords.xlsx")
voting.__doc__ = """
Congressional Voting Records

The `Congressional Voting Records <https://archive.ics.uci.edu/dataset/105/congressional+voting+records>`_ UCI dataset.

Examples
--------
>>> from clustvartools.datasets import voting
>>> from clustvartools import CatVARHCA
>>> X = voting.iloc[:,1:]
>>> clf =  CatVARHCA(n_clusters=2,diss_metric="cramer",metric="euclidean",method="ward")
>>> clf.fit(X)
CatVARHCA(n_clusters=2,diss_metric="cramer",metric="euclidean",method="ward")
"""

#------------------------------------------ all dataset -----------------------------------------------------

__all__ = [
    "burger",
    "canines",
    "jobrate",
    "uscrime",
    "voting"
]