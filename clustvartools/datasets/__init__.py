# -*- coding: utf-8 -*-
from __future__ import annotations
import pathlib
from pandas import read_excel, read_csv
from pyreadr import read_r
from collections import namedtuple

def namedtupledocstring(docstring, *ntargs):
    nt = namedtuple(*ntargs)
    class Dataset(nt):
        __doc__ = docstring
    return Dataset

DATASETS_DIR = pathlib.Path(__file__).parent / "data"

#------------------------------------------ apples dataset ----------------------------------------------------- 
apples = {
    "senso" : read_excel(DATASETS_DIR/"apples.xlsx",sheet_name="Feuil1",header=0,index_col=0),
    "pref" : read_excel(DATASETS_DIR/"apples.xlsx",sheet_name="Feuil2",header=0,index_col=0),
    "data" : read_excel(DATASETS_DIR/"apples.xlsx",sheet_name="Feuil3",header=0,index_col=0)
}
__doc__ = """
Apples Dataset

apples from southern hemisphere data set
Sensory characterization and consumers preference for 12 varieties of apples.

Returns
-------
Dataset : apples
    An object with the following attributes:

    senso : DataFrame of shape (12,43)
        Input data with sensory attributes.
    pref : DataFrame of shape (12, 2)
        Input data with hedonic scores given by a panel of 60 consumers.
    data: DataFrame of shape (32,8)
        Overall dataset.

Examples
--------
>>> from clustvartools.datasets import apples
>>> from clustvartools import CLV
>>> clf = CLV()
>>> clf.fit(apples.pref)
CLV()
"""
apples = namedtupledocstring(__doc__,"apples",apples.keys())(*apples.values())

#------------------------------------------ autos2005 dataset ----------------------------------------------------- 
autos2005 = {
    "actif" : read_excel(DATASETS_DIR/"autos2005.xlsx",sheet_name="Feuil1",index_col=0,header=0),
    "sup_var": read_excel(DATASETS_DIR/"autos2005.xlsx",sheet_name="Feuil2",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"autos2005.xlsx",sheet_name="Feuil3",index_col=0,header=0)
}
__doc__ = """
Autos 2005 Dataset

Returns
-------
Dataset : autos2005
    An object with the following attributes:

    actif: DataFrame of shape (38, 12)
        Actifs dataset.
    sup_var: DataFrame of shape (38, 3)
        Supplementary variables dataset.
    data: DataFrame of shape (38, 15)
        Overall dataset.

Examples
--------
>>> from clustvartools.datasets import autos2005
>>> from clustvartools import CLVmix
>>> clf = CLVmix(ncl=None,sup_var=range(12,16))
>>> clf.fit(autos2005.data)
CLVmix(ncl=None,sup_var=range(12,16))
"""
autos2005 = namedtupledocstring(__doc__,"autos2005",autos2005.keys())(*autos2005.values())

#------------------------------------------ burger dataset ----------------------------------------------------- 
burger = {
    "actif" : read_excel(DATASETS_DIR/"burger.xlsx",sheet_name="Feuil1",index_col=0,header=0),
    "sup_var" : read_excel(DATASETS_DIR/"burger.xlsx",sheet_name="Feuil2",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"burger.xlsx",sheet_name="Feuil3",index_col=0,header=0)
}
__doc__ = """
Burger King Dataset

Returns
-------
Dataset : burger
    An object with the following attributes:

    actif: DataFrame of shape (17,10)
        Active variables.
    sup_var: DataFrame of shape (17, 1)
        Supplementary variables.
    data: DataFrame of shape (17,11)
        Overall dataset.

Examples
--------
>>> from clustvartools.datasets import burger
>>> from clustvartools import HCAV, VarClus
>>> # clustering of variables - hierarchical
>>> clf = HCAV(sup_var=10)
>>> clf.fit(burger.data)
HCAV(sup_var=10)
>>> # clustering of variables - divise
>>> clf = VarClus(sup_var=10)
>>> clf.fit(burger.data)
VarClus(sup_var=10)
"""
burger = namedtupledocstring(__doc__,"burger",burger.keys())(*burger.values())

#------------------------------------------ canines dataset ----------------------------------------------------- 
canines = {
    "actif" : read_excel(DATASETS_DIR/"canines.xlsx",sheet_name="Feuil1",header=0,index_col=0),
    "sup_var" : read_excel(DATASETS_DIR/"canines.xlsx",sheet_name="Feuil2",header=0,index_col=0),
    "data" : read_excel(DATASETS_DIR/"canines.xlsx",sheet_name="Feuil3",header=0,index_col=0)
}
__doc__ = """
Canines Dataset

The data contains 27 races of dogs

Returns
-------
Dataset : canines
    An object with the following attributes:

    actif: DataFrame of shape (27,6)
        Active variables.
    sup_var: DataFrame of shape (27, 1)
        Supplementary variables.
    data: DataFrame of shape (27,7)
        Overall dataset.

Examples
--------
>>> from clustvartools.datasets import canines
>>> from clustvartools import CatHCAV, CatVARHCA
>>> # clustering of variables
>>> clf = CatHCAV(sup_var=6)
>>> clf.fit(canines.data)
CatHCAV(sup_var=6)
>>> # clustering modalities of categorical variables
>>> clf = CatVARHCA(sup_var=6)
>>> clf.fit(canines.data)
CatVARHCA(sup_var=6)
"""
canines = namedtupledocstring(__doc__,"canines",canines.keys())(*canines.values())

#------------------------------------------ cars dataset -----------------------------------------------------
cars = read_excel(DATASETS_DIR/"cars.xlsx",sheet_name="Feuil1",index_col=None,header=0)
cars.__doc__ = """
Cars Dataset

"""

#------------------------------------------ decathlon dataset ----------------------------------------------------- 
decathlon = {
    "actif" : read_excel(DATASETS_DIR/"decathlon.xlsx",sheet_name="Feuil1",index_col=0,header=0),
    "sup_var" : read_excel(DATASETS_DIR/"decathlon.xlsx",sheet_name="Feuil2",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"decathlon.xlsx",sheet_name="Feuil3",index_col=0,header=0)
}
__doc__ = """
Decathlon Dataset

Performance in decathlon

Returns
-------
Dataset : decathlon
    An object with the following attributes:

    actif: DataFrame of shape (41,10)
        Active variables.
    sup_var: DataFrame of shape (41, 2)
        Supplementary variables.
    data: DataFrame of shape (41,12)
        Overall dataset.

Examples
--------
>>> from clustvartools.datasets import decathlon
>>> from clustvartools import HCAV, VarClus
>>> # clustering of variables - hierarchical
>>> clf = HCAV(sup_var=(10,11))
>>> clf.fit(decathlon.data)
HCAV(sup_var=(10,11))
>>> # clustering of variables - divise
>>> clf = VarClus(sup_var=(10,11))
>>> clf.fit(decathlon.data)
VarClus(sup_var=(10,11))
"""
decathlon = namedtupledocstring(__doc__,"decathlon",decathlon.keys())(*decathlon.values())

#------------------------------------------ jobrate dataset ----------------------------------------------------- 
jobrate = {
    "actif" : read_excel(DATASETS_DIR/"jobrate.xlsx",sheet_name="Feuil1",index_col=None,header=0),
    "sup_var" : read_excel(DATASETS_DIR/"jobrate.xlsx",sheet_name="Feuil2",index_col=None,header=0),
    "data" : read_excel(DATASETS_DIR/"jobrate.xlsx",sheet_name="Feuil3",index_col=None,header=0),
}
__doc__ = """
Jobrate Dataset

Returns
-------
Dataset : jobrate
    An object with the following attributes:

    actif: DataFrame of shape (103,13)
        Actif variables.
    sup_var: DataFrame of shape (103, 1)
        Supplementary variables.
    data: DataFrame of shape (103,14)
        Overall dataset.

Examples
--------
>>> from clustvartools.datasets import jobrate
>>> from clustvartools import HCAV
>>> clf = HCAV(ncl=4,sup_var=13)
>>> clf.fit(jobrate.data)
HCAV(ncl=4,sup_var=13)
"""
jobrate = namedtupledocstring(__doc__,"jobrate",jobrate.keys())(*jobrate.values())

#------------------------------------------ olympic dataset ----------------------------------------------------- 
olympic = {
    "actif" : read_excel(DATASETS_DIR/"olympic.xlsx",sheet_name="Feuil1",index_col=0,header=0),
    "sup_var" : read_excel(DATASETS_DIR/"olympic.xlsx",sheet_name="Feuil2",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"olympic.xlsx",sheet_name="Feuil3",index_col=0,header=0)
}
__doc__ = """
Olympic Dataset

Performance in olympic

Returns
-------
Dataset : olympic
    An object with the following attributes:

    actif: DataFrame of shape (33,10)
        Active variables.
    sup_var: DataFrame of shape (33,2)
        Supplementary variables.
    data: DataFrame of shape (33,12)
        Overall dataset.

Examples
--------
>>> from clustvartools.datasets import olympic
>>> from clustvartools import HCAV, VarClus
>>> # clustering of variables - hierarchical
>>> clf = HCAV(sup_var=(10,11))
>>> clf.fit(olympic.data)
HCAV(sup_var=(10,11))
>>> # clustering of variables - divise
>>> clf = VarClus(sup_var=(10,11))
>>> clf.fit(olympic.data)
VarClus(sup_var=(10,11))
"""
olympic = namedtupledocstring(__doc__,"olympic",olympic.keys())(*olympic.values())

#------------------------------------------ poison dataset -----------------------------------------------------
poison = {
    "actif" : read_excel(DATASETS_DIR/"poison.xlsx",sheet_name="Feuil1",index_col=0,header=0),
    "sup_var" : read_excel(DATASETS_DIR/"poison.xlsx",sheet_name="Feuil2",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"poison.xlsx",sheet_name="Feuil3",index_col=0,header=0)
}
__doc__ = """
Poison Dataset

The data used here refer to a survey carried out on a sample of children of primary school who suffered from food poisoning. 
They were asked about their symptoms and about what they ate.
    
Returns
-------
Dataset : poison
    An object with the following attributes:

    actif: DataFrame of shape (55,11)
        Actif data.
    sup_var: DataFrame of shape (55,4)
        Supplementary variables.
    data: DataFrame of shape (55,15)
        Overall dataset.

Examples
--------
>>> from clustvartools.datasets import poison
>>> from clustvartools import CatCLV
>>> clf = CLV(ncl=None,sup_var = range(4))
>>> clf.fit(poison.data)
CLV(ncl=None,sup_var = range(4))
"""
poison = namedtupledocstring(__doc__,"poison",poison.keys())(*poison.values())

#------------------------------------------ uscrime dataset ----------------------------------------------------- 
uscrime = {
    "actif" : read_excel(DATASETS_DIR/"uscrime.xlsx",sheet_name="Feuil1",index_col=None,header=0),
    "sup_var" : read_excel(DATASETS_DIR/"uscrime.xlsx",sheet_name="Feuil2",index_col=None,header=0),
    "data" : read_excel(DATASETS_DIR/"uscrime.xlsx",sheet_name="Feuil3",index_col=None,header=0)
}
__doc__ = """
US Crime Dataset

These data are crime-related and demographic statistics for 47 US states in 1960. The data were collected from the FBI's Uniform Crime Report and other government agencies to determine how the variable crime rate depends on the other variables measured in the study.

    1. Crime.rate: # of offenses reported to police per million population
    2. Male14_24: The number of males of age 14-24 per 1000 population
    3. Southern.states: Indicator variable for Southern states (1, 0)
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
    
Returns
-------
Dataset : canines
    An object with the following attributes:

    actif: DataFrame of shape (47,13)
        Actif data.
    sup_var: DataFrame of shape (47,1)
        Supplementary variables.
    data: DataFrame of shape (47,14)
        Overall dataset.

References
----------
[1] see https://lib.stat.cmu.edu/DASL/Datafiles/USCrime.html

[2] Vandaele, W. (1978) Participation in illegitimate activities: Erlich revisited. In Deterrence and incapacitation, Blumstein, A., Cohen, J. and Nagin, D., eds., Washington, D.C.: National Academy of Sciences, 270-335. Methods: A Primer, New York: Chapman & Hall, 11. Also found in: Hand, D.J., et al. (1994) A Handbook of Small Data Sets, London: Chapman & Hall, 101-103.

Examples
--------
>>> from clustvartools.datasets import uscrime
>>> from clustvartools import HCAV
>>> clf = HCAV(ncl=3,sup_var=0)
>>> clf.fit(uscrime.data)
HCAV(ncl=3,sup_var=0)
"""
uscrime = namedtupledocstring(__doc__,"uscrime",uscrime.keys())(*uscrime.values())

#------------------------------------------ voting dataset -----------------------------------------------------
voting = {
    "actif" : read_excel(DATASETS_DIR/"congressvotingrecords.xlsx",sheet_name="Feuil1",index_col=None,header=0),
    "sup_var" : read_excel(DATASETS_DIR/"congressvotingrecords.xlsx",sheet_name="Feuil2",index_col=None,header=0),
    "data" : read_excel(DATASETS_DIR/"congressvotingrecords.xlsx",sheet_name="Feuil3",index_col=None,header=0)
}
__doc__ = """
Congressional Voting Records

The `Congressional Voting Records <https://archive.ics.uci.edu/dataset/105/congressional+voting+records>`_ UCI dataset.

Returns
-------
Dataset : voting
    An object with the following attributes:

    actif: DataFrame of shape (403,)
        Actif data.
    sup_var: DataFrame of shape (403,)
        Supplementary variables.
    data: DataFrame of shape (403,)
        Overall dataset.

Examples
--------
>>> from clustvartools.datasets import voting
>>> from clustvartools import CatHCAV, CatVARHCA
>>> # clustering of categorical variables
>>> clf = CatHCAV(ncl=2,sup_var=0)
>>> clf.fit(voting.data)
CatHCAV(ncl=2,sup_var=0)
>>> # clustering of categories of categorical variables
>>> clf = CatVARHCA(ncl=2,sup_var=0)
>>> clf.fit(voting.data)
CatVARHCA(ncl=2,sup_var=0)
"""
voting = namedtupledocstring(__doc__,"voting",voting.keys())(*voting.values())

#------------------------------------------ wine dataset -----------------------------------------------------
wine = {
    "actif" : read_excel(DATASETS_DIR/"wine.xlsx",sheet_name="Feuil1",index_col=0,header=0),
    "sup_var" : read_excel(DATASETS_DIR/"wine.xlsx",sheet_name="Feuil2",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"wine.xlsx",sheet_name="Feuil3",index_col=0,header=0)
}
__doc__ = """
Wine Dataset

The data used here refer to :math:`21` wines of Val de Loire and :math:`31` columns:

    * The first column corresponds to the label of origin.
    * The second column corresponds to the soil.
    * and the others correspond to sensory descriptors.
    
Returns
-------
Dataset : wine
    An object with the following attributes:

    actif: DataFrame of shape (21,27)
        Actif data.
    sup_var: DataFrame of shape (21,4)
        Supplementary variables.
    data: DataFrame of shape (21,31)
        Overall dataset.

Examples
--------
>>> from clustvartools.datasets import wine
>>> from clustvartools import CLV
>>> clf = CLV(ncl=4,sup_var = (0,1,29,30))
>>> clf.fit(wine.data)
CLV(ncl=4,sup_var = (0,1,29,30))
"""
wine = namedtupledocstring(__doc__,"wine",wine.keys())(*wine.values())

#------------------------------------------ all dataset -----------------------------------------------------

__all__ = [
    "apples",
    "autos2005",
    "burger",
    "canines",
    "cars",
    "decathlon",
    "jobrate",
    "olympic",
    "poison",
    "uscrime",
    "voting",
    "wine"
]