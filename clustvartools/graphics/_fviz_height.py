# -*- coding: utf-8 -*-
from pandas import DataFrame, Categorical
from plotnine import (
    ggplot,
    geom_bar, 
    geom_line, 
    geom_point, 
    geom_text, 
    aes, 
    theme_minimal, 
    labs,
    ylim,
    element_text,
    theme
)

def fviz_height(obj,
                geom = ("bar","line"),
                col_bar = "steelblue",
                bar_args = {"fill":"steelblue","width":None},
                col_line = "black",
                line_args = {},
                show_labels = False,
                y_lim = None,
                x_label = None,
                y_label = None,
                title = None,
                subtitle = None,
                pntheme = theme_minimal(),
                **kwargs):
    """
    Draw Height
    
    Parameters
    ----------
    obj : class
        An object of class which have ``tree``.

    geom : str, list, tuple, default = ("bar","line")
        The geometry to be used for the graph. Allowed values are the combinaison of ("bar","line"). 

        * "bar" to show only bar.
        * "line" to show only line.
        * ("bar", "line") to use both types.

    col_bar : str, default = "steelblue"
        Outline color for the bar plot.

    bar_args : dict, default = {"fill":"steelblue","width":None}
        A dictionary containing parameters (except color) for bar plot (see `plotnine.geom_bar <https://plotnine.org/reference/geom_bar.html>`_).

    col_line, str, default = "black"
        Color for the line plot.

    line_args : dict, default = {}
        A dictionary containing parameters (except color) for line plot (see `plotnine.geom_line <https://plotnine.org/reference/geom_line.html>`_).

    show_labels : bool, default = False
        If True, labels are added at the top of bars or points showing the information retained by each dimension.

    y_lim : list, tuple, default = None
        The range of the plotted y values.

    x_label : str, default = None
        The label text of x. If None, then x_label is chosen.
    
    y_label : str, default = None
        The label text of y. If None, then y_label is chosen.

    title : str, default = None
        The title of the graph you draw. If None, then a title is chosen.
    
    subtitle : str, default = None
        The subtitle of the graph you draw.
    
    pntheme : function, default = theme_minimal() 
        Plotnine theme name. Allowed values include plotnine official themes (see `themes <https://plotnine.org/guide/themes-premade.html>`_).

    **kwargs : Any
        Parameters use by `plotnine.theme <https://plotnine.org/reference/theme.html#plotnine.theme>`_.
    
    Returns
    -------
    A plotnine object.

    See also
    --------
    fviz_dend : Visualization of Dendrogram
    fviz_dend2 : Visualization of Dendrogram
    
    Examples
    --------
    >>> from clustvartools.datasets import decathlon
    >>> from clustvartools import CLV, fviz_height
    >>> clf = CLV(ncl=3)
    >>> clf.fit(decathlon.actif)
    CLV(ncl=3)
    >>> p = fviz_height(clf)
    >>> print(p.show())
    
    .. figure:: ../_static/fviz_height.png
        
            Histogram
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if obj contains tree
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not ("tree" in obj.call_._fields):
        raise TypeError(f"Not allowed to an object of class {obj.__class__.__name__}")
    
    # extract linkage matrix
    Z = obj.call_.tree.Z
    # number of elements
    n = Z.shape[0] + 1
    index = [f"({n+i+1}) {j} -> {j-1}" for i,j in enumerate(range(n, 1, -1))]
    text_labels = [str(round(x,3)) for x in Z[:,2]]
    # convert to pd.DataFrame
    df_height = DataFrame({"x" : Categorical(index,categories=index,ordered=True),"y" : Z[:,2]})
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # plot
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # initialization
    p = ggplot(df_height,aes(x="x",y="y",group=1))

    # show barplot
    if "bar" in geom :
        p = p +  geom_bar(stat="identity",color=col_bar,**bar_args)
    # show line
    if "line" in geom :
        p = (
            p 
            + geom_line(color=col_line,**line_args) 
            + geom_point(color=col_line)
        )
    
    # show labels
    if show_labels:
        p = p + geom_text(label=text_labels,ha="center",va="bottom")
    
    # set x_label
    if x_label is None:
        x_label = "Numéro du noeud"
    # set y_label
    if y_label is None:
        y_label = "Niveau d'aggrégation"
    # set title
    if title is None:
        title = "Diagramme des indices de niveau"
    # set subtitle
    if subtitle is None:
        subtitle = ""
    p = p + labs(x=x_label, y=y_label, title=title, subtitle=subtitle)
    # set y_lim
    if y_lim is not None:
        p = p + ylim(y_lim)

    # add theme
    p = p + pntheme
    
    # theme customization
    theme_dict = {"axis_text_x" : element_text(rotation=90, hjust=-1)}
    if kwargs is not None:
        theme_dict = {**theme_dict,**kwargs}
    p = p + theme(**theme_dict)
    return p
    