"""
Taken from:
https://raw.githubusercontent.com/MaksimEkin/COVID19-Literature-Clustering/
"""
from bokeh.models import Div

#header
header = Div(text="""<h1>SOCIAL WORK LITERATURE CLUSTERING</h1>""")

# title for the toolbox
toolbox_header = Div(text="")

# project description
description = Div(text="")

# steps description
description2 = Div(text="")

# citation
cite = Div(text="")

description_search = Div(text="""<h3>Filter by Text:</h3><p1>Search keyword to filter out the plot. It will search abstracts, 
titles, journals, and authors.</p1>""")

description_slider = Div(text="""<h3>Filter by the Clusters:</h3><p1>The slider below can be used to filter the target cluster. 
Simply slide the slider to the desired cluster number to display the plots that belong to that cluster. 
Slide back to 20 to show all.</p1>""")

description_keyword = Div(text="""<h3>Keywords:</h3>""")

description_current = Div(text="""<h3>Selected:</h3>""")

notes = Div(text="")

dataset_description = Div(text="")
