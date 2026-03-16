#Plotting periodic table elements

# import libraries
from bokeh.plotting import figure, curdoc
from bokeh.io import show
from bokeh.models.annotations import LabelSet, Label, Span, BoxAnnotation
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import RadioButtonGroup
from bokeh.layouts import layout

from bokeh.sampledata.periodic_table import elements
from bokeh.models import Range1d, PanTool, ResetTool, HoverTool, ColumnDataSource, Label, LabelSet

import pandas

#Remove rows with NaN values and then map standard states to colors
elements.dropna(inplace=True) #if inplace is not set to True the changes are not written to the dataframe
colormap = {'gas':'yellow', 'liquid':'orange', 'solid':'red'}
elements['color'] = [colormap[x] for x in elements['standard state']]
elements['size'] = elements["van der Waals radius"] / 10

#Create three ColumnDataSources for elements of unique standard states
gas = ColumnDataSource(elements[elements['standard state']=='gas'])
liquid = ColumnDataSource(elements[elements['standard state']=='liquid'])
solid = ColumnDataSource(elements[elements['standard state']=='solid'])

#Create the figure object
f = figure()

#adding glyphs
f.circle(x="atomic radius", y="boiling point", size='size', fill_alpha=0.2, color="color",
         legend_label='Gas',source=gas)

f.circle(x="atomic radius", y="boiling point", size='size', fill_alpha=0.2, color="color",
         legend_label='Liquid', source=liquid)

f.circle(x="atomic radius", y="boiling point", size='size', fill_alpha=0.2, color="color",
         legend_label='Solid', source=solid)

# add labels
f.xaxis.axis_label="Atomic radius"
f.yaxis.axis_label="Boiling point"

# create average boiling point
gas_average_boil = sum(gas.data['boiling point']) / len(gas.data['boiling point'])
liquid_average_boil = sum(liquid.data['boiling point']) / len(liquid.data['boiling point'])
solid_average_boil = sum(solid.data['boiling point']) / len(solid.data['boiling point'])

# add additional values for max and min
solid_min_boil = min(solid.data['boiling point'])
solid_max_boil = max(solid.data['boiling point'])

# make the spans at average boiling point
span_gas_average_boil = Span(location=gas_average_boil, dimension='width', line_color='yellow', line_width=2)
span_liquid_average_boil = Span(location=liquid_average_boil, dimension='width', line_color='orange', line_width=2)
span_solid_average_boil = Span(location=solid_average_boil, dimension='width', line_color='red', line_width=2)

# add the spans
# add the instance
f.add_layout(span_gas_average_boil)
f.add_layout(span_liquid_average_boil)
f.add_layout(span_solid_average_boil)

#Add labels to spans
label_span_gas_average_boil=Label(x=80, y=gas_average_boil, text="Gas average boiling point", render_mode="css",
                                 text_font_size="10px")
label_span_liquid_average_boil=Label(x=80, y=liquid_average_boil, text="Liquid average boiling point", render_mode="css",
                                    text_font_size="10px")
label_span_solid_average_boil=Label(x=80, y=solid_average_boil, text="Solid average boiling point", render_mode="css",
                                   text_font_size="10px")

#Add labels to figure
f.add_layout(label_span_gas_average_boil)
f.add_layout(label_span_liquid_average_boil)
f.add_layout(label_span_solid_average_boil)


# create function
def update_span(attr, old, new):
    # for radio button radio_button_group.active
    span_solid_boil.location=float(select.value)



# add select widgets
# options can be a list
options=[(str('solid_average_boil'),'Solid Average Boiling Pt'), (str('solid_min_boil'),'Solid Minimum Boiling Pt'), (str('solid_max_boil'),'Solid Max Boilt Pt')]
# radio_button_group=RadioButtonGroup(title='Span Values', labels=options)


select = Select(title='Span Values', options=options)

# select.on_change('active', update_span)
select.on_change('value', update_span)

lay_out=layout([[select]])

# curdoc is current document
curdoc().add_root(f)
curdoc().add_root(lay_out)
