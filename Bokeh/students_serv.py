# import libraries
from bokeh.plotting import figure, curdoc
from bokeh.io import show
from bokeh.models.annotations import LabelSet, Label
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Select
from bokeh.layouts import layout





source = ColumnDataSource(data=dict(average_grades=['B+', 'A', 'D-'],
                                   exam_grades=['A+', 'C', 'D'],
                                   students=['Stephan', 'Helder', 'Riazudidn']))

# create the figure
f = figure(x_range=['F', 'D-', 'D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'],
          y_range=['F', 'D-', 'D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'])



# add labels for glyphs
labels=LabelSet(x='average_grades',
               y='exam_grades',
               text='students',
               x_offset=5,
               y_offset=5,
               source=source)


f.add_layout(labels)

# add labels for glyphs
f.circle(x='average_grades', y='exam_grades', size=8, source=source)

# create function
def update_labels(attr, old, new):
    labels.text=select.value



# add select widgets
# options can be a list
options=[('average_grades','Average Grades'), ('exam_grades','Exam Names'), ('students','Student Names')]
select = Select(title='Attribute', options=options)
select.on_change('value', update_labels)

lay_out=layout([[select]])

curdoc().add_root(f)
curdoc().add_root(lay_out)
