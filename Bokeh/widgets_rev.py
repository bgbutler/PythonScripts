
# import libraries
from bokeh.plotting import figure, curdoc
from bokeh.models.widgets import TextInput, Button, Paragraph
from bokeh.layouts import layout


# create widgets
text_input = TextInput(value='Bryan')
button = Button(label='Generate Text')
output = Paragraph()


# create a function to update
def update():

    output.text = 'Hello, ' + text_input.value


button.on_click(update)

lo = layout([[button, text_input], [output]])

curdoc().add_root(lo)
