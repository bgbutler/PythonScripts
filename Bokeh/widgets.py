
# import libraries
from bokeh.io import output_file, show
from bokeh.models.widgets import TextInput, Button, Paragraph
from bokeh.layouts import layout

# prepare the bokeh output file
output_file('simple_bokeh.html')

# create widgets
text_input = TextInput(value='Bryan')
button = Button(label='Generate Text')
output = Paragraph()


# create a function to update
def update():

    output.text = 'Hello, ' + text_input


button.on_click(update)
lo = layout([[button, text_input], [output]])

show(lo)
