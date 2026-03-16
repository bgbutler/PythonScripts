# Dictionaries
# Dictionary Colors


# This program allows the user to draw shapes on the canvas
#	whose colors are based off of a dictionary. Click to 
#	draw shapes on the canvas.

import simplegui

# Global Variables

canvas_width = 400
canvas_height = 400
shapes = ["triangle", "circle", "square"]
shape_index = 0
shape_colors = {"triangle": "red", "circle": "blue", "square": "green"}
# Contains a list of valid colors.
colors = {"aqua": True, "black": True, "blue": True, "fuchsia": True, "gray": True, "green": True, "lime": True, "maroon": True, "navy": True, "olive": True, "purple": True, "red": True, "silver": True, "teal": True, "white": True, "yellow": True}
# Keeps track of shapes to be drawn on canvas
drawn_shapes = []

# Event Handlers
        
def draw(canvas):
    for s in drawn_shapes:
        # Draws the shapes using the stored type (s[0]) and
        #	position (s[1]) and using the current color stored
        #	 in shape_colors.
        if s[0] == "triangle":
            points = [[s[1][0] - 10, s[1][1] + 6], [s[1][0] + 10, s[1][1] + 6], [s[1][0], s[1][1] - 9]]
            canvas.draw_polygon(points, 2, shape_colors["triangle"])
        elif s[0] == "circle":
            canvas.draw_circle(s[1], 10, 2, shape_colors["circle"])
        elif s[0] == "square":
            points = [[s[1][0] - 10, s[1][1] - 10], [s[1][0] + 10, s[1][1] - 10], [s[1][0] + 10, s[1][1] + 10], [s[1][0] - 10, s[1][1] + 10]]
            canvas.draw_polygon(points, 2, shape_colors["square"])
        else:
            print "Error: Invalid shape: " + str(s[0])
    canvas.draw_text("Current shape: " + str(shapes[shape_index]), [30, 50], 20, "White")
    
def click(pos):
    drawn_shapes.append([shapes[shape_index], pos])
    
def rotate_shape():
    global shape_index
    shape_index = (shape_index + 1) % len(shapes)
    
def set_tri_color(color):
    if colors.get(color):
        shape_colors["triangle"] = color.lower()
        print "Triangle color changed to " + color.lower() + "."
        
def set_circ_color(color):
    if colors.get(color):
        shape_colors["circle"] = color.lower()
        print "Circle color changed to " + color.lower() + "."

def set_square_color(color):
    if colors.get(color):
        shape_colors["square"] = color.lower()
        print "Square color changed to " + color.lower() + "."
    
# Frame

frame = simplegui.create_frame("Dictionary Colors", canvas_width, canvas_height) 
frame.set_draw_handler(draw)
frame.set_mouseclick_handler(click)
frame.add_button("Rotate Shape", rotate_shape, 100)
frame.add_input("Triangle Color", set_tri_color, 100)
frame.add_input("Circle Color", set_circ_color, 100)
frame.add_input("Square Color", set_square_color, 100)

# Start
frame.start()