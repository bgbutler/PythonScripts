# Vectors
# Angle Vectors


# This program draws a vector when given an angle.

import simplegui
import math
import random

# Global Variables

canvas_width = 400
canvas_height = 400
angle = 0      
length = 150
    
# Helper Functions

# This returns the x and y components
#	of a vector. When added to the origin
#	of the vector, the result is the other
#	endpoint of the vector.
def get_vector():
    x = math.cos(angle) * length
    y = math.sin(angle) * length
    return [x, y]

def print_angle():
    print "Angle in Radians: " + str(angle)
    # Note that the math module has a function that
    #	converts radians to degrees.
    print "Angle in Degrees: " + str(math.degrees(angle))
    print

# Event Handlers
        
def draw(canvas):
    canvas.draw_text("Angle: " + str(angle), [10, 25], 25, "White")
    
    # Draws the axis
    center = [canvas_width // 2, canvas_height // 2]
    canvas.draw_line((0, center[1]), (canvas_width, center[1]), 3, "White")
    canvas.draw_line((center[0], 0), (center[0], canvas_height), 3, "White")
    
    # Gets that magnitude of the x and y
    #	components of the vector.
    v = get_vector()
    canvas.draw_line((center[0], center[1]), (center[0] + v[0], center[1] + v[1]), 5, "Red")
    
def set_r_angle(text):
    global angle
    angle = float(text)
    print_angle()
    
def set_d_angle(text):
    global angle
    # Note that the math module has a function that
    #	converts degrees to radians.
    angle = math.radians(float(text))
    print_angle()
    
# Frame and Timer

frame = simplegui.create_frame("Vectors", canvas_width, canvas_height) 

# Register Event Handlers

frame.set_draw_handler(draw)
frame.add_input("Angle (in radians)", set_r_angle, 100)
frame.add_input("Angle (in degrees)", set_d_angle, 100)

# Start
frame.start()