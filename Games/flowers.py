# Object Oriented Programming
# Flowers


# This program creates instances of the flower class every
#	time the mouse is clicked, which are then put in a list
#	and drawn on the screen.

import simplegui
import random
import math

# Global Variables

width = 600
height = 600
flowers = []
colors = ["aqua", "blue", "fuchsia", "maroon", "navy", "olive", "purple", "red", "silver", "teal", "white"]

# Classes

class Flower:
    def __init__(self, pos, radius, color, num_petals, border, angle = 0, angle_vel = 0):
        self.pos = pos
        self.radius = radius
        self.color = color
        self.angle = angle
        self.angle_vel = angle_vel
        self.num_petals = num_petals
        self.border = border
        
    # Causes the flower to rotate 
    def update(self):
        self.angle += self.angle_vel
    
    # Draws a flower using equally spaced circles
    def draw(self, canvas):
        i = 0
        while i < self.num_petals:
            a = (math.pi * 2 / self.num_petals) * i + self.angle
            p = [self.pos[0] + math.cos(a) * self.radius / 3, self.pos[1] + math.sin(a) * self.radius / 3]
            canvas.draw_circle(p, self.radius / 3, 2, self.border, self.color)
            i += 1
        # Draws the center of the flower
        canvas.draw_circle(self.pos, self.radius / 4, 2, "Yellow", "Yellow")

# Helper Functions

def random_color():
    return random.choice(colors)

# Event Handlers

def draw(canvas):
    for f in flowers:
        f.update()
        f.draw(canvas)
       
# Creates a new flower with a random radius (size), color,
#	number of petals, angle, and angular velocity. The 
#	flower's border is white.
def create_flower(pos):
    radius = random.random() * 30 + 20
    num_petals = random.randrange(4, 9)
    angle = random.random() * math.pi
    angular_vel = random.random() * .08 + .01
    flowers.append(Flower(pos, radius, random_color(), num_petals, "White", angle, angular_vel))
    
# Frame

frame = simplegui.create_frame("Flowers :)", width, height)
frame.set_draw_handler(draw)
frame.set_mouseclick_handler(create_flower)
frame.set_canvas_background("Lime")

# Start

frame.start()


