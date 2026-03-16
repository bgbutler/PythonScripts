# Object Oriented Programming
# Bubbles


# This program creates bubbles at random on the bottom of the
#	screen and at clicked locations on the screen. The 
#	bubbles then accelerate toward the top of the screen.

import simplegui
import random
import math

# Global Variables

width = 600
height = 600
bubbles = []

# Classes

class Bubble:
    def __init__(self, pos, radius, color):
        self.pos = list(pos)
        self.radius = radius
        self.color = color
        # Note that the user does not input or influence the
        #	velocity in any way because the __init__ method
        #	does not use a given parameter to initialize it.
        self.vel = [0, 0]
        # The user never inputs the acceleration either.
        #	The initializer calculates it based on the given
        #	radius.
        self.accel = [0, radius ** 3 / radius ** 2 * -0.001]
        
    def update(self):
        self.vel[0] += self.accel[0]
        self.vel[1] += self.accel[1]
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
    
    def draw(self, canvas):
        canvas.draw_circle(self.pos, self.radius, 2, self.color)
    
    # These methods allow the rest of the program to get 
    #	information about the state of the bubble.
    def get_pos(self):
        return self.pos
        
    def get_radius(self):
        return self.radius
        
# Event Handlers

def draw(canvas):
    for b in list(bubbles):
        b.update()
        b.draw(canvas)
        # Removes bubbles if they leave the screen to limit
        #	the number of objects being updated and drawn
        #	(If this is removed, the program gradually slows
        #	down. Feel free to try)
        if b.get_pos()[1] + b.get_radius() < 0:
            bubbles.remove(b)
    generate_bubbles()
        
# Creates a bubble with a random radius at the specified
#	position. Note that this is both the specified mouseclick
#	handler and also called by the generate_bubbles() function.
def create_bubble(pos):
    r = random.random() * 20 + 10
    bubbles.append(Bubble(pos, r, "Aqua"))
  
# Has a chance of creating a bubble at a random position
#	along the bottom of the screen.
def generate_bubbles():
    if random.random() < .07:
        pos = [random.random() * width, height]
        create_bubble(pos)
    
# Frame

frame = simplegui.create_frame("Bubbles :)", width, height)
frame.set_draw_handler(draw)
frame.set_mouseclick_handler(create_bubble)
frame.set_canvas_background("Blue")

# Start

frame.start()

