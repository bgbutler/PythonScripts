# Acceleration
# Balancing


# This program demonstrates angular acceleration by asking
#	the user to attempt to balance a stick using the arrow keys.
# The code using acceleration can be found in the update method
#	of the Stick class.

import simplegui
import math
import random

# Global Variables

canvas_width = 400
canvas_height = 400
# Note that the center of the image is located at the black
#	circle at the base of the stick.
stick_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/week7-stick.png")

# Classes

class Stick:
    def __init__(self, radius, center, image, image_center, image_size):
        self.radius = radius
        self.center = center
        self.angle = random.random() * random.choice([-1, 1]) * .01
        self.angle_vel = 0
        self.angle_acc = 0
        self.key_acc = 0
        self.image = image
        self.image_center = image_center
        self.image_size = image_size
        
    # Draws the image. Does not perform any calculations.
    def draw(self, canvas):
        scale = 2
        canvas.draw_image(self.image, self.image_center, self.image_size, self.center, [self.image_size[0] * scale, self.image_size[1] * scale], self.angle)
     
    # Updates the state of the object using self.angle_vel
    #	and self.angle_acc. The value of self.angle_acc 
    #	depends on whether or not the user is holding
    #	down an arrow key.
    def update(self):
        self.angle += self.angle_vel
        # Checks to see if the stick fell down too far
        #	(if this is not checked, self.get_gravity() 
        #	may return an error)
        if self.angle <= - math.pi / 4 or self.angle >= math.pi / 4:
            self.reset()
            
        # Updates the angle_vel using the acceleration.
        self.angle_vel += self.angle_acc
        self.angle_acc = self.get_gravity() + self.key_acc
        
    # Used to calculate the acceleration due to gravity
    #	of the stick. You don't need to worry about this
    #	for the project or this class.
    def get_gravity(self):
        if self.angle == 0:
            return 0
        else:
            return math.sin(self.angle) * .001
        
    # Used to change the acceleration due to the user
    def set_key_acc(self, a):
        self.key_acc = a
        
    def reset(self):
        self.angle = random.random() * random.choice([-1, 1]) * .1
        self.angle_vel = 0
        self.angle_acc = 0
        self.key_acc = 0
    
# Creating Class Instances

stick = Stick(50, [canvas_width // 2, canvas_height // 2], stick_image, [50, 50], [101, 101])

# Event Handlers
        
# Note that this draw function must draw the stick in addition
#	to telling it to update itself.
def draw(canvas):
    stick.update()
    stick.draw(canvas)
    canvas.draw_line((0, canvas_height // 2), (canvas_width, canvas_height // 2), 3, "Black")

def reset():
    stick.reset()
    
# Sets the key_acc of the stick depending on 
#	the keys pressed.
def keydown_handler(key):
    acc = .001
    if key == simplegui.KEY_MAP['left']:
        stick.set_key_acc(-acc)
    elif key == simplegui.KEY_MAP['right']:
        stick.set_key_acc(acc)
        
def keyup_handler(key):
    stick.set_key_acc(0)
    
# Frame and Timer

frame = simplegui.create_frame("Balance", canvas_width, canvas_height) 

# Register Event Handlers

frame.set_draw_handler(draw)
frame.set_keydown_handler(keydown_handler)
frame.set_keyup_handler(keyup_handler)
frame.set_canvas_background("Green")
frame.add_button("Reset", reset)
frame.add_label("Use the left and right arrow keys to balance the stick!")

# Start
frame.start()