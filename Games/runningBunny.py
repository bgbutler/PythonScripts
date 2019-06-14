# Sprite Animation
# Running Bunny


# This program draws an animated spiral sprite.

import simplegui
import math
import random

# Global Variables

canvas_width = 200
canvas_height = 200

image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/week8-bunny_sprite.png")
image_size = [100, 100]
image_center = [50, 50]
num_tiles = 3
cur_tile = 0
# This image doesn't look good when run at 60 fps, so
#	the delay variable slows it down to 6 fps
delay = 10
# Time keeps track of how many frames have passed since
#	the last tile incrementation.
time = 0

# Event Handlers
        
def draw(canvas):
    global cur_tile, time
    time += 1
    # Once an amount of time has passed equal to the delay,
    #	the program begins to draw the next tile in the
    #	sprite.
    if time == delay:
        time = 0
        cur_tile = (cur_tile + 1) % num_tiles
    # The program calculates the desired center based on
    #	the tile number and image width.
    canvas.draw_image(image, [image_center[0] + cur_tile * image_size[0], image_center[1]], image_size, [canvas_width // 2, canvas_height // 2], image_size)
    
# Frame

frame = simplegui.create_frame("Bouncing Sounds", canvas_width, canvas_height) 

# Register Event Handlers

frame.set_draw_handler(draw)
frame.set_canvas_background("Fuchsia")

# Start
frame.start()