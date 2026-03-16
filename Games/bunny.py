# Tiled Images
# Bunny Emotions


# This program draws a bunny and allows the user to pet or
#	poke it, which changes its emotion. The user can also
#	switch the color of the bunny between brown and white.

import simplegui

# Global Variables

canvas_width = 200
canvas_height = 200

# It might be helpful to look at the full image to better
#	understand what is going on.
image_center = (12, 12)
image_size = (24, 24)
image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/week6-emotions.png")
emotions = ["happy!", "indifferent.", "sad..."]
emotions_index = 1
color = 0

# Event Handlers
        
def draw(canvas):
    # The horizontal tile of the image is determined by the
    #	emotions_index, while the vertical tile of the image
    #	is determined by the color.
    tile_center = [image_center[0] + emotions_index * (image_size[0] + 1), image_center[1] + color * (image_size[1] + 1)]
    canvas.draw_image(image, tile_center, image_size, [canvas_width / 2, canvas_height / 2], [100, 100])
    
def pet():
    global emotions_index
    if emotions_index > 0:
        emotions_index -= 1    
    print "You petted the bunny!"
    print "The bunny is now " + emotions[emotions_index]
    print

def poke():
    global emotions_index
    # Note: len(emotions) should be the same as the number
    #	of tiles per row in the image.
    if emotions_index < len(emotions) - 1:
        emotions_index += 1
    print "You poked the bunny. That wasn't nice."
    print "The bunny is now " + emotions[emotions_index]
    print
        
def switch_color():
    global color
    color = 1 - color
    
# Frame

frame = simplegui.create_frame("Bunny Emotions", canvas_width, canvas_height) 

# Register Event Handlers

frame.set_draw_handler(draw)
frame.add_button("Pet", pet, 100)
frame.add_button("Poke", poke, 100)
frame.add_button("Switch Color", switch_color, 100)
frame.set_canvas_background("Green")

# Start
frame.start()