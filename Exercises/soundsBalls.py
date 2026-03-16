# Sound
# Bouncing Sounds


# This program draws a ball that makes sounds when it bounces
#	off the side of the canvas.

import simplegui
import math
import random

# Global Variables

canvas_width = 400
canvas_height = 400

sound_a = simplegui.load_sound("http://commondatastorage.googleapis.com/codeskulptor-assets/week7-bounce.m4a")
sound_b = simplegui.load_sound("http://commondatastorage.googleapis.com/codeskulptor-assets/week7-brrring.m4a")
sound_c = simplegui.load_sound("http://commondatastorage.googleapis.com/codeskulptor-assets/week7-button.m4a")

sounds = [sound_a, sound_b, sound_c]
sound_index = 0

# Classes
 
class Ball:
    def __init__(self, radius, color, bounds, sound):
        self.radius = radius
        self.pos = [bounds[0] // 2, bounds[1] // 2]
        self.vel = self.get_random_vel()
        self.color = color
        self.bounds = bounds
        self.sound = sound
        
    # Draws the ball. Does not perform calculations 
    #	or checks.
    def draw(self, canvas):
        canvas.draw_circle(self.pos, self.radius, 2, "White", self.color)
     
    # Updates the position of the ball. If the ball goes
    #	out of bounds, its velocity is reversed and its
    #	current sound is rewound and played.
    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        
        # Collision check and logic
        if self.pos[0] - self.radius < 0 or self.pos[0] + self.radius > self.bounds[0]:
            self.vel[0] = self.vel[0] * -1
            # Rewinds and plays the sound.
            self.sound.rewind()
            self.sound.play()
        if self.pos[1] - self.radius < 0 or self.pos[1] + self.radius > self.bounds[1]:
            self.vel[1] = self.vel[1] * -1
            # Rewinds and plays the sound.
            self.sound.rewind()
            self.sound.play()
        
    def get_random_vel(self):
        magnitude = random.random() * 3 + 2
        angle = random.random() * (math.pi * 2)
        return [magnitude * math.cos(angle), magnitude * math.sin(angle)]
        
    # Sets the sound that the ball uses
    def set_sound(self, s):
        self.sound = s
        
    def reset(self):
        self.pos = [self.bounds[0] // 2, self.bounds[1] // 2]
        self.vel = self.get_random_vel()
    
# Creating Class Instances

ball = Ball(25, "Red", [canvas_width, canvas_height], sounds[sound_index])

# Helper Functions

# Re-assigns the ball's sound based on the
#	new sound_index
def change_sound(sign):
    global sound_index
    sound_index = (sound_index + sign) % len(sounds)
    ball.set_sound(sounds[sound_index])

# Event Handlers
        
def draw(canvas):
    ball.update()
    ball.draw(canvas)
    
def keydown_handler(key):
    if key == simplegui.KEY_MAP["left"]:
        change_sound(-1)
    elif key == simplegui.KEY_MAP["right"]:
        change_sound(1)

def reset():
    ball.reset()
    
# Frame and Timer

frame = simplegui.create_frame("Bouncing Sounds", canvas_width, canvas_height) 

# Register Event Handlers

frame.set_draw_handler(draw)
frame.set_keydown_handler(keydown_handler)
frame.add_button("Reset", reset)
frame.add_label("Use the left and right arrow keys to change the sound!")

# Start
frame.start()