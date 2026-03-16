# Friction
# Curling


# This program demonstrates the use of friction through
#	a simple reflex videogame similar to curling. The user 
#	presses the spacebar to release the ball with an initial
#	velocity in an attempt to have it land on the target.
# The code using friction can be found in the update method
#	of the Ball class.

import simplegui
import math

# Global Variables

canvas_width = 800
canvas_height = 500
target_center = [600, 250]
started = False
# Contains the last five scores
last_five = [0, 0, 0, 0, 0]
starting_vel = 0

# Classes
 
class Ball:
    def __init__(self, radius, start, color, friction):
        self.radius = radius
        self.start = start
        self.pos = list(start)
        self.vel = [0, 0]
        self.friction = friction
        self.color = color
        
    def __str__(self):
        a = "BALL" + "\n"
        a += "Radius: " + str(self.radius) + "\n"
        a += "Starting Position: " + str(self.start) + "\n"
        a += "Position: " + str(self.pos) + "\n"
        a += "Velocity: " + str(self.vel) + "\n"
        a += "Friction: " + str(self.friction) + "\n"
        a += "Color: " + str(self.color) + "\n"
        return a
        
    # The draw method does not perform any calculations; it
    #	only draws the object.
    def draw(self, canvas):
        canvas.draw_circle(self.pos, self.radius, 2, "Black", self.color)
     
    # The update method is in charge of updating the state
    #	of the object, including its position and velocity.
    # This update method uses friction to slow the object.
    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        # Updates the velocity using friction.
        self.vel[0] *= self.friction
        self.vel[1] *= self.friction
     
    # get_pos and get_vel are accessor methods that return
    #	information about the object.
    def get_pos(self):
        return self.pos
        
    def get_vel(self):
        return self.vel
        
    # set_vel is a mutator method used to change the state
    #	of the object.
    def set_vel(self, vel):
        self.vel = vel
        
    def reset(self):
        self.pos = list(self.start)
        self.vel = [0, 0]
        
class Target:
    def __init__(self, num_rings, radius, colors, center):
        self.num_rings = num_rings
        # Initializes the radii from smallest to largest
        self.radii = []
        for i in range(num_rings):
            self.radii.append((i + 1) * radius / num_rings)
        self.colors = colors
        self.center = center
    
    def __str__(self):
        a = "TARGET" + "\n"
        a += "Number of Rings: " + str(self.num_rings) + "\n"
        a += "Radii: " + str(self.radii) + "\n"
        a += "Colors: " + str(self.colors) + "\n"
        a += "Center: " + str(self.center) + "\n"
        return a
    
    def draw(self, canvas):
        for i in range(self.num_rings):
            # The radii are listed from smallest to largest, but 
            #	the circles must be drawn from largest to smallest
            #	if all of them are to be visible.
            index = self.num_rings - i - 1
            canvas.draw_circle(self.center, self.radii[index], 2, "Black", self.colors[index])
     
    def get_score(self, pos):
        for r in self.radii:
            if self.distance(pos) < r:
                return (self.num_rings - self.radii.index(r)) * 10
        return 0
            
    def distance(self, point):
        return math.sqrt((point[0] - self.center[0]) ** 2 + (point[1] - self.center[1]) ** 2)
    
# Creating Class Instances

target = Target(3, 100, ["Red", "Yellow", "Blue"], [675, 250])
ball = Ball(25, [100, 250], "Black", .97)

# Event Handlers
        
def draw(canvas):
    global started, starting_vel
    
    # Displays the scores on the screen and calculates the total
    score_text = "Last five scores: "
    total = 0
    for s in last_five:
        score_text += str(s) + " "
        total += s
    score_text += " Total: " + str(total)
    canvas.draw_text(score_text, [50, 75], 30, "White")
    
    # Draws the velocity bar
    canvas.draw_polygon([(50, 400), (50, 425), (50 + starting_vel * 10, 425), (50 + starting_vel * 10, 400)], 2, "Black", "Aqua")
    
    target.draw(canvas)
    
    # If the update method is not called, the ball will not move.
    #	You can comment the line out to see for yourself.
    ball.update()
    # If the draw method is commented out instead, the game functions
    #	as expected, but the ball is not visible.
    ball.draw(canvas)
    
    if started:
        if ball.get_vel()[0] < .01:
            # Checks the position of the center of the ball and updates the score
            last_five.append(target.get_score(ball.get_pos()))
            last_five.pop(0)
            ball.reset()
            starting_vel = 0
            started = False
    else:
        starting_vel += .2

def reset():
    global last_five
    ball.reset()
    last_five = [0, 0, 0, 0, 0]
    
def keydown_handler(key):
    global started
    if key == simplegui.KEY_MAP['space'] and not started:
        ball.set_vel([starting_vel, 0])
        started = True
    
# Frame and Timer

frame = simplegui.create_frame("Curling", canvas_width, canvas_height) 

# Register Event Handlers

frame.set_draw_handler(draw)
frame.set_keydown_handler(keydown_handler)
frame.set_canvas_background("Green")
frame.add_button("Reset", reset)
frame.add_label("Push the space bar to send the ball flying!")

# Start
frame.start()