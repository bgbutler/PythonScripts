# Mouseclick Handlers
# Tic-Tac-Toe


# This program allows two users to play tic-tac-toe using
#	mouse input to the game.

import simplegui

# Global Variables

canvas_width = 300
canvas_height = 300
grid = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]
turn = "X"
won = False

# Helper Functions

def switch_turn():
    global turn, info
    if turn == "X":
        turn = "O"
    else:
        turn = "X"
    info.set_text("Player turn: " + turn)
        
# Returns 'True' if a player has won, false otherwise
def check_win():
    for a in range(0,3):
        if grid[a][0] != " " and grid[a][0] == grid[a][1] == grid[a][2]:
            return True
    for b in range(0,3):
        if grid[0][b] != " " and grid[0][b] == grid[1][b] == grid[2][b]:
            return True
    if grid[0][0] == grid[1][1] == grid[2][2] and grid[0][0] != " ":
        return True
    elif grid[0][2] == grid[1][1] == grid[2][0] and grid[0][2] != " ":
        return True
    else:
        return False

# Event Handlers
        
def draw(canvas):
    # Draws the grid lines
    canvas.draw_line([0, canvas_height // 3], [canvas_width, canvas_height // 3], 1, "White")
    canvas.draw_line([0, canvas_height // 3 * 2], [canvas_width, canvas_height // 3 * 2], 1, "White")
    canvas.draw_line([canvas_width // 3, 0], [canvas_width // 3, canvas_height], 1, "White")
    canvas.draw_line([canvas_width // 3 * 2, 0], [canvas_width // 3 * 2, canvas_height], 1, "White")
    
    # Draws the player choices using loops
    for r in range(0,3):
        for c in range(0,3):
            canvas.draw_text(grid[r][c], [c * canvas_width // 3 + 20, r * canvas_height // 3 + 80], 80, "Red")
            
def click(pos):
    global won, info
    if not won:
        # Checks to see if the click was on a grid line
        if pos[0] % (canvas_width // 3) != 0 and pos[1] % (canvas_height // 3) != 0:
            r = pos[1] // (canvas_height // 3)
            c = pos[0] // (canvas_width // 3)
            # Checks to see if a square is already taken
            if grid[r][c] == " ":
                grid[r][c] = turn
                if check_win():
                    won = True
                    info.set_text("Player " + turn + " wins!")
                else:
                    switch_turn()

def reset():
    global grid, turn, won, info
    grid = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]
    turn = "X"
    won = False
    info.set_text("Player turn: " + turn)
    
# Frame

frame = simplegui.create_frame("Tic-Tac-Toe", canvas_width, canvas_height) 

# Register Event Handlers

frame.set_draw_handler(draw)
frame.set_mouseclick_handler(click)
frame.add_button("Reset", reset)
info = frame.add_label("Player turn: " + turn)

# Start
frame.start()