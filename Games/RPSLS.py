import random

def name_to_number(name):
    # converts the names to numbers
    
        if name == "rock":
            name_to_number = int(0)
        elif name == "Spock":
            name_to_number = int(1)
        elif name == "paper":
            name_to_number = int(2)
        elif name == "lizard":
            name_to_number = int(3)
        elif name == "scissors":
            name_to_number = int(4)

        return name_to_number  
            
def number_to_name(number):
    #converts the computer numbers into names
    if number == 0:
        name = "rock"
    elif number == 1:
        name = "Spock"
    elif number == 2:
        name = "paper"
    elif number ==3:
        name = "lizard"
    else:
        name = "scissors"
   
    return name

def rpsls():
#asks the player to choose

    option = ["rock", "paper", "scissors", "lizard", "Spock"]

    rpsls_player = input ("Choose rock, paper, scissors, lizard, Spock:")
    print
    
    if rpsls_player not in option:
        print ("That is not a valid choice, try again")
    
    player = name_to_number(rpsls_player)

#computer chooses randomly    
    computer_choice = int(random.randrange(0,5))
    computer = number_to_name(computer_choice)
 
#show selections    
    print("Player chooses "+ rpsls_player)
    print("Computer chooses "+ computer)
    
   
#calculate winner    
    winner = player - computer_choice
    
    if winner == 0:
        print("The game is a tie")
     
    if winner < 0:
       newwinner = winner % 5
         
       if newwinner >2:
            print("The computer is the winner.")
       else:
            print("The player is the winner.")    
                
    if winner == 1 or winner ==2:
        print("The player is the winner.")
        
    elif winner ==3 or winner ==4:
        print("The computer is the winner.")
    
rpsls()     

