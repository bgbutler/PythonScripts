def rpsls():

    option = ["rock", "paper", "scissors", "lizard", "Spock"]
    
    choice = input ("Choose rock, paper , scissors, lizard, Spock:")
    i = 0       
    
    if choice in option:
        print choice    
    else:    
        print ("That is not a valid choice, try again")
        i = i + 1
        if i == 3:
              quit(rpsls)
        rpsls()
        
        
rpsls()    