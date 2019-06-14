# Dictionaries
# Secret Code


# This replaces select words in a phrase with other 'code
#	words' from a dictionary. Uncomment the various text
#	examples to try them out. Feel free to expand the code
#	or change the words to create your own version!

import simplegui

# Global Variables

code_words = {"hi": "bye", "president": "eagle", "target": "butterfly", "code": "obvious word", "impossible": "super easy"}
text = "Hi is my favorite word."
#text = "The President is on his way."
#text = "The target is in range."
#text = "The code is impossible to solve."
text = "The president is looking for the target but it might be impossible to find."

# Helper Functions
        
def translate():
    # This method separates a string into a list based on
    #	whitespace (more or less into words)
    text_list = text.split()
    index = 0
    for word in text_list:
        if word.lower() in code_words:
            text_list[index] = code_words[word.lower()]
            if index == 0:
                text_list[index] = text_list[index].capitalize()
        index += 1
    answer = ""
    for word in text_list:
        answer += word + " "
    print answer
    
translate()