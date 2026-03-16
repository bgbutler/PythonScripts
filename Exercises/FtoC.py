

# convert fahrenheit to celsius

def c_to_f(c):
    f = c*9/5 + 32
    return f


def get_input_celsius():
    get_celsius = input('Input a celsius value to convert to F: ')
    print("You entered " + get_celsius)
    deg_c = float(get_celsius)


    converted = c_to_f(deg_c)

    # for printing the floats must be converted to strings
    print(get_celsius + " C is converted to " + str(converted) + 'F')


get_input_celsius()
