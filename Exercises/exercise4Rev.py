temperatures = [10, -20, -289, 100]


# convert fahrenheit to celsius

def c_to_f(c):

    f = c*9/5 + 32

    return f


for temp in temperatures:
    if temp > -273.3:
        print(c_to_f(temp))
    else:
        print('That temperature does not make sense!')

