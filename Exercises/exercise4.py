temperatures = [10, -20, -289, 100]


# convert fahrenheit to celsius

def c_to_f(c):
    if c > -273.3:
        f = c*9/5 + 32
        return f
    else:
        return 'That temperature does not make sense!'


for temp in temperatures:
    print(c_to_f(temp))
