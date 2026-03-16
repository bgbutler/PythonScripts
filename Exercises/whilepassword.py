password = ''
while password != 'python123':
    password = raw_input('Enter your password: ')
    if password == 'python123':
        print('You are logged in now $ ')
    else:
        print('Sorry, try again!')


