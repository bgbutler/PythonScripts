import yagmail
import time

sender = 'bryan.g.butler@gmail.com'
receiver = 'bgbutler@me.com'

subject = "This is the subject!"

contents = """
Here is the content of the email
"""

x = 0
while (x<3):
    yag = yagmail.SMTP(user=sender, password='Ferrocenophane1+1')
    yag.send(to=receiver, subject=subject, contents=contents)
    print('Email Sent')
    x=+1
    time.sleep(10)