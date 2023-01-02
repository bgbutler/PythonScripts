import yagmail
import time
from datetime import datetime as dt

sender = 'bryan.g.butler@gmail.com'
receiver = 'bgbutler@me.com'

subject = "This is the subject!"

contents = """
Here is the content of the email
"""

while True:
    now = dt.now()
    if now.hour == 13 and now.minute == 15:
        yag = yagmail.SMTP(user=sender, password='Ferrocenophane1+1')
        yag.send(to=receiver, subject=subject, contents=contents)
        print('Email Sent')
        time.sleep(60)