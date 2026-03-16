import yagmail

sender = 'bryan.g.butler@gmail.com'
receiver = 'bgbutler@me.com'

subject = "This is the subject!"

contents = """
Here is the content of the email
"""

yag = yagmail.SMTP(user=sender, password='Ferrocenophane1+1')
yag.send(to=receiver, subject=subject, contents=contents)

print('Email Sent')