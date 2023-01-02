import yagmail

sender = 'bryan.g.butler@gmail.com'
receiver = 'bryan.g.butler@gmail.com'

subject = "This is the subject!"

# make the contacts a list
contents = ["Here is the content of the email"]

print(contents)

yag = yagmail.SMTP(user=sender, password='Ferrocenophane1+1')
yag.send(to=receiver, subject=subject, contents=contents, attachments='Files/text.txt')

print('Email Sent')