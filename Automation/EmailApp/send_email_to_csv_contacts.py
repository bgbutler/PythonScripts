import yagmail
import pandas as pd

contacts = pd.read_csv('contacts.csv', header=0)
sender = 'bryan.g.butler@gmail.com'


subject = "Test Email"

print(contacts)

# log in to email - just once
yag = yagmail.SMTP(user=sender, password='Ferrocenophane1+1')
contacts.reset_index()
for index, row in contacts.iterrows():
    print(row['name'], row['email'])
    # make a dynamic message
    contents = f"""
    Hi {row['name']} - this is a test of automation.
    """
    # need a way to obscure password or get it as an input
    # yag.send(to=row['email'], subject=subject, contents=contents)
    # print(contents)
    print('Email Sent')
