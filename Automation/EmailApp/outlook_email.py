# this sends from a hotmail or outlook account

import smtplib

sender = 'diurac_guy@hotmail.com'
receiver = 'bryan.g.butler@gmail.com'

password = 'nect4fun'


message = """\
Subject: Hello Hello

This is DG. This is the message of the email.
"""

# need the domain and port of the SMTP server
server = smtplib.SMTP('smtp.office365.com', 587)
server.starttls()
server.login(sender, password)
server.sendmail(sender, receiver, message)
server.quit()