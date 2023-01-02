# this sends from a hotmail or outlook account

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

sender = 'diurac_guy@hotmail.com'
receiver = 'bryan.g.butler@gmail.com'
password = 'nect4fun'

message = MIMEMultipart()
message['From'] = sender
message['To'] = receiver
message['Subject'] = 'Hello again'

body = """"
<h2>This is DG.</h2> This is the message of the email!
"""
mime_text = MIMEText(body, 'html')
message.attach(mime_text)

attachment_path = '/Users/Bryan/Documents/OnLineClasses/Python_Courses/AutomatePython/EmailApp/Files/house.jpg'
attachment_file = open(attachment_path, 'rb')
payload = MIMEBase('application', 'octate-stream')
payload.set_payload(attachment_file.read())
encoders.encode_base64(payload)
payload.add_header('Content-Disposition', 'attachment', filename='house.jpg')
message.attach(payload)

# need the domain and port of the SMTP server
server = smtplib.SMTP('smtp.office365.com', 587)
server.starttls()
server.login(sender, password)
message_text = message.as_string()
print(message_text)
server.sendmail(sender, receiver, message_text)
server.quit()