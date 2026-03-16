# this requires a twilio SMS account
# might need to do additional installs Node JS, Twilio

import os
import time
from twilio.rest import Client

account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(acoount_sid, auth_token)

i = 0
limit = True
while limit:
    message = client.messages \
        .create(
        body="Join Earth's mightiest heroes. Like Kevin Bacon.",
        from_='18609337573',
        to='18605087722')

    print(message.sid)
    time.sleep(60)
    i +=
    if i == 10:
        limit = False