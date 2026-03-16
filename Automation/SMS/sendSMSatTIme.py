# this requires a twilio SMS account
# might need to do additional installs Node JS, Twilio

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
# from twilio.rest import Client
import os
import time
import yagmail
from datetime import datetime as dt


def get_driver():
    # set options to make browsing easier
    options = webdriver.ChromeOptions()
    options.add_argument("disable-infobars")
    options.add_argument("star-maximized")
    options.add_argument("disable-dev-shm-usage")
    options.add_argument("no-sandbox")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument('disable-blink-features=AutomationControlled')
    driver = webdriver.Chrome(executable_path="/Users/Bryan/Documents/OnLineClasses/Python_Courses/AutomatePython/chromedriver", options=options)

    driver.get("https://smile.amazon.com/dp/B07K5DSHNW/?coliid=I2KL2JLDSGUTU1&colid=26CMMDKH4Q8G1&psc=1&ref_=lv_ov_lig_dp_it")
    return driver

def main():
    driver = get_driver()
    element = driver.find_element(by="xpath", value='/html/body/div[1]/div[2]/div[8]/div[6]/div[4]/div[9]/div[1]/div[1]/span/span[2]/span[2]')
    return element.text

orig_price = main()
print(f"The current price is $", str(orig_price).strip())

# now send the email or SMS
# append the prices to a list for comparison
prices = [orig_price]

# while True:
#     time.sleep(5)
#     price = main()
#     prices.append(price)
#     print(prices)
#
#     if prices[-1] < prices[-2]:
#         pass
#     del prices[-2]

sender = 'bryan.g.butler@gmail.com'
receiver = 'bgbutler@me.com'

subject = f"The price has just dropped to ${orig_price}. Hurry up"

contents = f"Hey, I just noticed the price of your item is now ${orig_price}."

# find a way to run without the password exposed
yag = yagmail.SMTP(user=sender, password='Ferrocenophane1+1')
yag.send(to=receiver, subject=subject, contents=contents)

print('Email Sent')




