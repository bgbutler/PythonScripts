from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

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

    driver.get("http://automated.pythonanywhere.com/login")
    return driver

def clean_text(text):
    """ Extract only temperature from text"""
    output = float(text.split(":")[1])
    return output

def main():
    driver = get_driver()
    # send_keys() is waiting for input, we are sending the PW
    driver.find_element(by="id", value="id_username").send_keys("automated")
    time.sleep(2)

    # press the return key after password
    driver.find_element(by="id", value="id_password").send_keys("automated" + Keys.RETURN)
    time.sleep(2)

    # clicks the home button on the dashboard after logging in
    driver.find_element(by="xpath", value="/html/body/nav/div/a")
    .click()
    time.sleep(2)

    # now scrape the temp value on the page
    text = driver.find_element(by="xpath", value="/html/body/div[1]/div/h1[2]")
    return clean_text(text)

print(main())