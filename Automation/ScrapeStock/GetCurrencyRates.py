# scrape current from x-rates.com
# https://www.x-rates.com/calculator/?from=USD&to=EUR&amount=1

from bs4 import BeautifulSoup
import requests
import inspect

def get_currency(in_currency, out_currency):
    url = f"""https://www.x-rates.com/calculator/?from={in_currency}&to={out_currency}&amount=1"""
    # get some headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}

    content = requests.get(url, headers=headers).text
    soup = BeautifulSoup(content, 'html.parser')
    rate = soup.find("span", class_="ccOutputRslt").get_text()
    rate = float(rate[0:-4])
    return rate, in_currency, out_currency




current_rate, in_curr, out_curr = get_currency('EUR', 'USD')
print(f"The current rate is 1 {in_curr} costs {current_rate} {out_curr}")