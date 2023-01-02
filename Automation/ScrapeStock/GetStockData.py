# url to scrape https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=345427200&period2=1646784000&interval=1d&events=history&includeAdjustedClose=true

import requests
from datetime import datetime

# have the user enter start date
ticker = input('Enter the ticker of the stock you want data: ')
start_date = input('Enter start date in yyyy/mm/dd format: ')
end_date = input('Enter end date in yyyy/mm/dd format: ')
import time

# convert date time format to the epoch format
from_datetime = datetime.strptime(start_date, '%Y/%m/%d')
to_datetime = datetime.strptime(end_date, '%Y/%m/%d')

from_epoch = str(int(time.mktime(from_datetime.timetuple())))
to_epoch = str(int(time.mktime(to_datetime.timetuple())))

# make the url
url = f"""https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1=
{from_epoch}&period2={to_epoch}&interval=1d&events=history&includeAdjustedClose=true"""

# get some headers
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}


content = requests.get(url, headers = headers).content
print(content)

# make a file
filename = ticker + ".csv"
with open(filename, 'wb') as file:
    file.write(content)
