import yfinance as yf
import datetime
import requests 
from bs4 import BeautifulSoup
import re
import pandas as pd

def get_crude_price():
    today = datetime.datetime.now()
    first_day = datetime.datetime(today.year, today.month, 1)
    crude = yf.Ticker("CL=F")
    data = crude.history(start=first_day, end=today)
    if data.empty or "Close" not in data.columns:
        raise ValueError("Failed to retrieve crude oil data.")
    return data["Close"].iloc[0]


def get_latest_gold_price():
    today = datetime.datetime.now()
    month_name = date.strftime("%B")
    year = str(date.year)
    first_day_text = datetime.datetime(today.year, today.month, 1).strftime("%d")
    url = f"https://www.uaegoldprice.com/gold-price-history/gold-price-in-{month_name}-{year}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise ValueError(f"Failed to fetch gold prices, status code {resp.status_code}")
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        raise ValueError("No gold price table found on the page")
    for row in table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) >= 3:
            date_text = cols[0].text.strip()
            price_24k_text = cols[2].text.strip()
            if date_text.startswith(first_day_text):
                price_clean = re.sub(r"[^\d.]", "", price_24k_text)
                return price_clean

date = datetime.datetime.now()
targetdate = date.strftime("%B")+ " "+ str(date.year)

def getuaeprice():
    url = "https://gulfnews.com/gold-forex/historical-fuel-rates"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/117.0"
    }

    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    if not table:
        raise Exception("No table found on page")

    for row in table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) >= 3:
            month = cols[0].text.strip()
            if month == targetdate:
                return cols[2].text.strip()

    return None


coprice = get_crude_price()*3.67
gold_price = get_latest_gold_price()
uaefuel = getuaeprice()

print(int(coprice)*3.67,gold_price, uaefuel)

#FILE PATH HERE
excelpath = ""

df = pd.read_excel(excelpath)

first_col_name = df.columns[0]
df.rename(columns={first_col_name: 'Date'}, inplace=True)

df.rename(columns={
    'Fuel price(AED)': 'Fuel price(AED)',
    'Crude Oil Barrel Price (USD)': 'Crude Oil Barrel Price (USD)',
    'Gold Prices (AED)': 'Gold Prices (AED)'
}, inplace=True)


df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
df.dropna(subset=['Date'], inplace=True)

today = datetime.datetime.now()
first_day_str = datetime.datetime(today.year, today.month, 1).strftime("%Y-%m-%d")


if not uaefuel or not coprice or not gold_price:
    print("No value found for one or more inputs. Write cancelled. ")

else:

    if not (df['Date'] == first_day_str).any():
        

        new_index = len(df)
        

        df.loc[new_index, 'Date'] = first_day_str
        df.loc[new_index, 'Fuel price(AED)'] = uaefuel
        df.loc[new_index, 'Crude Oil Barrel Price (USD)'] = coprice
        df.loc[new_index, 'Gold Prices (AED)'] = gold_price
        
        print(f"Added new row for {first_day_str} ")
        
        df.to_excel(excelpath, index=False)
        
    else:
        print(f"Row for {first_day_str} already exists → nothing changed")