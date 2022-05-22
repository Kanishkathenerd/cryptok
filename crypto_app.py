import datetime
import os


import streamlit as st
import base64
import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from PIL import Image
from bs4 import BeautifulSoup
import requests
from colorama import Fore, Style
from csv import DictReader, writer
import yfinance as yf

st.set_page_config(layout="wide")
image = Image.open('Light Red Stationary Etsy Order Receipt Banners.png')
st.image(image, use_column_width=True, output_format='png')
st.title("CryptoCurrency Web App")
st.markdown("""
This app retrieves cryptocurrency data from [CoinMarketCap](https://coinmarketcap.com), [Coinbase](https://www.coinbase.com) and [YahooFinance](http://finance.yahoo.com), regarding the top 100 Cryptos **right now**.
* **Note:** Any abrupt price change in the charts is due to a lack of data regarding the certain cryptocurrency caused by the API.
""")
# ! ---------------------------------
# ! About section
expander_bar = st.expander('About')
expander_bar.markdown("""
* **Python Libraries:** pandas, numpy, scikit-learn, base64, matplotlib, requests, bs4, json, time, datetime, seaborn, csv, PIL, colorama, yfinance
* **Data Source:** [CoinMarketCap](https://coinmarketcap.com), [Coinbase](https://www.coinbase.com), [YahooFinance](http://finance.yahoo.com)
""")

# ! -------------------------------------------
# ! Page layout
# ? Divide the page to three columns (col1 = sidebar) (col2 & col3 = page

col1 = st.sidebar
col2, col3 = st.columns((2,1)) # ! this means that the first col is 2 times greater than the second col

# ! -------------------------------------------


# ? WEB SCRAPE OUR DATA FROM CoinMarketCap
# @st.cache # ! this means data will be loaded only one time
def load_data(value, currency_price_unit):

    url = requests.get("https://coinmarketcap.com")
    soup = BeautifulSoup(url.content, "html.parser")

    data = soup.find('script', id='__NEXT_DATA__', type='application/json')

    ###### THE CORRECT SCRIPT ##########
    coins = {}
    coin_data = json.loads(data.contents[0])
    listings = coin_data['props']['initialState']['cryptocurrency']['spotlight']['data']
    listings_basic = coin_data['props']['initialState']['cryptocurrency']['listingLatest']['data']
    if int(value) == 5:

        coin_name = []
        coin_symbol = []
        market_cap = []
        pct_1h = []
        pct_24h = []
        pct_7d = []
        pct_30d = []
        pct_60d = []
        pct_90d = []
        vol_24h = []
        vol_7d = []
        price = []

        #currency_price_unit = "USD"
        for i in listings_basic:


            try:
                coin_name.append(i[13])
                #print(f'{Fore.YELLOW}{i[14]}{Style.RESET_ALL}')
            except Exception as e:
                pass
            try:
                coin_symbol.append(i[131])
                #print(f'{Fore.YELLOW}{i[130]}{Style.RESET_ALL}')
            except Exception as e:
                pass

            if currency_price_unit == 'USD':

                try:

                    price.append(i[64])
                    #print(f'{Fore.YELLOW}{i[65]}{Style.RESET_ALL}')

                except Exception as e:
                    pass
                try:
                    #for idx, item in enumerate(listings_basic[0]['keysArr']):
                    #    print(idx, ' - ', item)
                    #print(listings_basic[0]['keysArr'])
                    pct_1h.append(i[58])
                except Exception as e:
                    pass
                try:
                    pct_24h.append(i[59])
                except Exception as e:
                    pass
                try:
                    pct_7d.append(i[62])
                except Exception as e:
                    pass
                try:
                    pct_30d.append(i[60])
                except Exception as e:
                    pass
                try:
                    pct_60d.append(i[61])
                except Exception as e:
                    pass
                try:
                    pct_90d.append(i[63])
                except Exception as e:
                    pass
                try:
                    market_cap.append(i[55])
                except Exception as e:
                    pass
                try:
                    vol_24h.append(i[67])
                except Exception as e:
                    pass
                try:
                    vol_7d.append(i[69])
                except Exception as e:
                    pass
            elif currency_price_unit == "BTC":
                try:
                    price.append(i[26])
                except Exception as e:
                    pass
                try:
                    pct_1h.append(i[20])
                except Exception as e:
                    pass
                try:
                    pct_24h.append(i[21])
                except Exception as e:
                    pass
                try:
                    pct_7d.append(i[24])
                except Exception as e:
                    pass
                try:
                    pct_30d.append(i[22])
                except Exception as e:
                    pass
                try:
                    pct_60d.append(i[23])
                except Exception as e:
                    pass
                try:
                    pct_90d.append(i[25])
                except Exception as e:
                    pass
                try:
                    market_cap.append(i[17])
                except Exception as e:
                    pass
                try:
                    vol_24h.append(i[29])
                except Exception as e:
                    pass
                try:
                    vol_7d.append(i[31])
                except Exception as e:
                    pass
            elif currency_price_unit == "ETH":
                try:
                    price.append(i[45])
                except Exception as e:
                    pass
                try:
                    pct_1h.append(i[49])
                except Exception as e:
                    pass
                try:
                    pct_24h.append(i[40])
                except Exception as e:
                    pass
                try:
                    pct_7d.append(i[43])
                except Exception as e:
                    pass
                try:
                    pct_30d.append(i[41])
                except Exception as e:
                    pass
                try:
                    pct_60d.append(i[42])
                except Exception as e:
                    pass
                try:
                    pct_90d.append(i[44])
                except Exception as e:
                    pass
                try:
                    market_cap.append(i[36])
                except Exception as e:
                    pass
                try:
                    vol_24h.append(i[48])
                except Exception as e:
                    pass
                try:
                    vol_7d.append(i[50])
                except Exception as e:
                    pass
    else:

        table_keys = []
        for key in listings.keys():
            table_keys.append(key)
            print(f'{Fore.GREEN}{key}{Style.RESET_ALL}')
        # print(table_keys)

        coin_name = []
        coin_symbol = []
        market_cap = []
        pct_1h = []
        pct_24h = []
        pct_7d = []
        pct_30d = []
        pct_60d = []
        pct_90d = []
        vol_24h = []
        vol_7d = []
        price = []

        for i in listings[table_keys[int(value)]]:
            coin_name.append(i['name'])
            coin_symbol.append(i['symbol'])
            price.append(i['priceChange']['price'])
            try:
                pct_1h.append(i['priceChange']['priceChange1h'])
            except Exception as e:
                pct_1h.append(None)
            try:
                pct_24h.append(i['priceChange']['priceChange24h'])
            except Exception as e:
                pct_24h.append(None)
            try:
                pct_7d.append(i['priceChange']['priceChange7d'])
            except Exception as e:
                pct_7d.append(None)
            try:
                pct_30d.append(i['priceChange']['priceChange30d'])
            except Exception as e:
                pct_30d.append(None)
            try:
                market_cap.append(i['marketCap'])
            except Exception as e:
                market_cap.append(None)
            vol_24h.append(i['priceChange']['volume24h'])

            vol_7d.append(None)
            pct_60d.append(None)
            pct_90d.append(None)

    df = pd.DataFrame(columns=["Name", "Symbol", 'Price', 'pct_change_1h', 'pct_change_24h','pct_change_7d','pct_change_30d', "pct_change_60d", "pct_change_90d", "Market_cap", 'Volume_24h', "Volume_7d"])
    df['Name'] = coin_name
    df['Symbol'] = coin_symbol
    df['Price'] = price
    df['pct_change_1h'] = pct_1h
    df['pct_change_24h'] = pct_24h
    df['pct_change_7d'] = pct_7d
    df['pct_change_30d'] = pct_30d
    df['pct_change_60d'] = pct_60d
    df['pct_change_90d'] = pct_90d
    df['Market_cap'] = market_cap
    df['Volume_24h'] = vol_24h
    df["Volume_7d"] = vol_7d

    return df

        #print(coin_data)

dict_of_tables = {"Top Cryptocurrencies": "5", #put this first to be shown first
                  }
# "Trending Crypto": "0", "Most Visited Crypto": "1", "Recently Added": "2", "Top Gainers": "3", "Top Losers": "4"

list_of_units = ["USD", "BTC", "ETH"]

# ! 1. Sidebar = Main Panel
col1.header('Input Options')
# ! 2. Sidebar - Pick the list
type_of_frame = col1.selectbox("Type of list", options=dict_of_tables.keys())


# ! 2. Sidebar - Pick the unit
if type_of_frame != "Top Cryptocurrencies":
    df = load_data(dict_of_tables[str(type_of_frame)], currency_price_unit = 'USD')
else:
    unit = col1.radio('Select currency unit', options=list_of_units)
    df = load_data(dict_of_tables[str(type_of_frame)], unit)

# ! 3. Sidebar - Cryptocurrency selections
sorted_coin = sorted(df['Symbol'])
selected_coin = col1.multiselect("Cryptocurrency", sorted_coin, sorted_coin)

df_selected_coin = df[(df['Symbol'].isin(selected_coin))]



# ! -------------------------------------------

# ? Display on col2

# ! Slider - number of coins to diplay
num_coin = col2.slider("Display N-coins", 1, max_value=len(df_selected_coin), value=len(df_selected_coin))
df_coins = df_selected_coin[:num_coin]


# ! Title of dataframe
col2.subheader('Price Data of Selected Cryptocurrency')
# ! Dataframe display
col2.dataframe(df_coins.style.highlight_max(axis=0, props='color:white;background-color:green').highlight_min(axis=0, props='color:white;background-color:red'), width=900, height=400 )
col2.write(f"Data Dimensions: {str(df_coins.shape[0])}x{str(df_coins.shape[1])}")

# ! -------------------------------------------

# ? Download csv data

def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="crypto.csv">Download CSV File</a>'
    return href

col2.markdown(file_download(df_coins), unsafe_allow_html=True)

# ! -------------------------------------------
# ? Preparing data for Bar plot of % Price change

#col2.subheader('Table of % Price Change')
df_change = pd.concat([df_coins.Symbol, df_coins.pct_change_1h, df_coins.pct_change_24h, df_coins.pct_change_7d, df_coins.pct_change_30d, df_coins.pct_change_60d, df_coins.pct_change_90d ], axis=1)
df_change = df_change.set_index('Symbol')
df_change['positive_percent_change_1h'] = df_change['pct_change_1h'] > 0
df_change['positive_percent_change_24h'] = df_change['pct_change_24h'] > 0
df_change['positive_percent_change_7d'] = df_change['pct_change_7d'] > 0
df_change['positive_percent_change_30d'] = df_change['pct_change_30d'] > 0
df_change['positive_percent_change_60d'] = df_change['pct_change_60d'] > 0
df_change['positive_percent_change_90d'] = df_change['pct_change_90d'] > 0
#col2.dataframe(df_change)

# ! Conditional creation of Bar plot (time frame)
# ! Percent change timeframe
percent_timeframe = col3.selectbox('Percent change time frame',
                                    ['7d','24h', '1h', '30d', '60d', '90d'])
percent_dict = {"7d":'pct_change_7d',"24h":'pct_change_24h',"1h":'pct_change_1h','30d':"pct_change_30d", "60d":"pct_change_60d", "90d":"pct_change_90d" }

selected_percent_timeframe = percent_dict[percent_timeframe]

col3.subheader('% Price Change')

sort_values = col3.selectbox('Sort values?', ('Yes', 'No'))
df_change = df_change.sort_values(by=['Symbol'], ascending=False)
if percent_timeframe == '7d':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['pct_change_7d'])
    col3.write('*7 days period*')
    plt.figure(figsize=(5,25))
    plt.subplots_adjust(top = 1, bottom = 0)
    df_change['pct_change_7d'].plot(kind='barh',animated = True, color=df_change.positive_percent_change_7d.map({True: 'g', False: 'r'}))
    negative = 0
    counter = 0
    for pct in df_change.pct_change_7d:
        counter += 1
        if float(pct) < 0:
            negative += 1
        else:
            pass
    col3.markdown(f'''
**{negative} out of {counter}** coins had ***negative*** returns the last 7 days.''')
    col3.pyplot(plt)
elif percent_timeframe == '24h':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['pct_change_24h'])
    col3.write('*24 hour period*')
    plt.figure(figsize=(5,25))
    plt.subplots_adjust(top = 1, bottom = 0)
    df_change['pct_change_24h'].plot(kind='barh', color=df_change.positive_percent_change_24h.map({True: 'g', False: 'r'}))
    negative = 0
    counter = 0
    for pct in df_change.pct_change_24h:
        counter += 1
        if float(pct) < 0:
            negative += 1
        else:
            pass
    col3.markdown(f'''
    **{negative} out of {counter}** coins had ***negative*** returns the last 24 hours.''')
    col3.pyplot(plt)
elif percent_timeframe == '1h':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['pct_change_1h'])
    col3.write('*1 hour period*')
    plt.figure(figsize=(5,25))
    plt.subplots_adjust(top = 1, bottom = 0)
    df_change['pct_change_1h'].plot(kind='barh', color=df_change.positive_percent_change_1h.map({True: 'g', False: 'r'}))
    negative = 0
    counter = 0
    for pct in df_change.pct_change_1h:
        counter += 1
        if float(pct) < 0:
            negative += 1
        else:
            pass
    col3.markdown(f'''
    **{negative} out of {counter}** coins had ***negative*** returns the last hour.''')
    col3.pyplot(plt)
elif percent_timeframe == '30d':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['pct_change_30d'])
    col3.write('*30 days period*')
    plt.figure(figsize=(5, 25))
    plt.subplots_adjust(top=1, bottom=0)
    df_change['pct_change_30d'].plot(kind='barh', color=df_change.positive_percent_change_30d.map({True: 'g', False: 'r'}))
    negative = 0
    counter = 0
    for pct in df_change.pct_change_30d:
        counter += 1
        if float(pct) < 0:
            negative += 1
        else:
            pass
    col3.markdown(f'''
    **{negative} out of {counter}** coins had ***negative*** returns the last 30 days.''')
    col3.pyplot(plt)
elif percent_timeframe == '60d':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['pct_change_60d'])
    col3.write('*60 days period*')
    plt.figure(figsize=(5, 25))
    plt.subplots_adjust(top=1, bottom=0)
    df_change['pct_change_60d'].plot(kind='barh',                               color=df_change.positive_percent_change_60d.map({True: 'g', False: 'r'}))
    negative = 0
    counter = 0
    for pct in df_change.pct_change_60d:
        counter += 1
        if float(pct) < 0:
            negative += 1
        else:
            pass
    col3.markdown(f'''
    **{negative} out of {counter}** coins had ***negative*** returns the last 60 days.''')
    col3.pyplot(plt)
elif percent_timeframe == '90d':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['pct_change_90d'])
    col3.write('*90 days period*')
    plt.figure(figsize=(5, 25))
    plt.subplots_adjust(top=1, bottom=0)
    df_change['pct_change_90d'].plot(kind='barh', color=df_change.positive_percent_change_90d.map({True: 'g', False: 'r'}))
    negative = 0
    counter = 0
    for pct in df_change.pct_change_90d:
        counter += 1
        if float(pct) < 0:
            negative += 1
        else:
            pass
    col3.markdown(f'''
    **{negative} out of {counter}** coins had ***negative*** returns the last 90 days.''')
    col3.pyplot(plt)

# ! Building a correlation heatmap
@st.cache
def fetch_historicrypto_yf(df):
    failed_coins = []
    for symbol in df.Symbol:
        if os.path.exists(f'yf_{symbol}-USD.csv'):
            pass
        else:
        # ! we will fetch 60 days - data with information every 15 minutes for all coins
            try:
                data = yf.download(tickers=f'{symbol}-USD', period='60d', interval='15m')
                data.drop(['Open', "High", "Adj Close", 'Low'], inplace=True, axis=1)
                data.to_csv(f'yf_{symbol}-USD.csv', index_label='Date')
            except Exception as e:
                failed_coins.append(symbol)
                pass
    print(failed_coins)
    print('Done')

@st.cache
def renew_data(df):
    failed_coins = []
    for symbol in df.Symbol:
        print(symbol)
        try:
            data = yf.download(tickers=f'{symbol}-USD', period='1min')
            data.drop(['Open', "High", "Adj Close", 'Low'], inplace=True, axis=1)
            data.to_csv(f'yf_{symbol}-USD.csv', mode='a', header=False)
            #break
        except Exception as e:
            failed_coins.append(symbol)
            pass
    print(failed_coins)
    print('Done')

#@st.cache
def hollistic_df(df_coins):
    df_prices = pd.DataFrame(columns=[symbol for symbol in df_coins.Symbol])
    #len_df = 5680
    symbols_proccessed = 0
    with open(f'yf_BTC-USD.csv', 'r') as btc:
        new_df = pd.DataFrame()
        # print(f.name)
        reader = DictReader(btc)
        count_rows = 0
        for row in btc:
            count_rows += 1
        len_df = count_rows-1

        btc.close()

    for symbol in df_prices.columns:
        #st.progress(symbols_proccessed)
        try:
            with open(f'yf_{symbol}-USD.csv', 'r') as f:
                new_df = pd.DataFrame()
                #print(f.name)
                reader = DictReader(f)
                #print(reader)
                close_prices = []

                for row in reader:
                    close_prices.append(float(row["Close"]))
                if len(close_prices) < len_df:
                    how_many_nans = len_df - len(close_prices)
                    for nan in range(0, how_many_nans):
                        close_prices.insert(0, 0)

                new_df[symbol] = close_prices
                #col2.dataframe(new_df)
                f.close()
            df_prices[symbol] = pd.Series(new_df[symbol])
        except Exception as e:
            #st.exception(e)
            df_prices[symbol] = 1
            pass
        symbols_proccessed +=1
        print('Cryptos proccessed: ', symbols_proccessed)
    return df_prices

def new_hollistic_df(df_coins):
    df_prices = pd.DataFrame(columns=[symbol for symbol in df_coins.Symbol])
    symbols_proccessed = 0
    for symbol in df_prices.columns:
        try:
            file = pd.read_csv(f'yf_{symbol}-USD.csv')
            raw_df = pd.DataFrame(file.iloc[-1].values)
            new_df = pd.DataFrame(columns=[symbol])

            new_df[symbol] = raw_df.iloc[1].values
            df_prices[symbol] = pd.Series(new_df[symbol])
        except Exception as e:
            st.exception(f'{Exception}: {e}')
            df_prices[symbol] = 1
            pass
        symbols_proccessed +=1
        print('Cryptos proccessed: ', symbols_proccessed)

    return df_prices

#fetch_historicrypto_yf(df_coins)
df_prices = hollistic_df(df_coins)
#mutable_table = col2.dataframe(df_prices.tail(10))
#col2.write(df_prices.shape)

#df_prices = df_prices.dropna(axis=1)
correlation_df = df_prices[3000:].corr(method='pearson')

fig, ax = plt.subplots(figsize=(14,20)) #figsize=(12,7)
heatmap_selection = col2.selectbox('Correlation Heatmap Gram', ("No", "Yes"))
if heatmap_selection == 'Yes':
    sb.heatmap(data=correlation_df, xticklabels=True, yticklabels=True, annot=True)
    col2.subheader('Correlation heatmap of selected Crypto')
    col2.write(fig)
else:
    pass

plot_choice = col2.selectbox('Coin Plots', ['No', 'Yes'])
if plot_choice == 'Yes':

    coin_choice = col2.multiselect('Coin Plot', options=[symbol for symbol in df_coins.Symbol], default = df_coins['Symbol'][1])


    df_prices.replace(0, float('nan'), inplace=True)
    col2.markdown("""
* **Start Date:** 16/11/2021
* **Price change frequency:** 15 minutes
""")
    col2.subheader('Cryptocurrency Graph')
    plt.style.use('fivethirtyeight')
    my_crypto = df_prices[coin_choice]
    plt.figure(figsize=(12.2, 4.5))
    for c in my_crypto.columns.values:
        plt.plot(my_crypto[c], label=c)
    plt.xlabel('price change every 15-minutes')
    plt.ylabel('Crypto Price ($)')
    plt.legend(my_crypto.columns.values)
    col2.pyplot(plt)

    # Scale the data
    # the min-max scaler method scales the dataset so that all the input features lie between 0 and 100 inclusive
    from sklearn import preprocessing

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    scaled = min_max_scaler.fit_transform(df_prices[coin_choice])

    df_scale = pd.DataFrame(scaled, columns=df_prices[coin_choice].columns)
    # Visualize the scaled data
    my_crypto = df_scale

    col2.subheader('Crypto Scaled Graph')
    plt.figure(figsize=(12.4, 4.5))
    for c in my_crypto.columns.values:
        plt.plot(my_crypto[c], label=c)
    plt.xlabel('price change every 15-minutes')
    plt.ylabel('Crypto Scaled Price ($)')
    plt.legend(my_crypto.columns.values)
    col2.pyplot(plt)

    DSR = df_prices[coin_choice].pct_change(1)

    col2.subheader('15-minutes Simple Returns')
    plt.figure(figsize=(12, 4.5))
    for c in DSR.columns.values:
        plt.plot(DSR.index, DSR[c], label=c, lw=2, alpha=.7)
    plt.ylabel('Percentage (in decimal form')
    plt.xlabel('price change every 15-minutes')
    plt.legend(DSR.columns.values, loc='upper right')
    col2.pyplot(plt)

    # daily cumulative simple returns.
    col2.subheader('15-minutes Cumulative Simple Return')
    DCSR = (DSR+1).cumprod()
    plt.figure(figsize=(12.2, 4.5))
    for c in DCSR.columns.values:
        plt.plot(DCSR.index, DCSR[c], lw=2, label=c)
    plt.xlabel('price change every 15-minutes')
    plt.ylabel('Growth of $1 investment')
    plt.legend(DCSR.columns.values, loc='upper left', fontsize=10)
    col2.pyplot(plt)



