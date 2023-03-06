
import timeit
import logging
import pandas as pd
from binance.client import Client
import os
import sys

# calculate program run time
start = timeit.default_timer()

# log file to store error messages
log_filename = "main.log"
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p -')

# environment variables
try:
    # Binance
    api_key = os.environ.get('binance_api')
    api_secret = os.environ.get('binance_secret')

except KeyError as e: 
    msg = sys._getframe(  ).f_code.co_name+" - "+repr(e)
    print(msg)
    logging.exception(msg)
    # telegram.send_telegram_message(telegram.telegramToken_errors, telegram.eWarning, msg)
    sys.exit(msg) 

# Binance Client
try:
    client = Client(api_key, api_secret)

except Exception as e:
        msg = "Error connecting to Binance. "+ repr(e)
        print(msg)
        logging.exception(msg)
        # telegram.send_telegram_message(telegramToken, telegram.eWarning, msg)
        sys.exit(msg) 

def get_data(coinPair, time_frame):
    try:
        
        # if best Ema exist get price data 
        # lstartDate = str(1+gSlowMA*aTimeframeNum)+" "+lTimeframeTypeLong+" ago UTC"
        # sma200 = 200
        # lstartDate = str(sma200*aTimeframeNum)+" "+lTimeframeTypeLong+" ago UTC" 
        # ltimeframe = str(aTimeframeNum)+aTimeframeTypeShort

        frame = pd.DataFrame(client.get_historical_klines(coinPair
                                                        ,ltimetime_frameframe
    
                                                        # better get all historical data. 
                                                        # Using a defined start date will affect ema values. 
                                                        # To get same ema and sma values of tradingview all historical data must be used. 
                                                        # ,lstartDate
                                                        
                                                        ))

        frame = frame[[0,4]]
        frame.columns = ['Time','Close']
        frame.Close = frame.Close.astype(float)
        frame.Time = pd.to_datetime(frame.Time, unit='ms')
        return frame
    except Exception as e:
        msg = sys._getframe(  ).f_code.co_name+" - "+coinPair+" - "+repr(e)
        print(msg)
        # telegram.send_telegram_message(telegramToken, telegram.eWarning, msg)
        frame = pd.DataFrame()
        return frame