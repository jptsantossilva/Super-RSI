
"""
SUPER-RSI

The strategy roughly goes like this:

send alerts when:
    .RSI 1d / 4h / 1h / 30m / 15m <= 20
    .RSI 1d / 4h / 1h / 30m / 15m >= 80

---

Enter argument to choose run mode type (backtest or prod). If no argument is entered, prod mode will be chosen.
python3 super_rsi.py backtest

"""

import os
from binance.client import Client
import pandas as pd
import datetime
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply
import sys
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import time
import sqlite3
import ta
import telegram

msg = 'SUPER-RSI - Start'
print(msg)
telegram.send_telegram_message(telegram.eStart, msg)

print('Enter argument to choose run mode type (backtest or prod).\nIf no argument is entered, prod mode will be chosen.')
# total arguments
n = len(sys.argv)
# print("Total arguments passed:", n)
if n < 2:
    run_mode = "prod"
    run_mode = "backtest" # tests purpose
    print(f"{run_mode} mode") 
else:
    # argv[0] in Python is always the name of the script.
    # print("Argument is missing")
    # run_mode = input('Enter run mode (backtest or execution):')
    arg_run_mode = sys.argv[1]
    if arg_run_mode in['backtest','prod']:
        run_mode = arg_run_mode
    else:
        msg = "invalid argument"
        sys.exit(msg) 

# environment variables
try:
    # Binance
    api_key = os.environ.get('binance_api')
    api_secret = os.environ.get('binance_secret')

except KeyError as e: 
    msg = sys._getframe(  ).f_code.co_name+" - "+repr(e)
    print(msg)
    # logging.exception(msg)
    telegram.send_telegram_message(telegram.eWarning, msg)
    sys.exit(msg) 

# Binance Client
try:
    client = Client(api_key, api_secret)
except Exception as e:
        msg = "Error connecting to Binance. "+ repr(e)
        print(msg)
        # logging.exception(msg)
        telegram.send_telegram_message(telegram.eWarning, msg)
        sys.exit(msg) 

# database
    
# get the path to the current directory
dir_path = os.path.dirname(os.path.realpath(__file__))
# join the directory path with the name of the database file
db_path = os.path.join(dir_path, "super-rsi.db")
# create a connection to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# best _rsi table
cursor.execute('''CREATE TABLE IF NOT EXISTS best_rsi
                (symbol TEXT, \rsi_1d REAL, rsi_4h REAL, rsi_1h REAL, rsi_30m REAL, 
                 rsi_15m REAL, rsi_low REAL, rsi_high REAL, return_perc REAL, 
                 buyhold_return_perc REAL, backtest_start_date TEXT, backtest_end_date TEXT, 
                 num_trades INTEGER, PRIMARY KEY(symbol))''')
# symbols table
cursor.execute('''CREATE TABLE IF NOT EXISTS symbols (
               symbol TEXT,
               calc BOOLEAN,
               PRIMARY KEY(symbol))''')
# backtest trades table
cursor.execute('''CREATE TABLE IF NOT EXISTS backtest_trades (
                symbol TEXT, EntryPrice REAL, ExitPrice REAL, PnL REAL,
                ReturnPct REAL, EntryTime DATETIME, ExitTime DATETIME, Duration TEX)''')

# commit the changes to the database
conn.commit()

rsi_lookback_periods = 14 # 14 days


def EMA(values, n):
    """
    Return exp moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).ewm(span=n, adjust=False).mean()

def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()


def RSI_backtesting(values, n):
    """Relative strength index"""
    
    # # # Approximate; good enough
    # gain = pd.Series(values).diff()
    # loss = gain.copy()
    # gain[gain < 0] = 0
    # loss[loss > 0] = 0
    # rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean()
    # # return 100 - 100 / (1 + rs)
    # rs = 100 - 100 / (1 + rs)
    # # return rs

    rsi = ta.momentum.RSIIndicator(pd.Series(values), window=n)
    rs = rsi.rsi()

    return rs


def RSI(df, n):
    """Relative strength index"""
    
    # # Approximate; good enough
    # gain = pd.Series(values).diff()
    # loss = gain.copy()
    # gain[gain < 0] = 0
    # loss[loss > 0] = 0
    # rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean()
    # return 100 - 100 / (1 + rs)
    

    rsi = ta.momentum.RSIIndicator(df['Close'], window=n)
    df['rsi'] = rsi.rsi()


class Super_RSI(Strategy):
    rsi_length_1d = 14 # RSI lookback periods
    rsi_length_4h = 14
    rsi_length_1h = 14
    rsi_length_30m = 14
    rsi_length_15m = 14
    
    rsi_low = 20
    rsi_high = 80

    # rsi_low_1d = 20
    # rsi_low_4h = 20
    # rsi_low_1h = 20
    # rsi_low_30m = 20
    # rsi_low_15m = 20

    # rsi_high_1d = 80
    # rsi_high_4h = 80
    # rsi_high_1h = 80
    # rsi_high_30m = 80
    # rsi_high_15m = 80

    def init(self):
        
        # Compute RSI
        self.rsi_15m = self.I(RSI_backtesting, self.data.Close, self.rsi_length_15m)
        self.rsi_30m = resample_apply('30min', RSI_backtesting, self.data.Close, self.rsi_length_30m)
        self.rsi_1h  = resample_apply('1H', RSI_backtesting, self.data.Close, self.rsi_length_1h)
        self.rsi_4h  = resample_apply('4H', RSI_backtesting, self.data.Close, self.rsi_length_4h)
        self.rsi_1d  = resample_apply('D', RSI_backtesting, self.data.Close, self.rsi_length_1d)
        
        
    def next(self):
        price = self.data.Close[-1]
       
        # if all conditions are satisfied, enter long.
        if (not self.position and
                self.rsi_15m[-1] <= self.rsi_low and
                self.rsi_30m[-1] <= self.rsi_low and
                self.rsi_1h[-1]  <= self.rsi_low and
                self.rsi_4h[-1]  <= self.rsi_low and
                self.rsi_1d[-1]  <= self.rsi_low and
                
                # self.rsi_15m[-1] <= self.rsi_low_15m and
                # self.rsi_30m[-1] <= self.rsi_low_30m and
                # self.rsi_1h[-1]  <= self.rsi_low_1h and
                # self.rsi_4h[-1]  <= self.rsi_low_4h and
                # self.rsi_1d[-1]  <= self.rsi_low_1d and

                1 == 1):
            self.buy()
        
        # if all conditions are satisfied, close position
        else: 
            if (
                self.rsi_15m[-1] >= self.rsi_high and
                self.rsi_30m[-1] >= self.rsi_high and
                self.rsi_1h[-1]  >= self.rsi_high and
                self.rsi_4h[-1]  >= self.rsi_high and
                self.rsi_1d[-1]  >= self.rsi_high and

                # self.rsi_15m[-1] >= self.rsi_high_15m and
                # self.rsi_30m[-1] >= self.rsi_high_30m and
                # self.rsi_1h[-1]  >= self.rsi_high_1h and
                # self.rsi_4h[-1]  >= self.rsi_high_4h and
                # self.rsi_1d[-1]  >= self.rsi_high_1d and

                1 == 1): 
                self.position.close()
            
def get_data(Symbol, time_frame, start_date):
    msg = f'{Symbol} - getting data'
    print(msg)
    telegram.send_telegram_message('', msg)
    frame = pd.DataFrame(client.get_historical_klines(Symbol,
                                                      time_frame,
                                                      start_date
                                                      ))
    
    frame = frame.iloc[:,:6] # use the first 5 columns
    frame.columns = ['Time','Open','High','Low','Close','Volume'] #rename columns
    frame[['Open','High','Low','Close','Volume']] = frame[['Open','High','Low','Close','Volume']].astype(float) #cast to float
    frame['Date'] = frame['Time'].astype(str) 
    # set the 'date' column as the DataFrame index
    frame.set_index(pd.to_datetime(frame['Date'], unit='ms'), inplace=True) # make human readable timestamp)
    frame = frame.drop(['Date'], axis=1)

    # last row is the last candle and is not yet closed, so we will remove it
    # frame = frame.drop(frame.index[-1])

    return frame

def backtest_super_rsi(symbol):

    # backtest with 4 years of price data 
    #-------------------------------------
    today = date.today() 
    # print(today)
    # today - 4 years - 200 days
    pastdate = today - relativedelta(years=4) - relativedelta(days=200)
    tuple = pastdate.timetuple()
    timestamp = time.mktime(tuple)
    startdate = str(timestamp)
    # startdate = "15 Dec, 2018 UTC"
    # startdate = "12 May, 2022 UTC"
    # startdate = "4 year ago UTC"
    # startdate = "100 day ago UTC"
    #-------------------------------------

    time_frame = client.KLINE_INTERVAL_15MINUTE
    start_date = startdate
    df = get_data(symbol, time_frame, start_date)

    print('backtesting '+symbol)
    bt = Backtest(df, Super_RSI, cash=100000, commission=0.001)
    stats = bt.run()
    print(stats)

    print('optimizing '+symbol)
    stats, heatmap = bt.optimize(
                # rsi_length_1d  = range(10, 20, 5),  # RSI lookback periods
                # rsi_length_4h  = range(10, 20, 5),
                # rsi_length_1h  = range(10, 20, 5),
                # rsi_length_30m = range(10, 20, 5),
                # rsi_length_15m = range(10, 20, 5),
                # rsi_low = range(10, 30, 5),
                # rsi_high = range(70, 90, 5),

                rsi_low = 20,
                rsi_high = 80,

                # rsi_low_15m = range(20, 30, 5),
                # rsi_low_30m = range(20, 30, 5),
                # rsi_low_1h = range(20, 30, 5),
                # rsi_low_4h = range(20, 35, 5),
                # rsi_low_1d = range(20, 35, 5),

                # rsi_high_15m = range(80, 90, 5),
                # rsi_high_30m = range(80, 90, 5),
                # rsi_high_1h = range(80, 90, 5),
                # rsi_high_4h = range(80, 90, 5),
                # rsi_high_1d = range(80, 90, 5),

                maximize='Equity Final [$]',
                return_heatmap=True
                )


    heatmap.dropna()
    df_best = heatmap.sort_values(ascending=False).head(1)
    # print(df_best)

    # rsi_1d = int(df_best.index.get_level_values(0)[0])
    # rsi_4h = int(df_best.index.get_level_values(1)[0])
    # rsi_1h = int(df_best.index.get_level_values(2)[0])
    # rsi_30m = int(df_best.index.get_level_values(3)[0])
    # rsi_15m = int(df_best.index.get_level_values(4)[0])
    # rsi_low = int(df_best.index.get_level_values(5)[0])
    # rsi_high = int(df_best.index.get_level_values(6)[0])

    rsi_low = int(df_best.index.get_level_values(0)[0])
    rsi_high = int(df_best.index.get_level_values(1)[0])

    # rsi_low_15m = int(df_best.index.get_level_values(0)[0])
    # rsi_low_30m = int(df_best.index.get_level_values(1)[0])
    # rsi_low_1h = int(df_best.index.get_level_values(2)[0])
    # rsi_low_4h = int(df_best.index.get_level_values(3)[0])
    # rsi_low_1d = int(df_best.index.get_level_values(4)[0])
    # rsi_high_15m = int(df_best.index.get_level_values(5)[0])
    # rsi_high_30m = int(df_best.index.get_level_values(6)[0])
    # rsi_high_1h = int(df_best.index.get_level_values(7)[0])
    # rsi_high_4h = int(df_best.index.get_level_values(8)[0])
    # rsi_high_1d = int(df_best.index.get_level_values(9)[0])

    return_perc = round(stats['Return [%]'],2)
    buyhold_return_perc = round(stats['Buy & Hold Return [%]'],2)
    backtest_start_date = stats['Start'].strftime('%Y-%m-%d %H:%M:%S') 
    backtest_end_date = stats['End'].strftime('%Y-%m-%d %H:%M:%S')
    num_trades = stats['# Trades']
    df_trades = stats['_trades']

    # lista
    print('Results '+symbol+':')
    
    # print("rsi_1d = ",rsi_1d)
    # print("rsi_4h = ",rsi_4h)
    # print("rsi_1h = ",rsi_1h)
    # print("rsi_30m = ",rsi_30m)
    # print("rsi_15m = ",rsi_15m)

    print("rsi_low = ",rsi_low)
    print("rsi_high = ",rsi_high)
    
    # print("rsi_low_15m = ",rsi_low_15m)
    # print("rsi_low_30m = ",rsi_low_30m)
    # print("rsi_low_1h = ",rsi_low_1h)
    # print("rsi_low_4h = ",rsi_low_4h)
    # print("rsi_low_1D = ",rsi_low_1d)

    # print("rsi_high_15m = ",rsi_high_15m)
    # print("rsi_high_30m = ",rsi_high_30m)
    # print("rsi_high_1h = ",rsi_high_1h)
    # print("rsi_high_4h = ",rsi_high_4h)
    # print("rsi_high_1D = ",rsi_high_1d)

    print("Return [%] = ",round(return_perc,2))
    print("Buy & Hold Return [%] = ",round(buyhold_return_perc,2))
    print("Backtest start date =", backtest_start_date)
    print("Backtest end date =", backtest_end_date)
    print("Trades =", num_trades)
    print(df_trades)

    # values_list
    # values_list = [symbol, rsi_1d, rsi_4h, rsi_1h, rsi_30m, rsi_15m, rsi_low, rsi_high, return_perc, buyhold_return_perc, backtest_start_date, backtest_end_date, num_trades]
    values_list = [symbol, 14, 14, 14, 14, 14, rsi_low, rsi_high, return_perc, buyhold_return_perc, backtest_start_date, backtest_end_date, num_trades]
    
    # insert or update best rsi values to database
    cursor.execute('INSERT OR REPLACE INTO best_rsi VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', values_list)

    # delete existing backtesting trades and add new ones 
    cursor.execute(f'''DELETE FROM backtest_trades WHERE symbol = '{symbol}' ''')
    
    # Insert the new trades into the table
    if not df_trades.empty:
        # Select only the desired columns
        df_trades = df_trades[['EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct', 'EntryTime', 'ExitTime', 'Duration']]
        # add the new "symbol" column
        df_trades.insert(0, 'symbol', symbol)

        # Write the trades to the database
        df_trades.to_sql('backtest_trades', conn, if_exists='append', index=False)

    conn.commit()

#-----------------------------------------------------------------------
# calculate RSI 
#-----------------------------------------------------------------------
def apply_technicals(df, rsi_length):
    # calc RSI
    RSI(df, rsi_length)

def super_rsi(symbol):
    # 15min timeframe
    time_frame = client.KLINE_INTERVAL_15MINUTE

    # get start date
    today = date.today() 
    pastdate = today - relativedelta(days=(rsi_lookback_periods*10)) # used *10 to guarantee the qty of data to calculate rsi on 1D timeframe correctly 
    tuple = pastdate.timetuple()
    timestamp = time.mktime(tuple)
    start_date = str(timestamp)

    # define your SQL query to retrieve the columns you need
    query = f'''SELECT symbol, rsi_1d, rsi_4h, rsi_1h, rsi_30m, rsi_15m, rsi_low, rsi_high
            FROM best_rsi 
            WHERE symbol = '{symbol}' '''

    # execute the query and retrieve the data into a pandas dataframe
    df = pd.read_sql_query(query, conn)

    if not df.empty:
    # store the values in separate variables (assuming the dataframe has only one row)
        # symbol = df['symbol'][0]
        rsi_1d = int(df['rsi_1d'][0])
        rsi_4h = int(df['rsi_4h'][0])
        rsi_1h = int(df['rsi_1h'][0])
        rsi_30m = int(df['rsi_30m'][0])
        rsi_15m = int(df['rsi_15m'][0])
        rsi_low = int(df['rsi_low'][0])
        rsi_high = int(df['rsi_high'][0])
    else:
        rsi_1d = 14
        rsi_4h = 14
        rsi_1h = 14
        rsi_30m = 14
        rsi_15m = 14
        rsi_low = 25
        rsi_high = 80

    # get data from the 15m timeframe
    df_15m = get_data(symbol, time_frame, start_date)
    # Compute RSI
    apply_technicals(df_15m, rsi_15m)

    # check rsi value
    # we want the value before the last that corresponded to the last closed candle. 
    # The last one is the current and the candle is not yet closed
    value = round(df_15m['rsi'].iloc[-2],1) 
    result_low = value <= rsi_low
    result_high = value >= rsi_high

    msg_15m = f"{symbol} - RSI({rsi_15m}) 15m = {value}"
    print(msg_15m)
    telegram.send_telegram_message('', msg_15m)

    if not result_low:
        msg = f"{symbol} - RSI({rsi_15m}) 15m ≤ {rsi_low} - condition not fulfilled"
        print(msg) 
        telegram.send_telegram_message('', msg)
    
    if not result_high:
        msg = f"{symbol} - RSI({rsi_15m}) 15m ≥ {rsi_high} - condition not fulfilled"
        print(msg) 
        telegram.send_telegram_message('', msg)
    
    if not result_low and not result_high:
        return  # Exit the function
    
    if result_low or result_high:
        df_30m = df_15m.resample('30min').last()
        apply_technicals(df_30m, rsi_30m)
        # print(df_30m)

        value = round(df_30m['rsi'].iloc[-2],1)
        
        # if previous was below then test the current
        if result_low:
            result_low = value <= rsi_low
        # if previous was above then test the current
        if result_high:
            result_high = value >= rsi_high
        
        msg_30m = f"{symbol} - RSI({rsi_30m}) 30m = {value}"
        print(msg_30m)
        telegram.send_telegram_message('', msg_30m)

        if not result_low:
            msg = f"{symbol} - RSI({rsi_30m}) 30m ≤ {rsi_low} - condition not fulfilled"
            print(msg) 
            telegram.send_telegram_message('', msg)
    
        if not result_high:
            msg = f"{symbol} - RSI({rsi_30m}) 30m ≥ {rsi_high} - condition not fulfilled"
            print(msg) 
            telegram.send_telegram_message('', msg)
        
        if not result_low and not result_high:
            return  # Exit the function
        
    if result_low or result_high:
        df_1h = df_15m.resample('1H').last()
        apply_technicals(df_1h, rsi_1h)
        
        value = round(df_1h['rsi'].iloc[-2],1)
        result_low = value <= rsi_low
        result_high = value >= rsi_high
        
        msg_1h = f"{symbol} - RSI({rsi_1h}) 1H = {value}"
        print(msg_1h)
        telegram.send_telegram_message('', msg_1h)
        
        if not result_low:
            msg = f"{symbol} - RSI({rsi_1h}) 1H ≤ {rsi_low} - condition not fulfilled"
            print(msg) 
            telegram.send_telegram_message('', msg)
    
        if not result_high:
            msg = f"{symbol} - RSI({rsi_1h}) 1H ≥ {rsi_high} - condition not fulfilled"
            print(msg) 
            telegram.send_telegram_message('', msg)
        
        if not result_low and not result_high:
            return  # Exit the function
    
    if result_low or result_high:
        df_4h = df_15m.resample('4H').last()
        apply_technicals(df_4h, rsi_4h)
        
        value = round(df_4h['rsi'].iloc[-2],1)
        result_low = value <= rsi_low
        result_high = value >= rsi_high
        
        msg_4h = f"{symbol} - RSI({rsi_4h}) 4H = {value}"
        print(msg_4h)
        telegram.send_telegram_message('', msg_4h)
         
        if not result_low:
            msg = f"{symbol} - RSI({rsi_4h}) 4H ≤ {rsi_low} - condition not fulfilled"
            print(msg) 
            telegram.send_telegram_message('', msg)

        if not result_high:
            msg = f"{symbol} - RSI({rsi_4h}) 4H ≥ {rsi_high} - condition not fulfilled"
            print(msg) 
            telegram.send_telegram_message('', msg)

        if not result_low and not result_high:
            return  # Exit the function
        
    if result_low or result_high:
        df_1d = df_15m.resample('D').last()
        apply_technicals(df_1d, rsi_1d)
        
        value = round(df_1d['rsi'].iloc[-2],1)
        result_low = value <= rsi_low
        result_high = value >= rsi_high
        
        msg_1d = f"{symbol} - RSI({rsi_1d}) 1D = {value}"
        print(msg_1d)
        telegram.send_telegram_message('', msg_1d)
         
        if not result_low:
            msg = f"{symbol} - RSI({rsi_1d}) 1D ≤ {rsi_low} - condition not fulfilled"
            print(msg) 
            telegram.send_telegram_message('', msg)
        
        if not result_high:
            msg = f"{symbol} - RSI({rsi_1d}) 1D ≥ {rsi_high} - condition not fulfilled"
            print(msg) 
            telegram.send_telegram_message('', msg)

        if not result_low and not result_high:
            return  # Exit the function

    # if rsi is below min level or above max level in all timeframes we have a super rsi alert!
    if result_low or result_high:
        # get current date and time
        now = datetime.datetime.now()
        # format the current date and time
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

        msg = f"SUPER-RSI alert!\n{formatted_now}\n{symbol}\n{msg_15m}\n{msg_30m}\n{msg_1h}\n{msg_4h}\n{msg_1d}"
        
        if result_low:
            telegram.send_telegram_message(telegram.eEnterTrade, msg)
        elif result_high:
            telegram.send_telegram_message(telegram.eExitTrade, msg)

# backtest_super_rsi("BTCUSDT")
# backtest_super_rsi("ETHUSDT")

if run_mode == "prod":
    # define your SQL query to retrieve the symbols where calc is true
    query = "SELECT symbol FROM symbols WHERE calc = 1"

    # execute the query and retrieve the symbols into a list of tuples
    symbols = conn.execute(query).fetchall()

    # check if the symbols list is empty
    if not symbols:
        print("There are no symbols to calculate.")
    else:
        # iterate over the symbols and call current_super_rsi() for each one
        for symbol in symbols:
            super_rsi(symbol[0])

elif run_mode == "backtest":
    # define your SQL query to retrieve the symbols where calc is true
    query = "SELECT symbol FROM symbols WHERE calc = 1"

    # execute the query and retrieve the symbols into a list of tuples
    symbols = conn.execute(query).fetchall()

    # check if the symbols list is empty
    if not symbols:
        print("There are no symbols to calculate.")
    else:
        # iterate over the symbols and call current_super_rsi() for each one
        for symbol in symbols:
            backtest_super_rsi(symbol[0])

conn.close()

msg = 'SUPER-RSI - End'
print(msg)
telegram.send_telegram_message(telegram.eStop, msg)


