import requests
import os
import sys
# import logging
# import yaml

# # log file to store error messages
# log_filename = "main.log"
# logging.basicConfig(filename=log_filename, level=logging.INFO,
#                     format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p -')

# # get settings from config file
# # get trade_against to know which telegram bots to use (BUSD or BTC)
# try:
#     with open("config.yaml", "r") as file:
#         config = yaml.safe_load(file)
#     trade_against = config["trade_against"]

# except FileNotFoundError as e:
#     msg = "Error: The file config.yaml could not be found."
#     msg = msg + " " + sys._getframe(  ).f_code.co_name+" - "+repr(e)
#     print(msg)
#     logging.exception(msg)
#     # telegram.send_telegram_message(telegram.telegramToken_errors, telegram.eWarning, msg)
#     sys.exit(msg) 

# except yaml.YAMLError as e:
#     msg = "Error: There was an issue with the YAML file."
#     msg = msg + " " + sys._getframe(  ).f_code.co_name+" - "+repr(e)
#     print(msg)
#     logging.exception(msg)
#     # telegram.send_telegram_message(telegram.telegramToken_errors, telegram.eWarning, msg)
#     sys.exit(msg) 

# emoji
eStart   = u'\U000025B6'
eStop    = u'\U000023F9'
eWarning = u'\U000026A0'
eEnterTrade = u'\U0001F91E' # crossfingers
eExitTrade  = u'\U0001F91E' # crossfingers
eTradeWithProfit = u'\U0001F44D' # thumbs up
eTradeWithLoss   = u'\U0001F44E' # thumbs down
eInformation = u'\U00002139'

telegram_chat_id = ""
telegramtoken = ""
telegramtoken_alerts = ""
telegramtoken_errors = ""

# telegram timeout 5 seg
telegram_timeout = 5

def read_env_var():
    # environment variables
    
    global telegram_chat_id
    global telegramtoken
    global telegramtoken_alerts
    global telegramtoken_errors

    try:
        telegram_chat_id = os.environ.get('telegram_chat_id')
        telegramtoken = os.environ.get('telegramtoken_signals') # all messages
        telegramtoken_alerts = os.environ.get('telegramtoken_signals_alerts') # just alerts
        telegramtoken_errors = os.environ.get('telegramtoken_signals_errors') # just errors

    except KeyError as e: 
        msg = sys._getframe(  ).f_code.co_name+" - "+repr(e)
        print(msg)
        # logging.exception(msg)

# fulfill telegram vars
read_env_var()

def send_telegram_message(emoji, msg):

    msg = remove_chars_exceptions(msg)

    max_limit = 4096
    if emoji:
        additional_characters = eWarning+" <pre> </pre>Part [10/99]"
    else:
        additional_characters = "<pre> </pre>Part [10/99]"

    if emoji:
        msg = emoji+" "+msg

    num_additional_characters = len(additional_characters)
    max_limit = 4096 - num_additional_characters 

    if len(msg+additional_characters) > max_limit:
        # Split the message into multiple parts
        message_parts = [msg[i:i+max_limit] for i in range(0, len(msg), max_limit)]
        n_parts = len(message_parts)
        for i, part in enumerate(message_parts):
            print(f'Part [{i+1}/{n_parts}]\n{part}')
        
            lmsg = "<pre>Part ["+str(i+1)+"/"+str(n_parts)+"]\n"+part+"</pre>"
            
            params = {
            "chat_id": telegram_chat_id,
            "text": lmsg,
            "parse_mode": "HTML",
            }

            try:
                # if message is a warning, send message to the errors telegram chat bot 
                if emoji == eWarning:
                    resp = requests.post("https://api.telegram.org/bot{}/sendMessage".format(telegramtoken_errors), params=params, timeout=telegram_timeout)
                    resp.raise_for_status()
                
                # if message is a enter or exit trade alert, send message to the alerts telegram chat bot 
                if emoji in [eEnterTrade, eExitTrade]:
                    resp = requests.post("https://api.telegram.org/bot{}/sendMessage".format(telegramtoken_alerts), params=params, timeout=telegram_timeout)
                    resp.raise_for_status()
                
                # send all message to run telegram chat bot
                resp = requests.post("https://api.telegram.org/bot{}/sendMessage".format(telegramtoken), params=params, timeout=telegram_timeout)
                resp.raise_for_status()

            except requests.exceptions.HTTPError as errh:
                msg = sys._getframe(  ).f_code.co_name+" - An Http Error occurred:" + repr(errh)
                print(msg)
                # logging.exception(msg)
            except requests.exceptions.ConnectionError as errc:
                msg = sys._getframe(  ).f_code.co_name+" - An Error Connecting to the API occurred:" + repr(errc)
                print(msg)
                # logging.exception(msg)
            except requests.exceptions.Timeout as errt:
                msg = sys._getframe(  ).f_code.co_name+" - A Timeout Error occurred:" + repr(errt)
                print(msg)
                # logging.exception(msg)
            except requests.exceptions.RequestException as err:
                msg = sys._getframe(  ).f_code.co_name+" - An Unknown Error occurred" + repr(err)
                print(msg)
                # logging.exception(msg) 
            
    else: # message size < max size 4096

        # To fix the issues with dataframes alignments, the message is sent as HTML and wraped with <pre> tag
        # Text in a <pre> element is displayed in a fixed-width font, and the text preserves both spaces and line breaks
        lmsg = "<pre>"+msg+"</pre>"

        params = {
        "chat_id": telegram_chat_id,
        "text": lmsg,
        "parse_mode": "HTML",
        }
        
        try:            
            # if message is a warning, send message to the errors telegram chat bot 
            if emoji == eWarning:
                resp = requests.post("https://api.telegram.org/bot{}/sendMessage".format(telegramtoken_errors), params=params, timeout=telegram_timeout)
                resp.raise_for_status()
            
            # if message is a enter or exit trade alert, send message to the alerts telegram chat bot 
            if emoji in [eEnterTrade, eExitTrade]:
                resp = requests.post("https://api.telegram.org/bot{}/sendMessage".format(telegramtoken_alerts), params=params, timeout=telegram_timeout)
                resp.raise_for_status()
            
            # send all message to run telegram chat bot
            resp = requests.post("https://api.telegram.org/bot{}/sendMessage".format(telegramtoken), params=params, timeout=telegram_timeout)
            resp.raise_for_status()

        except requests.exceptions.HTTPError as errh:
            msg = sys._getframe(  ).f_code.co_name+" - An Http Error occurred:" + repr(errh)
            print(msg)
            # logging.exception(msg)
        except requests.exceptions.ConnectionError as errc:
            msg = sys._getframe(  ).f_code.co_name+" - An Error Connecting to the API occurred:" + repr(errc)
            print(msg)
            # logging.exception(msg)
        except requests.exceptions.Timeout as errt:
            msg = sys._getframe(  ).f_code.co_name+" - A Timeout Error occurred:" + repr(errt)
            print(msg)
            # logging.exception(msg)
        except requests.exceptions.RequestException as err:
            msg = sys._getframe(  ).f_code.co_name+" - An Unknown Error occurred" + repr(err)
            print(msg)
            # logging.exception(msg)

def send_telegram_photo(file_name):
    
    # get current dir
    cwd = os.getcwd()
    limg = cwd+"/"+file_name
    # print(limg)
    oimg = open(limg, 'rb')
    url = f"https://api.telegram.org/bot{telegramtoken}/sendPhoto?chat_id={telegram_chat_id}"
    
    try:
        resp = requests.post(url, files={'photo':oimg}, timeout=telegram_timeout) # this sends the message
        resp.raise_for_status()

    except requests.exceptions.HTTPError as errh:
        msg = sys._getframe(  ).f_code.co_name+" - An Http Error occurred:" + repr(errh)
        print(msg)
        # logging.exception(msg)
    except requests.exceptions.ConnectionError as errc:
        msg = sys._getframe(  ).f_code.co_name+" - An Error Connecting to the API occurred:" + repr(errc)
        print(msg)
        # logging.exception(msg)
    except requests.exceptions.Timeout as errt:
        msg = sys._getframe(  ).f_code.co_name+" - A Timeout Error occurred:" + repr(errt)
        print(msg)
        # logging.exception(msg)
    except requests.exceptions.RequestException as err:
        msg = sys._getframe(  ).f_code.co_name+" - An Unknown Error occurred" + repr(err)
        print(msg)
        # logging.exception(msg)

def send_telegram_file(file_name):
    
    # get current dir
    cwd = os.getcwd()
    file = cwd+"/"+file_name
    # print(limg)
    url = f"https://api.telegram.org/bot{telegramtoken}/sendDocument"
    
    try:
        with open(file, 'rb') as f:
            resp = requests.post(url, data={'chat_id': telegram_chat_id},files={'document':f}, timeout=telegram_timeout) # this sends the message
            resp.raise_for_status()

    except requests.exceptions.HTTPError as errh:
        msg = sys._getframe(  ).f_code.co_name+" - An Http Error occurred:" + repr(errh)
        print(msg)
        # logging.exception(msg)
    except requests.exceptions.ConnectionError as errc:
        msg = sys._getframe(  ).f_code.co_name+" - An Error Connecting to the API occurred:" + repr(errc)
        print(msg)
        # logging.exception(msg)
    except requests.exceptions.Timeout as errt:
        msg = sys._getframe(  ).f_code.co_name+" - A Timeout Error occurred:" + repr(errt)
        print(msg)
        # logging.exception(msg)
    except requests.exceptions.RequestException as err:
        msg = sys._getframe(  ).f_code.co_name+" - An Unknown Error occurred" + repr(err)
        print(msg)
        # logging.exception(msg)


def remove_chars_exceptions(string):
    # this is useful for the binance errors messages

    # define the characters to be removed
    chars_to_remove = ['<', '>', '{', '}', "'", '"']

    # use a loop to replace each character with an empty string
    for char in chars_to_remove:
        string = string.replace(char, '')

    return string
