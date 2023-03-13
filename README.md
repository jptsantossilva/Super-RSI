# Super-RSI

The strategy roughly goes like this:

send alerts when all timeframes RSI is in overbough or oversold condition.
```text
    .RSI 1d / 4h / 1h / 30m / 15m <= 20
    .RSI 1d / 4h / 1h / 30m / 15m >= 80
```
## Installation

- Environment variables
```text
    telegramtoken_signals=""
    telegramtoken_signals_alerts=""
    telegramtoken_signals_errors=""
```
- Crontab
 ```
    */15 * * * * cd /home/joaosilva/Documents/GitHub/Super-RSI && python3 super_rsi.py prod
```
- Python Libraries
```text
pip install ta==0.10.2
pip install backtesting==0.3.3
pip install python-binance==1.0.16
```


## Disclaimer
This software is for educational purposes only. Use the software at **your own risk**. The authors and all affiliates assume **no responsibility for your trading results**. **Do not risk money that you are afraid to lose**. There might be **bugs** in the code. This software does not come with **any warranty**.

## 📝 License

This project is [MIT](https://github.com/jptsantossilva/Binance-Trader-EMA-Cross/blob/main/LICENSE.md) licensed.

Copyright © 2023 [João Silva](https://github.com/jptsantossilva)




