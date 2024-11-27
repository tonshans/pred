import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from binance.client import Client
import pandas_ta as ta
import joblib
#import matplotlib.pyplot as plt
from cmd import Cmd 
import re
import requests
from time import sleep

## Load model2nya
model1 = joblib.load("models/model1.jl")
model2 = joblib.load("models/model2.jl")
model3 = joblib.load("models/model3.jl")
model4 = joblib.load("models/model4.jl")
model5 = joblib.load("models/model5.jl")
model6 = joblib.load("models/model6.jl")
model7 = joblib.load("models/model7.jl")

def get_klines(pair,timeframe='1d'): 
    tf = timeframe.lower()
    candle_needed = 60
    if tf == '1w':
        interfal = client.KLINE_INTERVAL_1WEEK 
        dateparser_range = str(candle_needed) + " week ago UTC"
    elif tf == '3d':
        interfal = client.KLINE_INTERVAL_3DAY  
        dateparser_range = str(3*candle_needed) + " day ago UTC"
    elif tf == '1d':
        interfal = client.KLINE_INTERVAL_1DAY
        dateparser_range = str(candle_needed) + " day ago UTC"
    elif tf == '12h':
        interfal = client.KLINE_INTERVAL_12HOUR 
        dateparser_range = str(12*candle_needed) + " hour ago UTC"
    elif tf == '8h':
        interfal = client.KLINE_INTERVAL_8HOUR 
        dateparser_range = str(8*candle_needed) + " hour ago UTC"
    elif tf == '6h':
        interfal = client.KLINE_INTERVAL_6HOUR 
        dateparser_range = str(6*candle_needed) + " hour ago UTC"
    elif tf == '4h':
        interfal = client.KLINE_INTERVAL_4HOUR 
        dateparser_range = str(4*candle_needed) + " hour ago UTC"
    elif tf == '2h':
        interfal = client.KLINE_INTERVAL_2HOUR 
        dateparser_range = str(2*candle_needed) + " hour ago UTC"
    elif tf == '1h':
        interfal = client.KLINE_INTERVAL_1HOUR 
        dateparser_range = str(candle_needed) + " hour ago UTC"
    elif tf == '30m':
        interfal = client.KLINE_INTERVAL_30MINUTE 
        dateparser_range = str(30*candle_needed) + " minute ago UTC"
    elif tf == '15m':
        interfal = client.KLINE_INTERVAL_15MINUTE 
        dateparser_range = str(15*candle_needed) + " minute ago UTC"
    elif tf == '5m':
        interfal = client.KLINE_INTERVAL_5MINUTE 
        dateparser_range = str(5*candle_needed) + " minute ago UTC"
    else:
        return None
    
    try:
        bk = client.get_historical_klines(pair, interfal, dateparser_range)

        klines = []
        ##  timw now dirubah mendekati utc
        time_now = datetime.now() - timedelta(hours=7, minutes=58)
        if len(bk) > 0 :
            for k in bk:
                if time_now > datetime.fromtimestamp(k[6]/1000):
                    klines.append({#'Open_time': k[0],
                                'Open': float(k[1]),
                                'High': float(k[2]),
                                'Low': float(k[3]),
                                'Close': float(k[4]),
                                'Volume': float(k[5]),
                                })
        klines_df = pd.DataFrame.from_dict(klines)

        ## ntar di crosscheck, apakah perlu ordering di balik, butuhnya sorting dari oldest ke newest
        #klines_df.iloc[::-1].reset_index(drop=True)  ## gak perlu di balik sortiran nya, sudah pass oldes ke newest datanya
        return klines_df
    except:
        return None


def get_idx_klines(pair,timeframe='1d',candle_to_fetch=61):
    idx_endpoint = "https://indodax.com/"

    try:
        klines = []
        _number_tf_str = re.findall(r'\d+', timeframe)
        if len(_number_tf_str) > 0:
            number_tf_str = _number_tf_str[0]
            if 'm' in timeframe:
                tf_in_minute = int(number_tf_str)
                tf_idx = number_tf_str
            elif 'h' in timeframe:
                tf_in_minute = int(number_tf_str) * 60
                tf_idx = str(tf_in_minute)
            elif 'd' in timeframe:
                tf_in_minute = int(number_tf_str) * (60*24)
                tf_idx = number_tf_str + 'D'
            elif 'w' in timeframe:
                tf_in_minute = int(number_tf_str) * ((60*24)*7)
                tf_idx = number_tf_str + 'W'

        time_now = datetime.now()
        time_start = time_now - timedelta(minutes=tf_in_minute * candle_to_fetch)
        timestamp_now = str(round(time_now.timestamp()))
        timestamp_start = str(round(time_start.timestamp()))

        ohlc_endpoint = "tradingview/history_v2?from=%s&symbol=%s&tf=%s&to=%s" %(timestamp_start,pair,tf_idx,timestamp_now)

        resp = requests.get(idx_endpoint + ohlc_endpoint)
        for k in resp.json()[:-1]:
            klines.append({#'Time': k['Time'],
                    'Open': float(k['Open']),
                    'High': float(k['High']),
                    'Low': float(k['Low']),
                    'Close': float(k['Close']),
                    'Volume': float(k['Volume']),
                    })
        return pd.DataFrame.from_dict(klines)
    except:
        return None

def generate_indi(klines):
    ''' 
    target: bv.loc[i,'TYP_MOV'] = 1 if bv.loc[i,'TYP'] > bv.loc[i,'TYP_MA8'] else -1

    '''
    try:
        bv = klines 
        bv['VOL_MA8'] = ta.sma(bv['Volume'], length=8)
        bv['VOL_MA13'] = ta.sma(bv['Volume'], length=13)
        bv['VOL_MA21'] = ta.sma(bv['Volume'], length=21)
        bv['VOL_RSI'] = ta.rsi(bv['Volume'])
        #_vol_stoch = ta.stochrsi(bv['Volume'])
        #bv['VOL_SK'] = _vol_stoch['STOCHRSIk_14_14_3_3']
        #bv['VOL_SD'] = _vol_stoch['STOCHRSId_14_14_3_3']
        _vol_rsi_stoch = ta.stochrsi(bv['VOL_RSI'])
        bv['VOL_RSI_SK'] = _vol_rsi_stoch['STOCHRSIk_14_14_3_3']
        bv['VOL_RSI_SD'] = _vol_rsi_stoch['STOCHRSId_14_14_3_3']
        _vol_macd = ta.macd(bv['Volume']) 		
        bv['VOL_MACD'] = _vol_macd['MACD_12_26_9']
        bv['VOL_MACDH'] = _vol_macd['MACDh_12_26_9']
        bv['VOL_MACDS'] = _vol_macd['MACDs_12_26_9']

        bv['TYP'] = (bv['High']+bv['Low']+bv['Close'])/3
        bv['TYP_MA8'] = ta.sma(bv['TYP'], length=8)
        bv['TYP_MA13'] = ta.sma(bv['TYP'], length=13)
        bv['TYP_MA21'] = ta.sma(bv['TYP'], length=21)

        bv['TYP_RSI'] = ta.rsi(bv['TYP'])
        _typ_rsi_stoch = ta.stochrsi(bv['TYP_RSI'])
        bv['TYP_RSI_SK'] = _typ_rsi_stoch['STOCHRSIk_14_14_3_3']
        bv['TYP_RSI_SD'] = _typ_rsi_stoch['STOCHRSId_14_14_3_3']
        _typ_macd = ta.macd(bv['TYP']) 		
        bv['TYP_MACD'] = _typ_macd['MACD_12_26_9']
        bv['TYP_MACDH'] = _typ_macd['MACDh_12_26_9']
        bv['TYP_MACDS'] = _typ_macd['MACDs_12_26_9']
        _typ_stoch = ta.stochrsi(bv['TYP'])
        bv['TYP_SK'] = _typ_stoch['STOCHRSIk_14_14_3_3']
        bv['TYP_SD'] = _typ_stoch['STOCHRSId_14_14_3_3']


        ha = ta.ha(bv['Open'],bv['High'],bv['Low'],bv['Close'])
        bv['HA_O'] = ha['HA_open']
        bv['HA_H'] = ha['HA_high']
        bv['HA_L'] = ha['HA_low']
        bv['HA_C'] = ha['HA_close']

        bv['HA_TYP'] = (bv['HA_H']+bv['HA_L']+bv['HA_C'])/3
        bv['HA_TYP_MA8'] = ta.sma(bv['HA_TYP'], length=8)
        bv['HA_TYP_MA13'] = ta.sma(bv['HA_TYP'], length=13)
        bv['HA_TYP_MA21'] = ta.sma(bv['HA_TYP'], length=21)
        bv['HA_TYP_RSI'] = ta.rsi(bv['HA_TYP'])
        _ha_rsi_stoch = ta.stochrsi(bv['HA_TYP_RSI'])
        bv['HA_TYP_RSI_SK'] = _ha_rsi_stoch['STOCHRSIk_14_14_3_3']
        bv['HA_TYP_RSI_SD'] = _ha_rsi_stoch['STOCHRSId_14_14_3_3']
        _ha_typ_macd = ta.macd(bv['HA_TYP']) 		
        bv['HA_TYP_MACD'] = _ha_typ_macd['MACD_12_26_9']
        bv['HA_TYP_MACDH'] = _ha_typ_macd['MACDh_12_26_9']
        bv['HA_TYP_MACDS'] = _ha_typ_macd['MACDs_12_26_9']

        #row = 1 # index di mulai dari 0, tapi hitung row di mulai dari 1
        for i,v in bv.iterrows():
            if i >= 1:
                bv.loc[i,'HA_MOV'] = 1 if (bv.loc[i,'HA_C'] > bv.loc[i,'HA_O']) else -1

                bv.loc[i,"O_CH"] = (bv.loc[i,'Open']-bv.loc[i-1,'Open'])/bv.loc[i,'Open']
                bv.loc[i,"H_CH"] = (bv.loc[i,'High']-bv.loc[i-1,'High'])/bv.loc[i,'High']
                bv.loc[i,"L_CH"] = (bv.loc[i,'Low']-bv.loc[i-1,'Low'])/bv.loc[i,'Low']
                bv.loc[i,"C_CH"] = (bv.loc[i,'Close']-bv.loc[i-1,'Close'])/bv.loc[i,'Close']
                bv.loc[i,"VOL_CH"] = (bv.loc[i,'Volume']-bv.loc[i-1,'Volume'])/bv.loc[i,'Volume']
                bv.loc[i,"TYP_CH"] = (bv.loc[i,'TYP']-bv.loc[i-1,'TYP'])/bv.loc[i,'TYP']

                bv.loc[i,"HA_O_CH"] = (bv.loc[i,'HA_O']-bv.loc[i-1,'HA_O'])/bv.loc[i,'HA_O']
                bv.loc[i,"HA_H_CH"] = (bv.loc[i,'HA_H']-bv.loc[i-1,'HA_H'])/bv.loc[i,'HA_H']
                bv.loc[i,"HA_L_CH"] = (bv.loc[i,'HA_L']-bv.loc[i-1,'HA_L'])/bv.loc[i,'HA_L']
                bv.loc[i,"HA_C_CH"] = (bv.loc[i,'HA_C']-bv.loc[i-1,'HA_C'])/bv.loc[i,'HA_C']
                
            if i >=7:
                bv.loc[i,'TYP_MOV'] = 1 if bv.loc[i,'TYP'] > bv.loc[i,'TYP_MA8'] else -1

                
        bv['TYP_CH_RSI'] = ta.rsi(bv['TYP_CH'])
        _typ_ch_stoch = ta.stochrsi(bv['TYP_CH'])
        bv['TYP_CH_SK'] = _typ_ch_stoch['STOCHRSIk_14_14_3_3']
        bv['TYP_CH_SD'] = _typ_ch_stoch['STOCHRSId_14_14_3_3']

        bv['TYP_MOV_MA8'] = ta.sma(bv['TYP_MOV'], length=8)
        bv['TYP_MOV_MA13'] = ta.sma(bv['TYP_MOV'], length=13)
        bv['TYP_MOV_MA21'] = ta.sma(bv['TYP_MOV'], length=21)
        _typ_mov_stoch = ta.stochrsi(bv['TYP_MOV'])
        bv['TYP_MOV_SK'] = _typ_mov_stoch['STOCHRSIk_14_14_3_3']
        bv['TYP_MOV_SD'] = _typ_mov_stoch['STOCHRSId_14_14_3_3']
        _typ_mov_macd = ta.macd(bv['TYP_MOV']) 		
        bv['TYP_MOV_MACD'] = _typ_mov_macd['MACD_12_26_9']
        bv['TYP_MOV_MACDH'] = _typ_mov_macd['MACDh_12_26_9']
        bv['TYP_MOV_MACDS'] = _typ_mov_macd['MACDs_12_26_9']
        bv['TYP_MOV_RSI'] = ta.rsi(bv['TYP_MOV'])
        bv['TYP_MOV_RSI_MA8'] = ta.sma(bv['TYP_MOV_RSI'], length=8)
        bv['TYP_MOV_RSI_MA13'] = ta.sma(bv['TYP_MOV_RSI'], length=13)
        bv['TYP_MOV_RSI_MA21'] = ta.sma(bv['TYP_MOV_RSI'], length=21)
        _typ_mov_rsi_stoch = ta.stochrsi(bv['TYP_MOV_RSI'])
        bv['TYP_MOV_RSI_SK'] = _typ_mov_rsi_stoch['STOCHRSIk_14_14_3_3']
        bv['TYP_MOV_RSI_SD'] = _typ_mov_rsi_stoch['STOCHRSId_14_14_3_3']
    
            
        bv['HA_MOV_MA8'] = ta.sma(bv['HA_MOV'], length=8)
        bv['HA_MOV_MA13'] = ta.sma(bv['HA_MOV'], length=13)
        bv['HA_MOV_MA21'] = ta.sma(bv['HA_MOV'], length=21)
        _ha_mov_stoch = ta.stochrsi(bv['HA_MOV'])
        bv['HA_MOV_SK'] = _ha_mov_stoch['STOCHRSIk_14_14_3_3']
        bv['HA_MOV_SD'] = _ha_mov_stoch['STOCHRSId_14_14_3_3']
        _ha_mov_macd = ta.macd(bv['HA_MOV']) 		
        bv['HA_MOV_MACD'] = _ha_mov_macd['MACD_12_26_9']
        bv['HA_MOV_MACDH'] = _ha_mov_macd['MACDh_12_26_9']
        bv['HA_MOV_MACDS'] = _ha_mov_macd['MACDs_12_26_9']
        bv['HA_MOV_RSI'] = ta.rsi(bv['HA_MOV'])
        bv['HA_MOV_RSI_MA8'] = ta.sma(bv['HA_MOV_RSI'], length=8)
        bv['HA_MOV_RSI_MA13'] = ta.sma(bv['HA_MOV_RSI'], length=13)
        bv['HA_MOV_RSI_MA21'] = ta.sma(bv['HA_MOV_RSI'], length=21)
        _ha_mov_rsi_stoch = ta.stochrsi(bv['HA_MOV_RSI'])
        bv['HA_MOV_RSI_SK'] = _ha_mov_rsi_stoch['STOCHRSIk_14_14_3_3']
        bv['HA_MOV_RSI_SD'] = _ha_mov_rsi_stoch['STOCHRSId_14_14_3_3']

        #return bv.drop(columns=['Open_time'])
        return bv
    except Exception as e:
        print(e)
        return None



def predict_this(indies):
    p = []
    p.append(model1.predict(indies)[0])
    p.append(model2.predict(indies)[0])
    p.append(model3.predict(indies)[0])
    p.append(model4.predict(indies)[0])
    p.append(model5.predict(indies)[0])
    p.append(model6.predict(indies)[0])
    p.append(model7.predict(indies)[0])
    return p


def predict(pair,timeframe='1d'):
    '''
    cuma wrapper biar enak manggilnya
    predict_this(generate_indi(get_klines(pair).iloc[[-1]]))
    '''
    klines = get_klines(pair,timeframe=timeframe)
    if klines is None:
        return None
    indi = generate_indi(klines).iloc[[-1]]
    #return [indi['MOV'].values[0]] + predict_this(generate_indi(klines).iloc[[-1]])
    return [indi['TYP_MOV'].values[0]] + predict_this(indi)

def idx_predict(pair,timeframe='1d'):
    klines = get_idx_klines(pair,timeframe=timeframe)
    if klines is None:
        return None
    indi = generate_indi(klines).iloc[[-1]]
    return [indi['TYP_MOV'].values[0]] + predict_this(indi)

def pred_plotfile(pred,pair):
    '''
    prediction to plot
    hasil di simpan dalam bentuk file pnd biar gampang di kirim sama bot
    '''
    prefix_dir = '/dev/shm/'
    time_now = datetime.now().strftime("%d-%m%y-%H%M")
    plot_filename = prefix_dir + "/" + pair + "-" + time_now + ".png"

    pred_df = pd.DataFrame(pred, columns=['Prediction'])
    plt.figure().set_figheight(1.5)
    plt.title(pair + ' - ' + time_now)
    plt.xlabel("Next Move")
    plt.plot(pred_df.index,pred_df['Prediction'],linewidth=3)
    plt.savefig(plot_filename)

    return plot_filename


def print_predict(pair,timeframe='1d',default_pair='USDT'):
    pred = predict(pair,timeframe=timeframe)
    if pred is None:
        return None
    time_now = datetime.now().strftime("%d-%m %H:%M")
    if len(pair.replace(default_pair,'')) < 4 :
        pred_str = '  ' +  pair + '--' + time_now + '-> '
    else:
        pred_str = '  ' +  pair + '-' + time_now + '-> '
    c = 0
    for p in pred:
        if p == 1:
            pred_str += '\033[1;32;40m [' + str(c)+']'
        else:
            pred_str += '\033[1;31;40m (' + str(c)+')'
        c +=1
    print(pred_str + '\033[0m')


def print_predict_tf_overlap(pair, tfs=['1d', '4h', '30m']):
    time_now = datetime.now().strftime("%d-%m %H:%M")
    print('  _' + pair + '_' + time_now + ':')
    for tf in tfs:
        pred = predict(pair,timeframe=tf)
        if pred is None:
            return None
        if len(tf)<3:
            pred_str = '  ' + tf + ' :-> '
        else:
            pred_str = '  ' + tf + ':-> '
        c = 0
        for p in pred:
            if p == 1:
                pred_str += '\033[1;32;40m [' + str(c)+']'
            else:
                pred_str += '\033[1;31;40m (' + str(c)+')'
            c +=1
        print(pred_str + '\033[0m')

#######################################
class predCmd(Cmd):
    timeframe = '1d' ##default timeframe
    default_pair = 'USDT'
    prompt = "\n" + timeframe+ ':' + default_pair +":> "

    def do_p(self, args):
        'p maksudnya predict, \nargument bisa di isi banyak pair pakai space separator ya'
        arg_split = args.split(' ')
        for arg in arg_split:
            print_predict(arg.upper() + self.default_pair, timeframe=self.timeframe, default_pair=self.default_pair)

    def do_ptf(self, args):
        'Predict timeframe overlap, nge-predict 1 pair dalam preset timeframe\nptf [.. pair ..]'
        arg_split = args.split(' ')
        for arg in arg_split:
            print_predict_tf_overlap(arg.upper() + self.default_pair)

    def do_pr(self, args):
        'preset list'
        preset_pairs = ['BTC', 'FET', 'FTM', 'HBAR', 'ZIL', 'LIT']
        for pair in preset_pairs:
            print_predict(pair + self.default_pair, self.timeframe)
    
    def do_tel(self, args):
        'cek status tel di idx'
        tfs=['1d', '4h', '30m']
        pair = 'TELIDR'

        time_now = datetime.now().strftime("%d-%m %H:%M")
        print('  _IDX_' + pair + '_' + time_now + ':')
        for tf in tfs:
            pred = idx_predict(pair,timeframe=tf)
            if pred is None:
                return None
            if len(tf)<3:
                pred_str = '  ' + tf + ' :-> '
            else:
                pred_str = '  ' + tf + ':-> '
            c = 0
            for p in pred:
                if p == 1:
                    pred_str += '\033[1;32;40m [' + str(c)+']'
                else:
                    pred_str += '\033[1;31;40m (' + str(c)+')'
                c +=1
            print(pred_str + '\033[0m')
            #sleep(2) ##biar gak error2 dia waktu query api nya

    def do_tf(self, args):
        'timeframe [ 1w 3d 1d 12h 8h 6h 4h 2h 1h 30m 15m 5m ]\ntf 4h\n'
        if len(args) == 0:
            print(self.timeframe)
        else:
            self.timeframe = args
            self.prompt = "\n" + self.timeframe+ ':' + self.default_pair +":> "

    def do_dpair(self, args):
        'set default pair. \nex: dpair USDT'
        self.default_pair = args.upper()
        self.prompt = "\n" + self.timeframe+ ':' + self.default_pair +":> "

    def do_exit(self, args):
        'Keluar'
        return True
#######################################


if __name__ == '__main__':
    client = Client('', '')

    app = predCmd()
    app.cmdloop('Enter a command to predict trend movement \n[p/ptf/pr/tf/tel/help]:')
