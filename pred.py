#import sys
import os.path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from datetime import datetime, timedelta
import pandas as pd
#import numpy as np
import pandas_ta as ta
import joblib
#import matplotlib.pyplot as plt
from cmd import Cmd 
import re
import requests
from tinydb import TinyDB, Query
from tinydb.storages import JSONStorage
from tinydb_serialization import SerializationMiddleware
from tinydb_serialization.serializers import DateTimeSerializer
import tinydb_encrypted_jsonstorage as tae
serialization = SerializationMiddleware(JSONStorage)
serialization.register_serializer(DateTimeSerializer(), 'TinyDate')

try:
    from binance.client import Client
    bin_client = Client('', '')
    connect_to_binance = True
except:
    connect_to_binance = False



def get_ticker(exchange, debug=False):
    'return : { __pair__ : __last_price__ }'
    price = {}
    if exchange.upper() == 'IDX':
        try:
            resp = requests.get('https://indodax.com/api/ticker_all')
            resp_json = resp.json()['tickers'] 
            for i in resp_json:
                c_price = resp_json[i]
                pair_name = i.replace("_", "").upper() #supaya standard penyebutannya
                price[pair_name] = c_price['last']
        except Exception as e:
            if debug:
                print('get_ticker : IDX ERROR')
                print(e)
            return None
        
    elif exchange.upper() == 'BIN': #binance
        if not connect_to_binance:
            if debug:
                print("get_ticker : Tidak bisa konek ke binance.")
            return None
        for p in bin_client.get_all_tickers():
            price[p['symbol']] = p['price']
    else:
        return None  ## exchange lainnya belum disupport.
    return price


def get_klines(exchange, pair, timeframe='1d', candle_to_fetch=61, debug=False):
    if debug:
        print(f"get_kines input : {exchange}, {pair}, {timeframe}, {candle_to_fetch}, {debug}")

    if exchange.upper() == 'IDX':
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

            if debug:
                print((idx_endpoint + ohlc_endpoint))
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
    
    ## Binance
    elif exchange == 'BIN': 
        if not connect_to_binance:
            print("Tidak bisa konek ke binance.")
            return None
        
        tf = timeframe.lower()
        if tf == '1w':
            interfal = bin_client.KLINE_INTERVAL_1WEEK 
            dateparser_range = str(candle_to_fetch) + " week ago UTC"
        elif tf == '3d':
            interfal = bin_client.KLINE_INTERVAL_3DAY  
            dateparser_range = str(3*candle_to_fetch) + " day ago UTC"
        elif tf == '1d':
            interfal = bin_client.KLINE_INTERVAL_1DAY
            dateparser_range = str(candle_to_fetch) + " day ago UTC"
        elif tf == '12h':
            interfal = bin_client.KLINE_INTERVAL_12HOUR 
            dateparser_range = str(12*candle_to_fetch) + " hour ago UTC"
        elif tf == '8h':
            interfal = bin_client.KLINE_INTERVAL_8HOUR 
            dateparser_range = str(8*candle_to_fetch) + " hour ago UTC"
        elif tf == '6h':
            interfal = bin_client.KLINE_INTERVAL_6HOUR 
            dateparser_range = str(6*candle_to_fetch) + " hour ago UTC"
        elif tf == '4h':
            interfal = bin_client.KLINE_INTERVAL_4HOUR 
            dateparser_range = str(4*candle_to_fetch) + " hour ago UTC"
        elif tf == '2h':
            interfal = bin_client.KLINE_INTERVAL_2HOUR 
            dateparser_range = str(2*candle_to_fetch) + " hour ago UTC"
        elif tf == '1h':
            interfal = bin_client.KLINE_INTERVAL_1HOUR 
            dateparser_range = str(candle_to_fetch) + " hour ago UTC"
        elif tf == '30m':
            interfal = bin_client.KLINE_INTERVAL_30MINUTE 
            dateparser_range = str(30*candle_to_fetch) + " minute ago UTC"
        elif tf == '15m':
            interfal = bin_client.KLINE_INTERVAL_15MINUTE 
            dateparser_range = str(15*candle_to_fetch) + " minute ago UTC"
        elif tf == '5m':
            interfal = bin_client.KLINE_INTERVAL_5MINUTE 
            dateparser_range = str(5*candle_to_fetch) + " minute ago UTC"
        else:
            return None
        
        try:
            bk = bin_client.get_historical_klines(pair, interfal, dateparser_range)
            
            if debug:
                print('__get_klines_debug___')
                print(bk)

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


def get_pair_list(exchange, fiat='USDT', debug=False):
    pairs = []
    ticker = get_ticker(exchange, debug=debug)
    if ticker is None:
        return None
    for pair in ticker:
        if pair.endswith(fiat):
            pairs.append(pair)
    return pairs


class Pred():
    debug = False
    model = 'default'
    available_model = ['default','novol']
    jl_model1 = None
    jl_model2 = None
    jl_model3 = None
    jl_model4 = None
    jl_model5 = None
    jl_model6 = None
    jl_model7 = None

    def __init__(self,model=None):
        if model is not None:
            self.model = model
        self.load_model()

    def set_model_type(self, model):
        if model in self.available_model:
            self.model = model

    def load_model(self):
        if self.model == 'default':
            self.jl_model1 = joblib.load("models/model1.jl")
            self.jl_model2 = joblib.load("models/model2.jl")
            self.jl_model3 = joblib.load("models/model3.jl")
            self.jl_model4 = joblib.load("models/model4.jl")
            self.jl_model5 = joblib.load("models/model5.jl")
            self.jl_model6 = joblib.load("models/model6.jl")
            self.jl_model7 = joblib.load("models/model7.jl")
        else:
            self.jl_model1 = joblib.load("models/"+self.model+"/model1.jl")
            self.jl_model2 = joblib.load("models/"+self.model+"/model2.jl")
            self.jl_model3 = joblib.load("models/"+self.model+"/model3.jl")
            self.jl_model4 = joblib.load("models/"+self.model+"/model4.jl")
            self.jl_model5 = joblib.load("models/"+self.model+"/model5.jl")
            self.jl_model6 = joblib.load("models/"+self.model+"/model6.jl")
            self.jl_model7 = joblib.load("models/"+self.model+"/model7.jl")


    def generate_indi(self, klines, debug=False):
        ''' 
        target: bv.loc[i,'TYP_MOV'] = 1 if bv.loc[i,'TYP'] > bv.loc[i,'TYP_MA8'] else -1

        '''
        try:
            bv = klines 
            if self.model == 'default':
                bv['VOL_MA8'] = ta.sma(bv['Volume'], length=8)
                bv['VOL_MA13'] = ta.sma(bv['Volume'], length=13)
                bv['VOL_MA21'] = ta.sma(bv['Volume'], length=21)
                bv['VOL_RSI'] = ta.rsi(bv['Volume'])
                ##_vol_stoch = ta.stochrsi(bv['Volume'])
                ##bv['VOL_SK'] = _vol_stoch['STOCHRSIk_14_14_3_3']
                ##bv['VOL_SD'] = _vol_stoch['STOCHRSId_14_14_3_3']
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

            #return bv[np.isfinite(bv).all(1)] #exclude yang mengandung infinate value
            if self.model == 'default':
                return bv
            else:    
                return bv.drop(columns=['Volume'])
        except Exception as e:
            if debug:
                print('[ERROR] Something wrong when generate_indi..!')
                print(e)
                print('indi rows : ' + str(len(bv)))
                print(bv.columns)
            return None


    def predict_this(self, indies):
        p = []
        p.append(self.jl_model1.predict(indies)[0])
        p.append(self.jl_model2.predict(indies)[0])
        p.append(self.jl_model3.predict(indies)[0])
        p.append(self.jl_model4.predict(indies)[0])
        p.append(self.jl_model5.predict(indies)[0])
        p.append(self.jl_model6.predict(indies)[0])
        p.append(self.jl_model7.predict(indies)[0])
        return p

    ## Disable dulu soalnya kebanyakan di pake di terminal juga
    #def pred_plotfile(self, pred,pair):
    #    '''
    #    prediction to plot
    #    hasil di simpan dalam bentuk file pnd biar gampang di kirim sama bot
    #    '''
    #    prefix_dir = '/dev/shm/'
    #    time_now = datetime.now().strftime("%d-%m%y-%H%M")
    #    plot_filename = prefix_dir + "/" + pair + "-" + time_now + ".png"
    #
    #    pred_df = pd.DataFrame(pred, columns=['Prediction'])
    #    plt.figure().set_figheight(1.5)
    #    plt.title(pair + ' - ' + time_now)
    #    plt.xlabel("Next Move")
    #    plt.plot(pred_df.index,pred_df['Prediction'],linewidth=3)
    #    plt.savefig(plot_filename)
    #
    #    return plot_filename


class PredCmd(Cmd):
    pred_main_class = None ## INI MUSTI DI SET SETIAP INHERITANCE..!!!
    timeframe = '1d' ##default timeframe
    default_fiat = 'USDT'
    exchange = 'BIN' # or 'IDX'
    model = 'default'
    prompt = ""
    preset_tfs = ['1w', '1d', '4h', '30m']  # timeframes
    preset_pairs = ['BTC', 'ETH', 'FET', 'FTM', 'HBAR', 'ZIL', 'SHIB']
    debug = False
    

    def __init__(self) -> None:
        self.prompt = "\n" + self.exchange + ':' + self.timeframe+ ':' + self.default_fiat +":> "
        super().__init__()

    def do_settf(self, args):
        'set timeframe [ 1w 3d 1d 12h 8h 6h 4h 2h 1h 30m 15m 5m ]\ntf 4h\n'
        if len(args) == 0:
            print(self.timeframe)
        else:
            self.timeframe = args
            self.prompt = "\n" + self.exchange + ':'  + self.timeframe+ ':' + self.default_fiat +":> "

    def do_settfs(self, args):
        'Set TimeFrame preset list (tfs)'
        if len(args) == 0:
            print(self.preset_tfs)
        else:
            self.preset_tfs = []
            for tf in args.split(" "):
                self.preset_tfs.append(tf)
            print(self.preset_tfs)

    def do_setpl(self, args):
        'set preset pair list'
        if len(args) == 0:
            print(self.preset_pairs)
        else:
            self.preset_pairs = []
            for pr in args.split(' '):
                self.preset_pairs.append(pr.upper())

    def do_setdf(self, args):
        'set default fiat (IDR / USDT). \nex: setdp USDT'
        if len(args) == 0:
            print(self.default_fiat)
        else:
            self.default_fiat = args.upper()
            if self.default_fiat == "IDR":
                self.exchange = 'IDX'
            self.prompt = "\n" + self.exchange + ':'  + self.timeframe+ ':' + self.default_fiat +":> "

    def do_setex(self, args):
        'set exchange [ bin | idx ]. \nex: setex idx'
        if len(args) == 0:
            print(self.exchange)
        else:
            self.exchange = args.upper()
            if self.exchange == 'BIN':
                self.default_fiat = 'USDT'
            elif self.exchange == 'IDX':
                self.default_fiat = 'IDR'
            self.prompt = "\n" + self.exchange + ':'  + self.timeframe+ ':' + self.default_fiat +":> "

    def do_model(self, args):
        'set model type'
        if len(args) == 0:
            print(self.model)
        else:
            if args in self.pred_main_class.available_model:
                self.pred_main_class.set_model_type(args)
                self.model = args
                self.pred_main_class.model = self.model
                self.pred_main_class.load_model()
            else:
                print('Available model : ')
                print(self.pred_main_class.available_model)

    def do_debug(self, args):
        'toggle ON/OFF Debug mode'
        if self.debug:
            self.debug = False
            print('Debug OFF')
        else:
            self.debug = True
            print('Debug ON')

    def do_exit(self, args):
        'Keluar'
        return True
    

class Holding():
    db = None
    query = None
    usd_idr = 15942  # 1 usd to idr rate

    def __init__(self,dbfile=None):
        if os.path.isfile('key'):
            kfile = open("key", "r")
            key = kfile.read()
            dbfile = 'db/myholding.json'
            self.db = TinyDB(encryption_key=key, path=dbfile, storage=tae.EncryptedJSONStorage)
            kfile.close()
            #db.storage.change_encryption_key("THE_NEW_KEY"))
        else:
            if dbfile is None:
                self.db = TinyDB('db/holding.json')
            else:
                self.db = TinyDB(dbfile)
        self.query = Query()

    def add_trans_buy(self, pair, amount, buy_price, exchange):
        self.db.insert({
            'pair': pair.upper(),
            'amount' : amount,
            'buy_price' : buy_price,
            'sell_price' : 0,
            'exchange' : exchange.upper(),
            })
    
    def add_trans_sell(self, pair, amount, sell_price, exchange):
        self.db.insert({
            'pair': pair.upper(),
            'amount' : amount,
            'buy_price' : 0,
            'sell_price' : sell_price,
            'exchange' : exchange.upper()
            })

    def remove_trans(self, doc_id):
        try:
            self.db.remove(doc_ids=[int(doc_id)])
            return True
        except:
            return False

    def ls_trans(self, pair=None, exchange=None):
        'list transaksi'
        holding = []
        if pair is None:
            rec = self.db.search(self.query.pair == pair)
        elif exchange is None:
            rec = self.db.search(self.query.exchange == exchange)
        else: #pair & exchange is None
            rec =  self.db.all()
                
        for h in self.db.all():
            holding.append({
            'doc_id' : h.doc_id,
            'pair': h['pair'],
            'amount': h['amount'],
            'buy_price': h['buy_price'],
            'sell_price': h['sell_price'],
            'exchange' : h['exchange']
            })
        return holding

    def value_now(self, default_fiat, pair=None, exchange=None ):
        pairs_value = []
        total_value = 0
        total_old_value = 0
        ticker = {}

        all_trans = self.ls_trans(pair=pair, exchange=exchange)
        holded = {}
        for trans in all_trans:
            dkey = trans['exchange'] + "-" + trans['pair']
            if dkey not in holded:
                if trans['buy_price'] is not None:
                    holded[dkey] = {'pair': trans['pair'], 
                                    'amount' : trans['amount'], 
                                    'avg_buy_price' : trans['buy_price'],
                                    'exchange': trans['exchange']
                                    }
            else:
                if trans['buy_price'] != 0 :
                    holded[dkey]['amount'] = holded[dkey]['amount'] + trans['amount'] 
                    holded[dkey]['avg_buy_price'] = ((holded[dkey]['avg_buy_price'] * holded[dkey]['amount']) + (trans['buy_price'] * trans['amount'])) / (holded[dkey]['amount'] + trans['amount']) 
                elif trans['sell_price'] != 0 :
                    holded[dkey]['amount'] = holded[dkey]['amount'] - trans['amount']
        
        for v in holded.values():
            if v['exchange'] not in ticker:
                ticker[v['exchange']] = get_ticker(v['exchange'])
            
            float_amount = float(v['amount'])
            if ticker[v['exchange']] is not None:
                try: #sapa tau ada pair yg ngasal input
                    if default_fiat == 'IDR' and 'USDT' in v['pair']:
                        coin_value = (float_amount * float(ticker[v['exchange']][v['pair']])) * self.usd_idr
                        old_value = (float_amount * float(v['avg_buy_price'])) * self.usd_idr
                    elif default_fiat == 'USDT' and 'IDR' in v['pair']:
                        coin_value = (float_amount * float(ticker[v['exchange']][v['pair']])) / self.usd_idr
                        old_value = (float_amount * float(v['avg_buy_price'])) / self.usd_idr
                    else:
                        coin_value = float_amount * float(ticker[v['exchange']][v['pair']])
                        old_value = (float_amount * float(v['avg_buy_price']))
                    percent_change = ((coin_value - old_value) / old_value) * 100
                    value_change = coin_value - old_value
                except:
                    coin_value = 0
                    percent_change = 0
                    value_change = 0
            else:
                coin_value = 0
                percent_change = 0
                value_change = 0

            if 'USDT' in v['pair']:
                coin_name = v['pair'].replace('USDT', '')
            elif 'IDR' in v['pair']:
                coin_name = v['pair'].replace('IDR', '')
            else:
                coin_name = v['pair']

            total_value += coin_value
            total_old_value += old_value
            pairs_value.append({#'pair': v['pair'],
                                'exchange' : v['exchange'],
                                'coin' : coin_name,
                                'amount' : format(int(float_amount),','),
                                #'buy_price' : v['avg_buy_price'],
                                'value_'+default_fiat: format(int(coin_value),','),
                                '%_ch' : round(percent_change, 1),
                                'val_ch' :  "{:,}".format(int(value_change),',') #'+' if value_change >= 0 else '-' +
                                })
        return {'total': total_value, 'total_change': total_value - total_old_value, 'detail' : pairs_value}

    def value_history(self, ohlc=False):
        pass

    def value_predict(self, hold_value_ohlc):
        pass


class MarketScan():
    ## INI MUSTI DI SET SETIAP INHERITANCE..!!!
    ## kenpa gak inherit aja class Pred? karena supaya dalam console semua setingnya class nya sama
    pred_main_class = None 

    exchange = "BIN"  
    default_fiat = "USDT"
    db = TinyDB('db/mscan.json', storage=serialization)
    debug = False
    tfs = ['1w','1d','4h','30m']

    def __init__(self):
        super().__init__()

    def scan(self,timeframe):
        'Scan market & hasilnya di simpan untuk analisa lebih lanjut.'
        pairs = get_pair_list(self.exchange, fiat=self.default_fiat, debug=self.debug)
        if pairs is None:
            if self.debug:
                print('scan : pairs is None')
            return None
        
        for pair in pairs:
            try:
                klines = get_klines(self.exchange, pair, timeframe=timeframe, candle_to_fetch=61, debug=self.debug)
                indies = self.pred_main_class.generate_indi(klines)
                indi = indies.iloc[[-1]]
                p = self.pred_main_class.predict_this(indi)
                self.db.insert({'datetime': datetime.now(), 
                                'exchange': self.exchange, 
                                'pair' : pair,
                                'timeframe' : timeframe,
                                'p0' : indi['TYP_MOV'].values[0],
                                'p1' : p[0],
                                'p2' : p[1],
                                'p3' : p[2],
                                'p4' : p[3],
                                'p5' : p[4],
                                'p6' : p[5],
                                'p7' : p[6],
                                })
            except:
                if self.debug:
                    print(f'[ERROR SCAN] {self.exchange} {pair} {timeframe}')
                pass
    
    def clean_expired_data(self):
        pass

    def bull_potential(self):
        'Return list of potential bullish pair'
        pass


if __name__ == '__main__':
    print("Pakai PredCmd aja buat cli interface nya.")