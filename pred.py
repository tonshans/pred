#import sys
import os.path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from datetime import datetime
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
#import matplotlib.pyplot as plt
from cmd import Cmd 
from tinydb import TinyDB, Query
import tinydb_encrypted_jsonstorage as tae

try:
    from binance.client import Client
    bin_client = Client('', '')
    connect_to_binance = True
except:
    connect_to_binance = False



def get_single_ticker(exchange, pair):
    if exchange == 'idx':
        resp = requests.get("https://indodax.com/api/ticker/" + pair.upper())
        resp.json()['ticker']
        _responnya_begini = """
        {'high': '115',
        'low': '80',
        'vol_tel': '12207711.89945255',
        'vol_idr': '1217518468',
        'last': '108',
        'buy': '108',
        'sell': '109',
        'server_time': 1733213597}
        """


class Pred():
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


    def generate_indi(self, klines):
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
            print('Error di generate_indi')
            print(e)
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


    def pred_plotfile(self, pred,pair):
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


class PredCmd(Cmd):
    caller_id = ''
    pred_main_class = None ## ini musti di set setiap inheritance
    timeframe = '1d' ##default timeframe
    default_pair = 'USDT'
    model = 'default'
    prompt = ""
    preset_pairs = ['BTC', 'ETH', 'FET', 'FTM', 'HBAR', 'ZIL', 'LIT']
    

    def __init__(self) -> None:
        self.prompt = "\n" + self.caller_id + ':' + self.timeframe+ ':' + self.default_pair +":> "
        super().__init__()

    def do_settf(self, args):
        'set timeframe [ 1w 3d 1d 12h 8h 6h 4h 2h 1h 30m 15m 5m ]\ntf 4h\n'
        if len(args) == 0:
            print(self.timeframe)
        else:
            self.timeframe = args
            self.prompt = "\n" + self.caller_id + ':'  + self.timeframe+ ':' + self.default_pair +":> "

    def do_setpl(self, args):
        'set preset pair list'
        if len(args) == 0:
            print(self.preset_pairs)
        else:
            self.preset_pairs = []
            for pr in args.split(' '):
                self.preset_pairs.append(pr.upper())

    def do_setdp(self, args):
        'set default pair. \nex: setdp USDT'
        if len(args) == 0:
            print(self.default_pair)
        else:
            self.default_pair = args.upper()
            self.prompt = "\n" + self.caller_id + ':'  + self.timeframe+ ':' + self.default_pair +":> "

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

    

    def do_exit(self, args):
        'Keluar'
        return True
    

class holding():
    db = None
    query = None

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

    def add(self, pair, amount, exchange):
        self.db.insert({
        'pair': pair.upper(),
        'amount' : amount,
        'exchange' : exchange.upper(),
        })

    def remove(self, doc_id):
        try:
            self.db.remove(doc_id=doc_id)
            return True
        except:
            return False

    def ls(self, pair=None, exchange=None):
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
            'exchange' : h['exchange']
            })
        return holding

    def value(self, pair=None, exchange=None ):
        pairs_value = []
        total_value = 0
        exchange_list = []
        ticker_list = []
        holded = self.ls(pair=pair, exchange=exchange)
        
        ## ngelist exchange nya dulu
        for h in holded:
            if h['exchange'] not in exchange_list:
                exchange_list.append(h['exchange'])
                ticker_list.append({'exchange' : h['exchange'], 'ticker' : get_ticker(h['exchange'])})

        return {'total': total_value, 'detail' : pairs_value}

    def value_history(self, ohlc=False):
        pass

    def value_predict(self, hold_value_ohlc):
        pass