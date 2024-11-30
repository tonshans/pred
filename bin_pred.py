import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from datetime import datetime, timedelta
import pandas as pd
from binance.client import Client
import re
import requests

from pred import Pred, PredCmd
pred = Pred()


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

## sementara biarkan aja dulu func ini disini
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

def idx_predict(pair,timeframe='1d'):
    klines = get_idx_klines(pair,timeframe=timeframe)
    if klines is None:
        return None
    indi = pred.generate_indi(klines).iloc[[-1]]
    return [indi['TYP_MOV'].values[0]] + pred.predict_this(indi)

def predict(pair,timeframe='1d'):
    '''
    cuma wrapper biar enak manggilnya
    pred.predict_this(pred.generate_indi(get_klines(pair).iloc[[-1]]))
    '''
    try:
        klines = get_klines(pair,timeframe=timeframe)
        if klines is None:
            return None
        indi = pred.generate_indi(klines).iloc[[-1]]
        #return [indi['MOV'].values[0]] + pred.predict_this(pred.generate_indi(klines).iloc[[-1]])
        return [indi['TYP_MOV'].values[0]] + pred.predict_this(indi)
    except:
        print('Predict say SORRY...')
        return None

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
class binCmd(PredCmd):
    def __init__(self) -> None:
        self.pred_main_class = pred
        super().__init__()

    def do_tel(self, args):
        '''
        Special Case
        untuk cek status tel di idx, supaya gak pindah2 mode exchange cuma buat cek tel aja
        '''
        tfs=['1d', '4h', '30m']
        pair = 'TELIDR'

        time_now = datetime.now().strftime("%d-%m %H:%M")
        print('  _IDX_' + pair + '_' + time_now + ':')
        for tf in tfs:
            pred_result = idx_predict(pair,timeframe=tf)
            if pred_result is None:
                return None
            if len(tf)<3:
                pred_str = '  ' + tf + ' :-> '
            else:
                pred_str = '  ' + tf + ':-> '
            c = 0
            for p in pred_result:
                if p == 1:
                    pred_str += '\033[1;32;40m [' + str(c)+']'
                else:
                    pred_str += '\033[1;31;40m (' + str(c)+')'
                c +=1
            print(pred_str + '\033[0m')
            #sleep(2) ##biar gak error2 dia waktu query api nya

    def do_p(self, args):
        '''
        Predict\n
        argument bisa di isi banyak pair pakai space separator ya
        '''
        arg_split = args.split(' ')
        for arg in arg_split:
            print_predict(arg.upper() + self.default_pair, timeframe=self.timeframe, default_pair=self.default_pair)

    def do_pl(self, args):
        '''
        Predict List\n
        Predict Pair dalam List preset\n
        list bisa di ubah dengan command setpr
        '''
        for pair in self.preset_pairs:
            print_predict(pair + self.default_pair, self.timeframe)

        ## special case, ###############################################
        pair = 'TELIDR'
        pred_result = idx_predict(pair,timeframe=self.timeframe)
        time_now = datetime.now().strftime("%d-%m %H:%M")
        if pred_result is not None:
            time_now = datetime.now().strftime("%d-%m %H:%M")
            pred_str = '  ' +  pair + '---' + time_now + '-> '
            
            c = 0
            for p in pred_result:
                if p == 1:
                    pred_str += '\033[1;32;40m [' + str(c)+']'
                else:
                    pred_str += '\033[1;31;40m (' + str(c)+')'
                c +=1
            print(pred_str + '\033[0m')

    def do_ptf(self, args):
        '''
        Predict TimeFrame
        Predict multi timeframe overlap, nge-predict 1 pair dalam preset timeframe\n
        ptf [.. pair ..]
        '''
        arg_split = args.split(' ')
        for arg in arg_split:
            print_predict_tf_overlap(arg.upper() + self.default_pair) 

    def do_ptfl(self,args):
        '''
        Predict TimeFrame List
        Predict multi timeframe overlap, nge-predict 1 pair dalam preset timeframe\n
        menggunakan PAIR dari list preset_pair 
        '''
        for pair in self.preset_pairs:
            print_predict_tf_overlap(pair.upper() + self.default_pair) 
        self.do_tel(None)


#######################################
if __name__ == '__main__':
    client = Client('', '')

    app = binCmd()
    app.cmdloop('Enter a command to predict trend movement \n[p/pl/ptf/ptfl/tf/tel/help]:')
