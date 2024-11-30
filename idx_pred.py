import requests
from datetime import datetime,timedelta
import re
import pandas as pd

from pred import Pred, PredCmd
pred = Pred()

def get_klines(pair,timeframe='1d',candle_to_fetch=61):
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
class idxCmd(PredCmd):
    def __init__(self) -> None:
        self.pred_main_class = pred
        self.caller_id = 'idx'
        self.default_pair = 'IDR'
        self.preset_pairs = ['BTC', 'ETH', 'FET', 'FTM', 'HBAR', 'ZIL', 'LIT', 'TEL']
        super().__init__()

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

    def do_ptf(self, args):
        '''
        Predict TimeFrame
        Predict multi timeframe overlap, nge-predict 1 pair dalam preset timeframe\n
        ptf [.. pair ..]
        '''
        arg_split = args.split(' ')
        for arg in arg_split:
            print_predict_tf_overlap(arg.upper() + self.default_pair)

    def do_ptfl(self, args):
        '''
        Predict TimeFrame List
        Predict multi timeframe overlap, nge-predict 1 pair dalam preset timeframe\n
        menggunakan PAIR dari list preset_pair 
        '''
        for pair in self.preset_pairs:
            print_predict_tf_overlap(pair.upper() + self.default_pair)

    

#######################################
if __name__ == '__main__':
    app = idxCmd()
    app.cmdloop('IDX Pred\nEnter a command to predict trend movement \n[p/pl/ptf/ptfl/tf/help]:')
