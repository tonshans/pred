import requests
from datetime import datetime,timedelta
import re
import pandas as pd
from cmd import Cmd 

from pred_novol import generate_indi, predict_this


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
    predict_this(generate_indi(get_klines(pair).iloc[[-1]]))
    '''
    klines = get_klines(pair,timeframe=timeframe)
    if klines is None:
        return None
    indi = generate_indi(klines).iloc[[-1]]
    return [indi['TYP_MOV'].values[0]] + predict_this(indi)

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
    default_pair = 'IDR'
    prompt = "\nIDX:" + timeframe+ ':' + default_pair +":> "


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
        'preset list, monitor tel aja dulu disini'
        preset_pair = 'TEL'
        print_predict_tf_overlap(preset_pair)

    def do_tf(self, args):
        'timeframe [ 1w 3d 1d 12h 8h 6h 4h 2h 1h 30m 15m 5m ]\ntf 4h\n'
        if len(args) == 0:
            print(self.timeframe)
        else:
            self.timeframe = args
            self.prompt = "\nIDX:" + self.timeframe+ ':' + self.default_pair +":> "

    def do_dpair(self, args):
        'set default pair. \nex: dpair USDT'
        self.default_pair = args.upper()
        self.prompt = "\nIDX:" + self.timeframe+ ':' + self.default_pair +":> "

    def do_exit(self, args):
        'Keluar'
        return True
#######################################
if __name__ == '__main__':
    app = predCmd()
    app.cmdloop('IDX Pred\nEnter a command to predict trend movement \n[p/ptf/pr/tf/help]:')