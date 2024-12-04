from datetime import datetime
from pred import Pred, PredCmd, get_klines, Holding
pred = Pred()
hold = Holding()
    
def predict(pair,timeframe='1d'):
    try:
        klines = get_klines('idx', pair,timeframe=timeframe)
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
    if (len(pair) < 7) :
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

def print_holded_value():
    h = hold.value_now()
    print('Total : ' + str(h['total']))
    for cval in h['detail']:
        print(cval['exchange'] + ":" + cval['pair'] + " : " + str(cval['amount']) + " : " + str(cval['value']))

def add_holded_coin(exchange, pair, amount, buy_price):
    hold.add(pair, amount, buy_price, exchange)

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

    def do_pltf(self, args):
        '''
        Predict List per Time Frame\n
        Predict Pair dalam List preset di loop dalam list timeframe\n
        list bisa di ubah dengan command setpr
        '''
        for tf in self.preset_tfs:
            print(tf)
            for pair in self.preset_pairs:
                print_predict(pair + self.default_pair, tf)

    def do_ptf(self, args):
        '''
        Predict TimeFrame
        Predict multi timeframe overlap, nge-predict 1 pair dalam preset timeframe\n
        ptf [.. pair ..]
        '''
        arg_split = args.split(' ')
        for arg in arg_split:
            print_predict_tf_overlap(arg.upper() + self.default_pair,tfs = self.preset_tfs)

    def do_ptfl(self, args):
        '''
        Predict TimeFrame List
        Predict multi timeframe overlap, nge-predict 1 pair dalam preset timeframe\n
        menggunakan PAIR dari list preset_pair 
        '''
        for pair in self.preset_pairs:
            print_predict_tf_overlap(pair.upper() + self.default_pair)

    def do_h(self, args):
        '''HOLD,\ncek nilai koin yang sedang di hold.'''
        print_holded_value()

    

#######################################
if __name__ == '__main__':
    app = idxCmd()
    app.cmdloop('IDX Pred\nEnter a command to predict trend movement \n[p/pl/ptf/ptfl/tf/help]:')
