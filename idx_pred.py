from datetime import datetime
from pred import Pred, PredCmd, get_klines, Holding
pred = Pred()
hold = Holding()
    
def predict(pair,timeframe='1d',exchange='idx'):
    try:
        klines = get_klines(exchange, pair,timeframe=timeframe)
        if klines is None:
            return None
        indi = pred.generate_indi(klines).iloc[[-1]]
        return [indi['TYP_MOV'].values[0]] + pred.predict_this(indi)
    except:
        print('Predict say SORRY...')
        return None

def print_predict(pair,timeframe='1d',default_pair='USDT',exchange='idx'):
    pred = predict(pair,timeframe=timeframe,exchange=exchange)
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

def print_predict_tf_overlap(pair, tfs=['1w', '1d', '4h', '30m'],exchange='idx'):
    time_now = datetime.now().strftime("%d-%m %H:%M")
    print('  _' + pair + '_' + time_now + ':')
    for tf in tfs:
        pred = predict(pair,timeframe=tf,exchange=exchange)
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

def print_holded_value(default_pair):
    h = hold.value_now()
    print(f"Total : {h['total']:,} {default_pair}")
    for cval in h['detail']:
        print(f"{cval['exchange']} : {cval['amount']:,} {cval['pair']} @ {cval['value']:,} {default_pair}")

def add_trans(exchange, trans_type, pair, amount, price):
    if trans_type == 'buy':
        hold.add_trans_buy(pair, amount, price, exchange)
    elif trans_type == 'sell':
        hold.add_trans_sell(pair, amount, price, exchange)

def remove_trans(doc_id):
    hold.remove_trans(doc_id)

def print_transaction():
    for t in hold.ls_trans():
        if t['buy_price'] != 0:
            print('[' + str(t['doc_id']) + '] ' + t['exchange'] + ' BUY ' + str(t['amount']) + ' ' + t['pair'] + ' @ ' + str(t['buy_price']) )
        elif t['sell_price'] != 0:
            print('[' + str(t['doc_id']) + '] ' + t['exchange'] + ' SELL ' + str(t['amount']) + ' ' + t['pair'] + ' @ ' + str(t['sell_price']) )


#######################################
class idxCmd(PredCmd):
    def __init__(self) -> None:
        self.pred_main_class = pred
        self.exchange = 'idx'
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
            print_predict(arg.upper() + self.default_pair, timeframe=self.timeframe, default_pair=self.default_pair, exchange=self.exchange)

    def do_pl(self, args):
        '''
        Predict List\n
        Predict Pair dalam List preset\n
        list bisa di ubah dengan command setpr
        '''
        for pair in self.preset_pairs:
            print_predict(pair + self.default_pair, self.timeframe,exchange=self.exchange)

    def do_pltf(self, args):
        '''
        Predict : List -> TimeFrame \n
        Predict Pair dalam List preset di loop dalam list timeframe\n
        list bisa di ubah dengan command setpr
        '''
        for pair in self.preset_pairs:
            print_predict_tf_overlap(pair.upper() + self.default_pair, tfs = self.preset_tfs, exchange=self.exchange)
        

    def do_ptf(self, args):
        '''
        Predict TimeFrame
        Predict multi timeframe overlap, nge-predict 1 pair dalam preset timeframe\n
        ptf [.. pair ..]
        '''
        arg_split = args.split(' ')
        for arg in arg_split:
            print_predict_tf_overlap(arg.upper() + self.default_pair, tfs = self.preset_tfs, exchange=self.exchange)

    def do_ptfl(self, args):
        '''
        Predict : TimeFrame -> List
        Predict multi timeframe overlap, nge-predict 1 pair dalam preset timeframe\n
        menggunakan PAIR dari list preset_pair 
        '''
        for tf in self.preset_tfs:
            print(tf)
            for pair in self.preset_pairs:
                print_predict(pair + self.default_pair, tf, exchange=self.exchange)

    def do_hval(self, args):
        '''HOLD Value,\ncek nilai koin yang sedang di hold.'''
        print_holded_value(self.default_pair)

    def do_hls(self, args):
        'Hold List transaction'
        print_transaction()

    def do_hbuy(self, args):
        '''Hold buy transaction, 
        Add item yang di hold\nhadd pair amount buy_price exchange
        '''
        arg =  args.split(" ")
        if len(arg) < 4:
            print('Add item yang di hold\nhbuy _pair_ _amount_ _buy_price_ _exchange_')
        else:
            pair = arg[0].upper()
            amount = arg[1]
            price = arg[2]
            exchange = arg[3].upper()
            #print("%s %s %s %s " %(exchange, pair, amount, buy_price))
            add_trans(exchange, 'buy', pair, amount, price)

    def do_hsell(self, args):
        '''Hold Sell transaction, 
        Add item yang di hold\nhsell pair amount price exchange
        '''
        arg =  args.split(" ")
        if len(arg) < 4:
            print('Add item yang di hold\nhadd _pair_ _amount_ _buy_price_ _exchange_')
        else:
            pair = arg[0].upper()
            amount = arg[1]
            price = arg[2]
            exchange = arg[3].upper()
            #print("%s %s %s %s " %(exchange, pair, amount, buy_price))
            add_trans(exchange, 'buy', pair, amount, price)
    
    def do_hrm(self, args):
        '''Hold Remove transaction\ncek doc_id dari command hls\nex. hrm _doc_id_
        '''
        if remove_trans(args):
            print('Transaction record ' + args + ' removed')
        else:
            print('Remove Transaction record ' + args + ' FAILED.')
        

#######################################
if __name__ == '__main__':
    app = idxCmd()
    app.cmdloop('Pred\nEnter a command to predict trend movement \n[p/pl/ptf/ptfl/tf/help]:')
