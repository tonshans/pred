from datetime import datetime
from tabulate import tabulate
from pred import Pred, PredCmd, get_klines, Holding
pred = Pred()
hold = Holding()
    
def predict(pair,timeframe='1d',exchange='IDX',debug=False):
    if exchange == 'BIN' and 'IDR' in pair:
        print(f'{pair} Fiat Not supported')
        return None
    
    try:
        klines = get_klines(exchange, pair,timeframe=timeframe,debug=debug)

        if debug:
            print(f"predict input : {pair}, {timeframe},{ exchange}, {debug}")
            print(f"predict klines output : {klines}")
        if klines is None:
            return None
        ctime = klines.iloc[-1]['Ctime']
        indi = pred.generate_indi(klines,debug=debug).iloc[[-1]]
        return {
            'ctime' : ctime,
            'pred': [indi['TYP_MOV'].values[0]] + pred.predict_this(indi)
        }
    except Exception as e:
        print('[ERROR] Predict say SORRY...')
        print(e)
        return None

def print_predict(pair,timeframe='1d', exchange='IDX', debug=False):
    if exchange == 'BIN' and 'IDR' in pair:
        print(f'{pair} Fiat Not supported')
        return None
    
    pred = predict(pair,timeframe=timeframe,exchange=exchange,debug=debug)
    if debug:
        print(f"print_predict input : {pair}, {timeframe}, {exchange}, {debug}")
        print(f'print_predict pred output : {pred}')
    if pred is None:
        return None
    #time_now = datetime.now().strftime("%d-%m %H:%M")
    time_now = pred['ctime'].strftime("%d-%m %H:%M")
    
    if 'USDT' in pair:
        pair_len = 8
    else:
        pair_len = 7

    if (len(pair) < pair_len) :
        pred_str = '  ' +  pair + '--' + time_now + '-> '
    else:
        pred_str = '  ' +  pair + '-' + time_now + '-> '
    c = 0
    for p in pred['pred']:
        if p == 1:
            pred_str += '\033[1;32;40m [' + str(c)+']'
        else:
            pred_str += '\033[1;31;40m (' + str(c)+')'
        c +=1
    print(pred_str + '\033[0m')

def print_predict_tf_overlap(pair, tfs=['1w', '1d', '4h', '30m'],exchange='IDX',debug=False):
    if exchange == 'BIN' and 'IDR' in pair:
        print(f'{pair} Fiat Not supported')
        return None
    
    print('  _' + pair + '_' )
    for tf in tfs:
        pred = predict(pair,timeframe=tf,exchange=exchange, debug=debug)
        if pred is None:
            return None
        ctime = pred['ctime'].strftime("%d-%m %H:%M")
        if len(tf)<3:
            pred_str = '  ' + tf + ' :'+ ctime +'-> '
        else:
            pred_str = '  ' + tf + ':'+ ctime +'-> '
        c = 0
        for p in pred['pred']:
            if p == 1:
                pred_str += '\033[1;32;40m [' + str(c)+']'
            else:
                pred_str += '\033[1;31;40m (' + str(c)+')'
            c +=1
        print(pred_str + '\033[0m')

def print_holded_value(default_fiat):
    h = hold.value_now(default_fiat)
    header = h['detail'][0].keys()
    rows = [x.values() for x in h['detail']]
    print(f"\nTotal : {round(h['total'],2):,} {default_fiat}  ({round(h['total_change'],2):,} change)\n")
    print(tabulate(rows, header))
    

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
    def __init__(self):
        self.pred_main_class = pred
        self.exchange = 'BIN'
        self.default_fiat = 'USDT'
        
        self.preset_pairs = ['BTC', 'ETH', 'FET', 'FTM', 'HBAR', 'ZIL', 'SHIB', 'TEL']
        super().__init__()

    def do_p(self, args):
        '''
        Predict\n
        argument bisa di isi banyak pair pakai space separator ya
        '''
        arg_split = args.split(' ')
        for arg in arg_split:
            print_predict(arg.upper() + self.default_fiat, timeframe=self.timeframe, exchange=self.exchange, debug=self.debug)

    def do_pl(self, args):
        '''
        Predict List\n
        Predict Pair dalam List preset\n
        list bisa di ubah dengan command setpr
        '''
        arg = args.split(' ')
        if len(arg) > 0  and arg[0] != '' :
            pairs = arg
        else:
            pairs = self.preset_pairs
        for pair in pairs:
            print_predict(pair + self.default_fiat, timeframe=self.timeframe, exchange=self.exchange, debug=self.debug)

    def do_pltf(self, args):
        '''
        Predict : List -> TimeFrame \n
        Predict Pair dalam List preset di loop dalam list timeframe\n
        list bisa di ubah dengan command setpr, atau di set setelah command ini.
        pltf : akan menggunakan list dari preset
        pltf BTCUSDT BNBUSDT ETHUSDT : akan menggunakan list dari command argument
        '''
        for pair in self.preset_pairs:
            print_predict_tf_overlap(pair.upper() + self.default_fiat, tfs = self.preset_tfs, exchange=self.exchange, debug=self.debug)
        

    def do_ptf(self, args):
        '''
        Predict TimeFrame
        Predict multi timeframe overlap, nge-predict 1 pair dalam preset timeframe\n
        ptf [.. pair ..]
        '''
        arg_split = args.split(' ')
        for arg in arg_split:
            print_predict_tf_overlap(arg.upper() + self.default_fiat, tfs = self.preset_tfs, exchange=self.exchange, debug=self.debug)

    def do_ptfl(self, args):
        '''
        Predict : TimeFrame -> List
        Predict multi timeframe overlap, jika tanpa argumen menggunakan PAIR dari list preset_pair, 
        jika dengan argumen maka list dalam argumen akan digunakan sebagai input list timeframe
        ptfl : tanpa argumen maka akan menggunakan list timeframe preset
        ptfl 4h 1h 15m  : akan menggunakan timeframe dari list
        '''
        arg = args.split(' ')
        if len(arg) > 0 and arg[0] != '' :
            tfs = arg
        else:
            tfs = self.preset_tfs

        for tf in tfs:
            print(tf)
            for pair in self.preset_pairs:
                print_predict(pair + self.default_fiat, timeframe=tf, exchange=self.exchange, debug=self.debug)

    def do_hval(self, args):
        '''HOLD Value,\ncek nilai koin yang sedang di hold.'''
        print_holded_value(self.default_fiat)

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
        
    def do_tel(self, args):
        '''
        Special Case
        untuk cek status tel di idx, supaya gak pindah2 mode exchange cuma buat cek tel aja
        '''
        pair = 'TELIDR'

        print_predict_tf_overlap(pair, tfs=['1w', '1d', '4h', '30m'],exchange='idx',debug=self.debug)


#######################################
if __name__ == '__main__':
    app = idxCmd()
    app.cmdloop('Pred - a Realistic Statistical Hopium for Holders\ntype help for command list:')
