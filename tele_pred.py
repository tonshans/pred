from pred import predict, pred_plotfile
from datetime import datetime
import os

from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters



## Enable logging -------------------------------------
import logging
logging.basicConfig( format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
## set higher logging level for httpx to avoid all GET and POST requests being logged
#logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
##-----------------------------------------------------

def pred_to_html(pred):
    html_pred = '<i>' + datetime.now().strftime("%d-%m-%Y %H:%M") + '</i> :: '

    c = 0
    for p in pred:
        if p == 1:
            html_pred += '<b>'+str(c)+'</b> '
        else:
            html_pred += '<s>'+str(c)+'</s> '
        c +=1
    return html_pred


def check_msg_sender(user):
    #User(first_name='Th', id=152042850, is_bot=False, language_code='en', username='tonhan')
    if user['id'] == 152042850:
        return True
    else:
        return False


def process_text(text, mode=None):
    default_tf = '1d'
    ts = text.split(' ')
    len_ts = len(ts) 

    if text.upper().startswith('P '):
        if len_ts < 1:
            return "Zonk..!"
        c_start = 1 ## mulai fetch pair mulai dari
        tf = default_tf
    else:
        ## kalo text di mulai dari PTF
        if len_ts < 2:
            return "Zonk..!"
        c_start = 1 ## mulai fetch pair mulai dari
        tf = ts[1]

    c = 0
    text_html = ''

    for t in ts:
        if c >= c_start :
            pair = str(t).upper()
            pred = predict(pair,timeframe=tf)
            if pred is not None:
                text_html += '<b>[' + tf + '-'+ pair + ']</b> ' + pred_to_html(pred)
                if len_ts > 2:
                    text_html += '\n'
        c+=1

    if text_html == '':
        return 'Sorry..'
    return text_html



async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Echo the user message.
    pake ini aja untuk nge-filter mesg jadi command
    biar gak perlu pake garismiring tiap kasih command
    """
    if check_msg_sender(update.effective_user):
        msg = update.message.text
        if msg.upper().startswith('P ') or msg.upper().startswith('PTF '):
            text_html = process_text(msg)
            await update.message.reply_html(text_html)
        elif msg.upper().startswith('PP '):
            ## Prediction Plot
            ts = msg.split(' ')
            if len(ts) < 1:
                await update.message.reply_text("Zonkk..!!")
            else:
                pair = ts[1].upper()
                pred = predict(pair)
                if pred is None: 
                    await update.message.reply_text("Sorry..!!")
                else:
                    plot_filename = pred_plotfile(pred,pair)
                    await update.message.reply_photo(plot_filename)
                    os.remove(plot_filename)
        elif msg.upper() == 'B':
            text_html = '<b>Batch check: </b> \n' + process_text('P BTCUSDT FETUSDT FTMUSDT HBARUSDT ZILUSDT LITUSDT')
            await update.message.reply_html(text_html)
        else:
            await update.message.reply_text(msg)


##-------------------------------------------------------
def main() -> None:
    """Start the bot."""
    application = Application.builder().token("243459966:AAG1QrLd0QJ7fGcgRWnY0jhn1BAsyEVUgYM").build()

    #application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    ## Run the bot until the user presses Ctrl-C
    print('OK sudah running...')
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
