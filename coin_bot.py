import json
from urllib.request import Request, urlopen
import time
import telepot
from operator import eq

class korbit:
    reqBTC = Request('https://api.korbit.co.kr/v1/ticker?currency_pair=btc_krw' , headers={'User-Agent': 'Mozilla/5.0'})
    readBTC = urlopen(reqBTC).read()
    jsonBTC = json.loads(readBTC)
    FindBTC = jsonBTC['last']
    BTC = int(FindBTC)
    reqETH = Request('https://api.korbit.co.kr/v1/ticker?currency_pair=eth_krw' , headers={'User-Agent': 'Mozilla/5.0'})
    readETH = urlopen(reqETH).read()
    jsonETH = json.loads(readETH)
    FindETH = jsonETH['last']
    ETH = int(FindETH)
    reqETC = Request('https://api.korbit.co.kr/v1/ticker?currency_pair=etc_krw' , headers={'User-Agent': 'Mozilla/5.0'})
    readETC = urlopen(reqETC).read()
    jsonETC = json.loads(readETC)
    FindETC = jsonETC['last']
    ETC = int(FindETC)
    reqXRP = Request('https://api.korbit.co.kr/v1/ticker?currency_pair=xrp_krw' , headers={'User-Agent': 'Mozilla/5.0'})
    readXRP = urlopen(reqXRP).read()
    jsonXRP = json.loads(readXRP)
    FindXRP = jsonXRP['last']
    XRP = int(FindXRP)

class tele_bot(object):
    def __init__(self,TOKEN):
        self.bot = telepot.Bot(TOKEN)
        self.bot.message_loop(self.handle)

    def handle(self,msg):
        content_type, chat_type, chat_id = telepot.glance(msg)
        print(content_type, chat_type, chat_id)

        if content_type == 'text':
            # self.MSGQ.put(msg['text'])
            if eq(msg['text'],'/BTC'):
                self.bot.sendMessage(chat_id, "1 BTC(KRW)" + str(korbit.BTC))
            elif eq(msg['text'],'/ETH'):
                self.bot.sendMessage(chat_id, "1 ETH(KRW)" + str(korbit.ETH))
            elif eq(msg['text'], '/ETC'):
                self.bot.sendMessage(chat_id, "1 ETC(KRW)" + str(korbit.ETC))
            elif eq(msg['text'], '/XRP'):
                self.bot.sendMessage(chat_id, "1 XRP(KRW)" + str(korbit.XRP))
            elif eq(msg['text'], '/HELP') or eq(msg['text'], '/help'):
                self.bot.sendMessage(chat_id,"please send /BTC or /ETH or /ETC or /XRP")

if __name__ == "__main__":

    TOKEN = '텔레그램 @BotFather에게 받으셔서 이곳에 넣어주세요'
    chat = tele_bot(TOKEN)
    print('Listening ...')

    # Keep the program running.
    while 1:
        time.sleep(10)
#