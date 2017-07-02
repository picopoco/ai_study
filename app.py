#!/usr/bin/env python

from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request

import json
import os

from flask import Flask
from flask import request
from flask import make_response
from operator import eq

# Flask app should start in global layout
app = Flask(__name__)


@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)

    print("Request:")
    print(json.dumps(req, indent=4))

    res = processRequest(req)

    res = json.dumps(res, indent=4)
    # print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


def processRequest(req):

    if req.get("result").get("action") == "korbitCoinCheck":

        result = req.get("result").get("parameters")
        query = result.get('currency-name')

        if eq(query, "BTC") or eq(query, "btc"):
            speech = query + 'is ' + str(korbit.BTC) + '(KRW)'
        elif eq(query, "ETH") or eq(query, "eth"):
            speech = query + 'is ' + str(korbit.ETH) + '(KRW)'
        elif eq(query, "ETC") or eq(query, "etc"):
            speech = query + 'is ' + str(korbit.ETC) + '(KRW)'
        elif eq(query, "XRP") or eq(query, "xrp"):
            speech = query + ' is ' + str(korbit.XRP) + '(KRW)'
        else:
            speech = query + 'is unknown'

        speech = speech
        return {
            "speech": speech,
            "displayText": speech,
            # "data": data,
            # "contextOut": [],
            "source": "coin-webhook"
        }
    else:
        return {}

    return res


class korbit:
    reqBTC = Request('https://api.korbit.co.kr/v1/ticker?currency_pair=btc_krw' , headers={'User-Agent': 'Mozilla/5.0'})
    readBTC = urlopen(reqBTC).read().decode('utf-8')
    jsonBTC = json.loads(readBTC)
    FindBTC = jsonBTC['last']
    BTC = int(FindBTC)
    reqETH = Request('https://api.korbit.co.kr/v1/ticker?currency_pair=eth_krw' , headers={'User-Agent': 'Mozilla/5.0'})
    readETH = urlopen(reqETH).read().decode('utf-8')
    jsonETH = json.loads(readETH)
    FindETH = jsonETH['last']
    ETH = int(FindETH)
    reqETC = Request('https://api.korbit.co.kr/v1/ticker?currency_pair=etc_krw' , headers={'User-Agent': 'Mozilla/5.0'})
    readETC = urlopen(reqETC).read().decode('utf-8')
    jsonETC = json.loads(readETC)
    FindETC = jsonETC['last']
    ETC = int(FindETC)
    reqXRP = Request('https://api.korbit.co.kr/v1/ticker?currency_pair=xrp_krw' , headers={'User-Agent': 'Mozilla/5.0'})
    readXRP = urlopen(reqXRP).read().decode('utf-8')
    jsonXRP = json.loads(readXRP)
    FindXRP = jsonXRP['last']
    XRP = int(FindXRP)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))

    print("Starting app on port %d" % port)

    app.run(debug=False, port=port, host='0.0.0.0')
