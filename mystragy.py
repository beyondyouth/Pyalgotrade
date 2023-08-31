import datetime
import json
import time

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

from pyalgotrade.stratanalyzer.drawdown import DrawDown

'''
@Author: dong.zhili
@Date: 1970-01-01 08:00:00
@LastEditors: dong.zhili
@LastEditTime: 2020-05-25 12:51:53
@Description: 
'''
import datetime
import os
from pyalgotrade import strategy, broker, plotter
from pyalgotrade.barfeed import quandlfeed
from pyalgotrade.stratanalyzer import returns, sharpe, drawdown, trades
from pyalgotrade.technical import ma, macd, rsi, stoch, bollinger, aroon
from pyalgotrade.technical import cross, highlow, atr
from pyalgotrade.talibext import indicator

class MyStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, bBandsPeriod):
        super(MyStrategy, self).__init__(feed)
        self.__instrument = instrument
        # 使用调整后的数据
        if feed.barsHaveAdjClose():
            self.setUseAdjustedValues(True)
        
        # 统计收盘价
        self.__price = feed[instrument].getPriceDataSeries()
        # 计算macd指标
        self.__macd = macd.MACD(self.__price, 12, 26, 9)
        # 计算KD指标
        self.__stoch = stoch.StochasticOscillator(feed[instrument], 9, 3)
        # 计算rsi指标
        self.__rsi7 = rsi.RSI(self.__price, 7)
        self.__rsi23 = rsi.RSI(self.__price, 23)
        # 计算布林线
        self.__bbands = bollinger.BollingerBands(self.__price, bBandsPeriod, 2)

        self.__maxN = highlow.High(self.__price, 20)
        self.__minN = highlow.Low(self.__price, 26)

        self.__ema5 = ma.EMA(self.__price, 21)
        self.__sma5 = ma.SMA(self.__price, 5)
        self.__sma21 = ma.SMA(self.__price, 16)

        self.__atr = atr.ATR(feed[instrument], 21)
        self.__dict = {}

        self.setDebugMode(False)
        self.__date = self.getCurrentDateTime()
        self.__realDate = datetime.datetime.now().strftime("%Y-%m-%d 00:00:00")

        self.__cash1 = self.getBroker().getCash() // 5
        self.__share1 = 0

        self.__cash2 = self.getBroker().getCash() // 5
        self.__share2 = 0

        self.__cash3 = self.getBroker().getCash() // 5
        self.__share3 = 0

        self.__cash4 = self.getBroker().getCash() // 5
        self.__share4 = 0

        self.__cash5 = self.getBroker().getCash() // 5
        self.__share5 = 0

        self.__aroon = aroon.AroonBands(self.__price, bBandsPeriod, 2)

 
    def getPriceDS(self):
        return self.__price
    
    def getSMA5(self):
        return self.__sma5
    
    def getEMA5(self):
        return self.__ema5
    
    def getSMA21(self):
        return self.__sma21
    
    def getRSI23(self):
        return self.__rsi23
    
    def getAroon(self):
        return self.__aroon
    
    def onOrderUpdated(self, order):
        orderType = "Buy" if order.isBuy() else "Sell"
        # self.info("%s order %d updated - Status: %s" % (
        #     orderType, order.getId(), basebroker.Order.State.toString(order.getState())
        # ))
 
    def onBars(self, bars):
        # atr = self.__atr.getDataSeries()
        if self.__bbands.getLowerBand()[-1] == None:
            return
        if self.__atr[-1] == None:
            return
        if self.__minN[-1] == None:
            return
        atr = self.__atr[-1]
        # print(self.__atr[-1])
        # print(self.__atr.__dict__)
        # aroon_up = 0
        # aroon_down = 0
        # if self.__sma21[-1] is None:
        #     return
        
        aroon = indicator.AROON(self.getFeed().getDataSeries(self.__instrument), 252, 26)
        if aroon == None:
            return
            print("----1---")
        aroon_down = aroon[0][-1]
        aroon_up = aroon[1][-1]
        # print(aroon_up)
        # print(aroon_down)
            # print(type(aroon[-1]))
            # print(aroon[-1].shape)
        brk = self.getBroker()
        shares = brk.getShares(self.__instrument)
        bar = bars[self.__instrument]
        # if self.__bbands.getMiddleBand()[-3] is None:
        #     return
        # if aroon_up > 50 and aroon_up > aroon_down:
        
        # if self.__atr[-1] > 0:
        # print(bar.getClose(), aroon_up)
        # if aroon_up > 70:# and self.__sma21[-1] > self.__sma21[-2] and self.__sma21[-2] > self.__sma21[-3] and self.__sma21[-1] - self.__sma21[-2] > self.__sma21[-2] - self.__sma21[-3]:
        # buy1 = self.__sma5[-1] - atr * 1
        # buy2 = self.__sma5[-1] - atr * 3
        # buy3 = self.__sma5[-1] - atr * 5
        # buy4 = self.__sma5[-1] - atr * 7
        # buy5 = self.__sma5[-1] - atr * 9

        # sell5 = self.__sma5[-1] + atr * 9
        # sell4 = self.__sma5[-1] + atr * 7
        # sell3 = self.__sma5[-1] + atr * 5
        # sell2 = self.__sma5[-1] + atr * 3
        # sell1 = self.__sma5[-1] + atr * 1
        lower = self.__bbands.getLowerBand()[-1]
        middle = self.__bbands.getMiddleBand()[-1]
        upper = self.__bbands.getUpperBand()[-1]
        buy1 = middle - (middle-lower)*0.1
        buy2 = middle - (middle-lower)*0.3
        buy3 = middle - (middle-lower)*0.5
        buy4 = middle - (middle-lower)*0.7
        buy5 = middle - (middle-lower)*0.9

        sell5 = middle + (upper-middle)*0.1
        sell4 = middle + (upper-middle)*0.3
        sell3 = middle + (upper-middle)*0.5
        sell2 = middle + (upper-middle)*0.7
        sell1 = middle + (upper-middle)*0.9

        brk.getCash(False)
        sharesToBuy = 0

        if bar.getClose() < buy1 and self.__share1 == 0:
            # if shares == 0: 
                # sharesToBuy = int(brk.getCash(False) / bar.getClose())
            sharesToBuy1 = int(self.__cash1 / bar.getClose())
            self.__share1 = sharesToBuy1
            self.__cash1 -= sharesToBuy1 * bar.getClose()
            sharesToBuy += sharesToBuy1
        if bar.getClose() < buy2 and self.__share2 == 0:
            sharesToBuy2 = int(self.__cash2 / bar.getClose())
            self.__share2 = sharesToBuy2
            self.__cash2 -= sharesToBuy2 * bar.getClose()
            sharesToBuy += sharesToBuy2
        if bar.getClose() < buy3 and self.__share3 == 0:
            sharesToBuy3 = int(self.__cash3 / bar.getClose())
            self.__share3= sharesToBuy3
            self.__cash3 -= sharesToBuy3 * bar.getClose()
            sharesToBuy += sharesToBuy3
        if bar.getClose() < buy4 and self.__share4 == 0:
            sharesToBuy4 = int(self.__cash4 / bar.getClose())
            self.__share4= sharesToBuy4
            self.__cash4 -= sharesToBuy4 * bar.getClose()
            sharesToBuy += sharesToBuy4
        if bar.getClose() < buy5 and self.__share5 == 0:
            sharesToBuy5 = int(self.__cash5 / bar.getClose())
            self.__share5= sharesToBuy5
            self.__cash5 -= sharesToBuy5 * bar.getClose()
            sharesToBuy += sharesToBuy5

        if sharesToBuy > 0:
            self.info(f"Placing buy market order for {sharesToBuy} shares with price {bar.getClose()} buy1:{buy1} buy2:{buy2} buy3:{buy3}")
            self.marketOrder(self.__instrument, sharesToBuy, onClose=True)
        # elif aroon_down < 50 and aroon_up < aroon_down:
        # elif aroon_down > 70 or aroon_up < 60:# or aroon_down > 70:
        # elif bar.getClose() < self.__sma5[-1] + atr:
        #     if shares > 0:

        #         self.info(f"Placing sell market order for {shares} shares with price {bar.getClose()}")
        #         self.marketOrder(self.__instrument, -1*shares, onClose=True)
        sharesToSell = 0
        if bar.getClose() > sell1 and self.__share1 > 0:
            sharesToSell += self.__share1
            self.__cash1 += self.__share1 * bar.getClose()
            self.__share1 = 0
        if bar.getClose() > sell2 and self.__share2 > 0:
            sharesToSell += self.__share2
            self.__cash2 += self.__share2 * bar.getClose()
            self.__share2 = 0
        if bar.getClose() > sell3 and self.__share3 > 0:
            sharesToSell += self.__share3
            self.__cash3 += self.__share3 * bar.getClose()
            self.__share3 = 0
        if bar.getClose() > sell4 and self.__share4 > 0:
            sharesToSell += self.__share4
            self.__cash4 += self.__share4 * bar.getClose()
            self.__share4 = 0
        if bar.getClose() > sell5 and self.__share5 > 0:
            sharesToSell += self.__share5
            self.__cash5 += self.__share5 * bar.getClose()
            self.__share5 = 0
        
        
        if sharesToSell > 0:
            self.info(f"Placing sell market order for {sharesToSell} shares with price {bar.getClose()} sell1:{sell1} sell2:{sell2} sell3:{sell3}")
            self.marketOrder(self.__instrument, -1*sharesToSell, onClose=True)

def run_strategy(instrument):
    bBandsPeriod = 104
    # instrument = "399003"
    
    # 下载股票数据
    # if not os.path.isfile(instrument+".csv"):
    # if os.path.isfile(instrument+".csv"):
    #     os.remove(instrument+".csv")

    # 从CSV文件加载bar feed
    feed = quandlfeed.Feed()
    feed.addBarsFromCSV(instrument, instrument+".csv")
    
    # 创建MyStrategy实例
    myStrategy = MyStrategy(feed, instrument, bBandsPeriod)
    myStrategy.setDebugMode(False)

    plt = plotter.StrategyPlotter(myStrategy, True, True, True)
    # 图例添加BOLL
    # plt.getInstrumentSubplot(instrument).addDataSeries("sma21", myStrategy.getSMA21())
    plt.getInstrumentSubplot(instrument).addDataSeries("sma5", myStrategy.getSMA5())
    plt.getInstrumentSubplot(instrument).addDataSeries("ema5", myStrategy.getEMA5())

    plt.getOrCreateSubplot("rsi").addDataSeries("rsi", myStrategy.getRSI23())

    # plt.getOrCreateSubplot("aroon").addDataSeries("aroon", myStrategy.getAroon())
    

    # 添加回测分析
    returnsAnalyzer = returns.Returns()
    myStrategy.attachAnalyzer(returnsAnalyzer)

    # 添加夏普比率分析
    sharpeRatioAnalyzer = sharpe.SharpeRatio()
    myStrategy.attachAnalyzer(sharpeRatioAnalyzer)

    drawdownAnalyzer = drawdown.DrawDown()
    myStrategy.attachAnalyzer(drawdownAnalyzer)

    tradesAnalyzer = trades.Trades()
    myStrategy.attachAnalyzer(tradesAnalyzer)


    # 运行策略
    myStrategy.run()
    
    # 输出投资组合的最终资产总值
    print(instrument)
    print("最终资产总值: $%.2f" % myStrategy.getBroker().getEquity())
    # 输出年度收益
    print("总收益: %.2f %%" % (returnsAnalyzer.getCumulativeReturns()[-1] * 100))
    # 输出夏普比率
    print("夏普比率: %.2f" % sharpeRatioAnalyzer.getSharpeRatio(0))
    print("最大回撤: %.2f" % drawdownAnalyzer.getMaxDrawDown())
    print("胜手: %d" % tradesAnalyzer.getProfitableCount())
    print("负手: %d" % tradesAnalyzer.getUnprofitableCount())
    # return (returnsAnalyzer.getCumulativeReturns()[-1] * 100)
    # 展示折线图
    plt.plot()


def get_hist(code, start, end, timeout = 10,
                    retry_count=3, pause=0.001):
    '''
    code 股票代码
    '''
    def _code_to_symbol(code):
        '''
            生成 symbol 代码标志
        '''
        if len(code) != 6:
            return code
        else:
            return f'sh{code}' if code[:1] in ['5', '6', '9'] or code[:2] in ['11', '13'] else f'sz{code}'
    code = _code_to_symbol(code)
    url = "http://api.finance.ifeng.com/akdaily/?code=%s&type=last" % code
    
    for _ in range(retry_count):
        time.sleep(pause)
        try:
            text = requests.get(url, timeout=timeout).text
            if len(text) < 15: #no data
                return None
        except Exception as e:
            print(e)
        else:
            js = json.loads(text)
            cols = ['date', 'open', 'high', 'close', 'low', 'volume',
                        'price_change', 'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10', 'v_ma20']
            df = pd.DataFrame(js['record'], columns=cols)
            df = df.applymap(lambda x: x.replace(u',', u''))
            df[df==''] = 0
            for col in cols[1:]:
                df[col] = df[col].astype(float)
            if start is not None:
                df = df[df.date >= start]
            if end is not None:
                df = df[df.date <= end]
            df = df.set_index('date')
            df = df.sort_index()
            df = df.reset_index(drop=False)
            return df
    raise IOError('获取失败 请检查网络')

def create_dataframe2(code):
    # today = datetime.datetime.now() # .strftime("%Y%m%d")
    # two_month_ago = today - relativedelta(months=12)
    # df = get_hist(code, two_month_ago.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
    df = get_hist(code, None, None)

    df2 = pd.DataFrame({'Date' : df['date'], 'Open' : df['open'],
                        'High' : df['high'], 'Low' : df['low'],
                        'Close' : df['close'],'Volume' : df['volume'],
                        'Adj Close':df['close']})

    param = {"Referer": "https://finance.sina.com.cn"}
    resp = requests.get(f"https://hq.sinajs.cn/list=sz{code},s_sz{code}", headers=param)

    data = resp.content.decode(encoding='gb2312')

    lines = data.splitlines()
    list1 = lines[0].split("\"")[1].split(',')
    list2 = lines[1].split("\"")[1].split(',')
    # print(list1)
    # print(list2)
    date = list1[30]
    open = list1[1]
    high = list1[4]
    low = list1[5]
    close = list1[3]
    volume = list2[4]
    new=pd.DataFrame({'Date':date,
                  'Open':round(float(open),2),
                  'High':round(float(high),2),
                  'Low':round(float(low),2),
                  'Close':round(float(close),2),
                  'Volume':round(float(volume),1),
                  'Adj Close':round(float(close),2)},index=[1])

    if new.iat[0, 0] != df2.iat[-1, 0]:
        df2 = pd.concat([new,df2], axis=0 ,ignore_index=True)
    return list1[0], df2


if __name__ == "__main__":
    # 证券 399975
    arr_code = ["399975"]
    sum = 0.0
    count = 0
    for code in arr_code:
        try:
            if not os.path.exists(f"{code}.csv"):
                name, df = create_dataframe2(code)
                if name is None:
                    continue
                df.to_csv(f"{code}.csv", index=False)
        except Exception as e:
            print("error happend", e)
        else:
            sum += run_strategy(code)
            count += 1
    print("所有股票的平均收益率: ", sum/count)
