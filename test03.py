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

        self.__maxN = highlow.High(self.__price, 21)
        self.__minN = highlow.Low(self.__price, 26)

        self.__ema5 = ma.EMA(self.__price, 21)
        self.__sma5 = ma.SMA(self.__price, 5)
        self.__sma21 = ma.SMA(self.__price, 21)

        self.__dict = {}

        self.setDebugMode(False)
        self.__date = self.getCurrentDateTime()
        self.__realDate = datetime.datetime.now().strftime("%Y-%m-%d 00:00:00")

        self.__aroon = aroon.AroonBands(self.__price, bBandsPeriod, 2)

    def getMaxN(self):
        return self.__maxN
 
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
        lower = self.__bbands.getLowerBand()[-1]
        if lower is None:
            return
        shares = self.getBroker().getShares(self.__instrument)
        bar = bars[self.__instrument]
        if shares == 0 and self.__sma21[-1] > self.__sma21[-2] > self.__sma21[-3] and self.__sma21[-1] - self.__sma21[-2] > self.__sma21[-2] - self.__sma21[-3]:
            sharesToBuy = int(self.getBroker().getCash(False) / bar.getClose())
            self.info(f"Placing buy market order for {sharesToBuy} shares")
            self.marketOrder(self.__instrument, sharesToBuy)
        elif shares > 0:
            if self.__sma21[-1] < self.__sma21[-2] < self.__sma21[-3]:# and self.__sma21[-2] - self.__sma21[-1] > self.__sma21[-3] - self.__sma21[-2]:
        # and bar.getClose() < self.__maxN[-1] * 0.95:
                self.info(f"Placing sell market order for {shares} shares")
                self.marketOrder(self.__instrument, -1*shares)

def run_strategy(instrument, name = ""):
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
    plt.getOrCreateSubplot("maxN").addDataSeries("maxN", myStrategy.getMaxN())

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
    print(instrument, name)
    print("最终资产总值: $%.2f" % myStrategy.getBroker().getEquity())
    # 输出年度收益
    print("总收益: %.2f %%" % (returnsAnalyzer.getCumulativeReturns()[-1] * 100))
    # 输出夏普比率
    print("夏普比率: %.2f" % sharpeRatioAnalyzer.getSharpeRatio(0))
    print("最大回撤: %.2f" % drawdownAnalyzer.getMaxDrawDown())
    print("胜手: %d" % tradesAnalyzer.getProfitableCount())
    print("负手: %d" % tradesAnalyzer.getUnprofitableCount())
    return (returnsAnalyzer.getCumulativeReturns()[-1] * 100)
    # 展示折线图
    # plt.plot()


def get_hist(code, start, end, timeout = 10,
                    retry_count=3, pause=0.001):
    '''
    从凤凰财经获取股票历史数据
    code        股票代码
    start       起始日期
    end         结束日期
    timeout     超时时间
    retry_count 重试次数
    pause       请求间隔
    '''
    def _code_to_symbol(code):
        '''
            生成 symbol 代码标志，用于区分上证深证
        '''
        if len(code) != 6:
            return code
        else:
            return f'sh{code}' if code[:1] in ['5', '6', '9'] or code[:2] in ['11', '13'] else f'sz{code}'

    code = _code_to_symbol(code)
    url = f"http://api.finance.ifeng.com/akdaily/?code={code}&type=last"

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

def create_dataframe(code):
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
    print(data)
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
    arr_code = ["399006", "399997", "399417"] # 399006
    sum = 0.0
    count = 0
    for code in arr_code:
        name = ""
        try:
            if not os.path.exists(f"{code}.csv"):
                name, df = create_dataframe(code)
                if name is None:
                    continue
                df.to_csv(f"{code}.csv", index=False)
        except Exception as e:
            print("error happend", e)
        else:
            sum += run_strategy(code, name)
            count += 1
    print("所有股票的平均收益率: ", sum/count)
