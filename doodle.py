#%%
from copy import copy
import time
tic = time.time()

import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
import math
import matplotlib.patches as patches
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation

class Portfolio:
    def __init__(self, start, portfolio):
        self.start = start
        self.portfolio = portfolio

class Run:
    def __init__(self, start, end, portfolio):
        self.start = start
        self.end = end
        self.portfolio = portfolio
        self.daterange = pd.DataFrame(index=pd.date_range(start=start,end=end))
    
    def datevalues(self):
        start = self.start
        end = self.end
        datevalues = pd.DataFrame(index=pd.date_range(start,end))

def loadFeed(ticker):
    binancetickers = ["XMR", "BTTUSDT", "WINUSDT", "VETUSDT", "NKNUSDT", "ZILUSDT", "HOTUSDT", "FTMUSDT", "HBARUSDT", "TLMUSDT", "ANKRUSDT", "TFUELUSDT", "SHIBUSDT", "TRXUSDT", "DGBUSDT", "BNBUSDT", "MITHUSDT", "COTIUSDT", "CTXCUSDT", "VTHOUSDT", "ONTUSDT", "REEFUSDT", "DNTUSDT", "BTSUSDT", "IOTAUSDT", "AIONUSDT", "RIFUSDT", "ARDRUSDT", "JSTUSDT", "FUNUSDT", "FIOUSDT", "XEMUSDT", "TCTUSDT", "DOCKUSDT", "CKBUSDT", "STXUSDT", "DATAUSDT", "STPTUSDT", "GTOUSDT", "OCEANUSDT", "WAVESUSDT", "RSRUSDT", "RVNUSDT", "PONDUSDT", "SLPUSDT", "STMXUSDT", "MDTUSDT", "KEYUSDT", "MBLUSDT", "SXPUSDT", "MFTUSDT", "NBSUSDT", "LINAUSDT", "PERLUSDT", "DUSKUSDT", "AKROUSDT", "DENTUSDT", "ADAUSDT", "DOTUSDT", "ALGOUSDT", "IOTXUSDT", "ROSEUSDT", "IRISUSDT", "CELRUSDT", "ARPAUSDT", "COSUSDT", "IOSTUSDT", "BELUSDT", "ONEUSDT", "VITEUSDT"]
    file = ticker+'EUR_1440.csv'
    if ticker in binancetickers:
        file = ticker+'.csv'
        #print(ticker)
        source = pd.read_csv(file, usecols=[0,1,2,3,4,5], names=["DateTime","Open","High","Low","Close","Volume"], header=1)
        source['DateTime'] = (pd.to_datetime(source['DateTime'],unit='ms'))
    else:
        source = pd.read_csv(file, usecols=[0,1,2,3,4,5], names=["DateTime","Open","High","Low","Close","Volume"], header=None)
        source['DateTime'] = (pd.to_datetime(source['DateTime'],unit='s')) 
    dates = []
    for i in range(0,len(source)):
        dates.append(source['DateTime'][i])
    prices = []
    for i in range(0,len(source)):
        prices.append(source['Close'][i])
    feed = pd.DataFrame(index=dates,columns=['close'], dtype='float', data=prices)
    return feed

def getStats(simresults,stratType):
    ind = []
    for i in range(0,len(simresults)):
        a = 'run '+str(i)
        ind.append(simresults.T[a].max())  
    stats = pd.DataFrame(index=simresults.index,columns=['max'],data=ind)
    ind = []
    for i in range(0,len(simresults)):
        a = 'run '+str(i)
        ind.append(simresults.T[a].min()) 
    stats['min'] = pd.DataFrame(index=simresults.index,columns=['min'],data=ind)
    stats['close'] = simresults['day '+str(len(simresults.columns)-1)]
    mins = []
    for i in range(0,len(stats['close'])):
        mins.append(stats['min'][i])
        """
        if stratType == "Long-Short":
            if stats['min'][i] < 0:
                mins.append(stats['min'][i])
        elif stratType == "Long":
            if stats['min'][i] < startAmount:
                mins.append(stats['min'][i])
        else:
            pass
        """
    maxes = []
    for i in range(0,len(stats['close'])):
        maxes.append(stats['max'][i])
        """
        if stratType == "Long-Short":
            if stats['max'][i] > 0:
                mins.append(stats['max'][i])
        elif stratType == "Long":
            if stats['max'][i] > startAmount:
                mins.append(stats['max'][i])
        else:
            pass
        """
    minmax = []
    for i in range(0,len(mins)):
        minmax.append(mins[i])
    for j in range(0,len(maxes)):
        minmax.append(maxes[j])
    stats.minmax = minmax
    ind = []
    for i in range(0,len(simresults)):
        a = 'run '+str(i)
        ind.append(simresults.T[a].mean()) 
    stats['avg'] = pd.DataFrame(index=simresults.index,columns=['min'],data=ind)
    return stats

def doSim(startAmount,basketsize,rundate,period,interval,iters,strat,lookback):
      runArrayIndex = []
      for i in range(0, iters): 
        runArrayIndex.append('run '+str(i))
      runArrayCols = []
      for i in range(0, period+1): 
        runArrayCols.append('day '+str(i))
      simresults = pd.DataFrame(index=runArrayIndex, columns=runArrayCols)

      iteration = 0
      for i in range(iters):
    
        list = pd.DataFrame(index=prices.head().columns, columns=['chg'])
        for j in range(len(prices.head().columns)):
          list['chg'][j] = prices.pct_change(periods=lookback)[prices.head().columns[j]][rundate]

        basket = pd.DataFrame(data=[1],columns=['ticker'])
        listsize = int(list.count())
        if basketsize > listsize:
            tempbasketsize = listsize
        else:
            tempbasketsize = basketsize
        baseAmount = startAmount / tempbasketsize
        if strat == "bottom":
            sortAsc = True
        else:
            sortAsc = False
        for j in range(tempbasketsize):
          basket.loc[j] = list.sort_values(by='chg', ascending=sortAsc).index[j]
          #print(basket.loc[j])

        basket['amount'] = basket['ticker']
        for j in range(tempbasketsize):
          basket['amount'].loc[j] = baseAmount / prices[basket['ticker'][j]][rundate]

        portfolio = pd.DataFrame(columns=['amount'],data=basket['amount'])
        portfolio.index = basket['ticker']
        
        endrundate = rundate + datetime.timedelta(days=period)

        p1 = Portfolio(rundate, portfolio)
        r1 = Run(rundate, endrundate, p1.portfolio)

        tickers = r1.portfolio.index
        runvalues = pd.DataFrame(index=pd.date_range(start=r1.start,end=r1.end), columns=['amount'])

        for j in range(len(r1.daterange)):
          value = 0
          for i in range(len(portfolio.index)):
            date = r1.start + datetime.timedelta(days=j)
            value = value + (r1.portfolio['amount'][tickers[i]] * prices[tickers[i]][date])
          runvalues['amount'][j] = round(value, 2)

        for i in range(0,len(runvalues['amount'])):
          simresults.at['run '+str(iteration), 'day '+str(i)] = runvalues['amount'][i]

        rundate = rundate + datetime.timedelta(days=interval)
        endrundate = endrundate + datetime.timedelta(days=interval)
        iteration = iteration + 1
      return simresults

def PrintStart():
    print("startAmount: ",startAmount)
    print("# baskets:   ",basketsize)
    print("baseAmount:  ",round(baseAmount,2))
    print(" --- ",)
    print("strategy:")
    if stratType == "Long-Short":
        print("Long-Short EW")
        print(" --- ",)
        print("Every",interval,"day(s):")
        print("Long the",lookback1,"day %chg",strat1,basketsize,"of available tickers")
        print("Short the",lookback2,"day %chg",strat2,basketsize,"of available tickers")
    elif stratType == "Long":
        print("Long EW")
        print(" --- ",)
        print("Every",interval,"day(s):")
        print("Long the",lookback1,"day %chg",strat1,basketsize,"of available tickers")
    print("Hold for",period,"days")
    print(" --- ",)
    print("sim start: ",rundate)
    print("sim iters: ",iters,'\n','\n'," --- ",'\n')


#initialize helper variables
numBins = 2
binWidth = 1

#load ticker variables
tickers = ["BTTUSDT", "WINUSDT", "VETUSDT", "NKNUSDT", "ZILUSDT", "HOTUSDT", "FTMUSDT", "HBARUSDT", "TLMUSDT", "ANKRUSDT", "TFUELUSDT", "SHIBUSDT", "TRXUSDT", "DGBUSDT", "BNBUSDT", "MITHUSDT", "COTIUSDT", "CTXCUSDT", "VTHOUSDT", "ONTUSDT", "REEFUSDT", "DNTUSDT", "BTSUSDT", "IOTAUSDT", "AIONUSDT", "RIFUSDT", "ARDRUSDT", "JSTUSDT", "FUNUSDT", "FIOUSDT", "XEMUSDT", "TCTUSDT", "DOCKUSDT", "CKBUSDT", "STXUSDT", "DATAUSDT", "STPTUSDT", "GTOUSDT", "OCEANUSDT", "WAVESUSDT", "RSRUSDT", "RVNUSDT", "PONDUSDT", "SLPUSDT", "STMXUSDT", "MDTUSDT", "KEYUSDT", "MBLUSDT", "SXPUSDT", "MFTUSDT", "NBSUSDT", "LINAUSDT", "PERLUSDT", "DUSKUSDT", "AKROUSDT", "DENTUSDT", "ADAUSDT", "DOTUSDT", "ALGOUSDT", "IOTXUSDT", "ROSEUSDT", "IRISUSDT", "CELRUSDT", "ARPAUSDT", "COSUSDT", "IOSTUSDT", "BELUSDT", "ONEUSDT", "VITEUSDT"]
for i in range(0,len(tickers)):
    ticker = tickers[i]
    vars()[ticker] = loadFeed(tickers[i])

#set date ranges, create prices DataFrame
for i in range(0,1):
    starttime = datetime.datetime.strptime("01-01-2018", "%d-%m-%Y")
    endtime = datetime.datetime.strptime("31-05-2021", "%d-%m-%Y")
    prices = pd.DataFrame(index=pd.date_range(start=starttime,end=endtime))

#load prices
for i in range(0,len(tickers)):
    ticker = tickers[i]
    prices[ticker] = vars()[ticker]['close']

#set initial conditions
for i in range(0,1):
    startAmount = 100
    basketsize = 5
    baseAmount = startAmount / basketsize
    
    period = 28
    rundate = datetime.datetime.strptime("01-01-2020", "%d-%m-%Y")
    #rundate = datetime.datetime.strptime("01-04-2018", "%d-%m-%Y")
    endrundate = rundate + datetime.timedelta(days=period)
    interval = 1
    iters = int((endtime - endrundate).days / interval) + 1
    #maxiters = (endtime - endrundate).days + 1
    iters = 50
    
    #lookback = 7
    lookback1 = 28
    lookback2 = 7
    
    stratType = "Long-Short"
    strat1 = "top"
    strat2 = "top"
    
    #stoploss = -25
    #stopTypeSL = 'tight'
    #takeprofit = 50
    #stopTypeTP = 'loose'
    #trigger = 13
    #stopTypeTR = 'tight'

def getSimResults(resultType):
    #implement crude stoploss
    def Stoploss(stoploss,stopTypeSL):
        for i in range(0,len(simresults.index)):
            for j in range(0,period):
                if simresults['day '+str(j)]['run '+str(i)] < stoploss:
                    remaining = period - j + 1
                    for k in range(0,remaining):
                        if stopTypeSL == 'loose':
                            simresults['day '+str(j+k)]['run '+str(i)] = simresults['day '+str(j)]['run '+str(i)]
                        else:
                            simresults['day '+str(j+k)]['run '+str(i)] = stoploss
                else:
                    pass
    #implement crude take-profit
    def Takeprofit(takeprofit, stopTypeTP):
        for i in range(0,len(simresults.index)):
            for j in range(0,period):
                if simresults['day '+str(j)]['run '+str(i)] > takeprofit:
                    remaining = period - j + 1
                    for k in range(0,remaining):
                        if stopTypeTP == 'loose':
                            simresults['day '+str(j+k)]['run '+str(i)] = simresults['day '+str(j)]['run '+str(i)]
                        else:
                            simresults['day '+str(j+k)]['run '+str(i)] = takeprofit
                else:
                    pass
    #implement crude trailing stop
    def TrailingStop(trigger,stopTypeTR):
        for i in range(0,len(simresults.index)):
            for j in range(0,period):
                if simresults['day '+str(j)]['run '+str(i)] > trigger:
                    remaining = period - j + 1
                    for k in range(0,remaining):
                        if simresults['day '+str(j)]['run '+str(i)] < trigger:
                            if stopTypeTR == 'loose':
                                simresults['day '+str(j+k)]['run '+str(i)] = simresults['day '+str(j)]['run '+str(i)]
                            if stopTypeTR == 'tight':
                                simresults['day '+str(j+k)]['run '+str(i)] = trigger
                        else:
                            pass
    #calculate simresults
    try:
        if stratType == "Long-Short":
            simresults1 = doSim(startAmount,basketsize,rundate,period,interval,iters,strat1,lookback1)
            simresults2 = doSim(startAmount,basketsize,rundate,period,interval,iters,strat2,lookback2)
            simresults = simresults1 - simresults2
        elif stratType == "Long":
            simresults = doSim(startAmount,basketsize,rundate,period,interval,iters,strat,lookback)
    except NameError:
        pass
    #calculate sl/tp/ts
    try:
        Stoploss(stoploss,stopTypeSL)
    except NameError:
        pass
    try:
        Takeprofit(takeprofit, stopTypeTP)
    except NameError:
        pass
    try:
        TrailingStop(trigger,stopTypeTR)
    except NameError:
        pass
    #calculate simresults pct chg
    try:
        if stratType == "Long-Short":
            chg1 = simresults1.T.pct_change()
            chg2 = simresults2.T.pct_change()
            chg = chg1 - chg2
            chg = chg.T
        elif stratType == "Long":
            chg = simresults.T.pct_change()
            chg = chg.T
    except NameError:        
        print("NameError")
        pass
    #return value
    if resultType == 'simresults':
        return simresults
    elif resultType == 'pct chg':
        return chg
def MakeStats(simresults):
    stats = getStats(simresults,stratType)
    return stats

PrintStart()

#do the sim and get results
for i in range(0,1):
    tic = time.time()
    simresults = getSimResults('simresults')
    simresultsPctChg = getSimResults('pct chg')
    toc = time.time()
    print("Finished simulation. ",f"{toc-tic:.3f} sec. elapsed")
    print("sim results")
    print(simresults)
    print("sim results - % chg")
    print(simresultsPctChg)

#calculate stats
for i in range(0,1):
    tic = time.time()
    stats = MakeStats(simresults)
    toc = time.time()
    print("Finished stats. ",f"{toc-tic:.3f} sec. elapsed")

# set stratType vars
for i in range(0,1):
    if stratType == "Long-Short":
        zeropoint = 0
    elif stratType == "Long":
        zeropoint = 100
    stratTypeMin = zeropoint - stats['min'].min()
    stratTypeMax = stats['max'].max() - zeropoint
    if stratTypeMax > stratTypeMin:
        rangelimit = stratTypeMax
    else:
        rangelimit = stratTypeMin
    binWidth = 5
    numBins = 2 * (int(rangelimit / binWidth) + 1)
    
    xdisplaymargin = 0.05
    ydisplaymargin = 0.05


def summarizeStats(stats,stratType):
    statsMin = stats['min'].mean()
    statsMax = stats['max'].mean()
    statsClose = stats['close'].mean()
    statsMean = stats['avg'].mean()
    statsProfit = (stats['close'].sum())
    print("stats")
    print(stats)
    print("summary")
    if stratType == "Long":    
        # for single run
        print("Avg min:",round(startAmount*((statsMin/startAmount)-1),2),"%")
        print("Avg max:",round(startAmount*((statsMax/startAmount)-1),2),"%")
        print("Avg close:",round(startAmount*((statsClose/startAmount)-1),2),"%")
        print("Avg mean:",round(startAmount*((statsMean/startAmount)-1),2),"%")
        print("Avg profit:",round(statsProfit,2),"%")
    elif stratType == "Long-Short":
        #for dif between runs
        print("Avg min: ",round(((statsMin)),2))
        print("Avg max: ",round(((statsMax)),2))
        print("Avg close: ",round(((statsClose)),2))
        print("Avg mean: ",round(((statsMean)),2))
        print("Tot profit: ",round(statsProfit,2))
        print("Avg profit: ",round((statsProfit/stats['close'].count()),2))
        print("# runs: ",stats['close'].count())
    else:
        return 0
summarizeStats(stats,stratType)
#%%

def lineGraph(simresults):
    x = simresults.columns
    for i in range(0,len(simresults)):
        y = []
        for j in range(0,len(x)):
            y.append(simresults['day '+str(j)].iat[i])
        a = plt.plot(x,y, alpha=(len(x)/100))
        #xLabelNumbers = (i for i in range(0,len(simresults.columns)))
        #a.set_xticklabels(xLabelNumbers)
    plt.show()
#lineGraph(simresults)

def barGraph(simresultsPctChg):
    x = simresultsPctChg.columns
    for i in range(0,len(simresultsPctChg)):
        y = []
        for j in range(0,len(x)):
            y.append(simresultsPctChg['day '+str(j)].iat[i])
        a = plt.bar(x,y, alpha=(len(x)/100))
        #a.set_xlim(left=zeropoint, right=(1+xdisplaymargin)*x.max())
        #xLabelNumbers = (i for i in range(0,len(simresults.columns)))
        #a.set_xticklabels(xLabelNumbers)
    plt.show()
#barGraph(simresultsPctChg)

def aniHisto():
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    # Fixing bin edges
    #HIST_BINS = np.linspace(-4, 4, 100)
    #numBins = int(len(simresults) / 4)
    #numBins = int(round(math.sqrt(len(simresults)),0)*1.5)
    #HIST_BINS = np.linspace(stats['min'].min(), stats['max'].max(), numBins)
    HIST_BINS = np.linspace(zeropoint-rangelimit, zeropoint+rangelimit, numBins)
    # histogram our data with numpy
    data = np.random.randn(1000)
    #n, _ = np.histogram(data, HIST_BINS)
    
    def prepare_animation(bar_container):
        def animate(frame_number):
            data = []
            for i in range(1,len(simresults)):
                data.append(simresults['day '+str(frame_number)].iloc[i])
            # simulate new data coming in
            #data = np.random.randn(1000)
            n, _ = np.histogram(data, HIST_BINS)
            #print(HIST_BINS)
            for count, rect in zip(n, bar_container.patches):
                if rect.xy[0] < zeropoint:
                    rect.set_facecolor('red')
                else:
                    rect.set_facecolor('green')                 
                rect.set_height(count)
            return bar_container.patches
        return animate
    
    fig, ax = plt.subplots()
    _, _, bar_container = ax.hist(data, HIST_BINS, lw=0.5, ec='gray', fc='none', alpha=0.5, align='right')
    ax.set_ylim(top=int(len(simresults)/3)+1)  # set safe limit to ensure that all data is visible.
    
    # chg axes if stoploss/takeprofit
    for i in range(0,1):
        try:
            ax.set_xlim(left=stoploss)
        except NameError:
            pass
        try:
            ax.set_xlim(right=takeprofit)
        except NameError:
            pass
    
    ani = animation.FuncAnimation(fig, prepare_animation(bar_container), len(simresults.columns), repeat=False, blit=True)
    plt.show()
#aniHisto()

def plotStats():        
    n_bins = numBins
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=1)
    axs = gs.subplots()
    
    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(stats['close'].T, bins=n_bins, alpha=0.5)
    axs[1].hist(stats.minmax, bins=n_bins, alpha=0.5)
    
    axs[0].set_xlim(left=(1+xdisplaymargin)*min(stats.minmax), right=(1+xdisplaymargin)*max(stats.minmax))
    axs[1].set_xlim(left=(1+xdisplaymargin)*min(stats.minmax), right=(1+xdisplaymargin)*max(stats.minmax))
    
    xleft, xright = -2*startAmount, 2*startAmount
    ybottom, ytop = 0 , ((binWidth*numBins)/2*len(stats['close'].T))
    axs[0].set_xlim(left=xleft, right=xright)
    axs[1].set_xlim(left=xleft, right=xright)
    axs[0].set_ylim(bottom=ybottom)
    axs[1].set_ylim(bottom=ybottom)
    #axs[0].set_ylim(bottom=ybottom, top=int((ytop)/1.5)+1)
    #axs[1].set_ylim(bottom=ybottom, top=ytop)
    
    axs[0].set_title('Close')
    axs[1].set_title('MinMax')
            
    plt.show()
#plotStats()

def scatterHistoMin():
    min_is_zeropoint =  stats[stats['min'] == zeropoint]
    min_below_zeropoint =  stats[stats['min'] < zeropoint]
    min_below_zeropoint = min_below_zeropoint[~min_below_zeropoint['min'].isin(min_is_zeropoint['min'])]
    
    x = min_below_zeropoint['min']
    y = min_below_zeropoint['close']
    
    closes_above_zeropoint = min_below_zeropoint[(min_below_zeropoint['close'] > zeropoint)]
    lowest_min_with_close_above_zeropoint = closes_above_zeropoint['min'].min()
    lowest_line = (int(lowest_min_with_close_above_zeropoint / binWidth) + (math.copysign(1, lowest_min_with_close_above_zeropoint)*1)) * binWidth
    
    xdisplaymargin = 0.05
    ydisplaymargin = 0.05
    
    def scatter_hist(x, y, ax, ax_histx, ax_histy):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=True)
        ax_histy.tick_params(axis="y", labelleft=True)
        
        # the scatter plot:
        ax.scatter(x, y)
        
        # now determine nice limits by hand:
        binwidth = binWidth
        xymax = rangelimit
        lim = (int(xymax/binwidth) + 1) * binwidth
        
        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=bins, label='min')
        ax_histy.hist(y, bins=bins, label='close', orientation='horizontal')
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    
    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))
    
    ax = fig.add_axes(rect_scatter)
    ax.set_xlim(left=(1-xdisplaymargin)*x.min(), right=zeropoint)
    ax.set_xlabel("min")
    ax.set_ylim(top=(1+ydisplaymargin)*y.max(), bottom=(1-ydisplaymargin)*y.min())
    ax.set_ylabel("close")
    
    ax.add_patch(patches.Rectangle(
        xy=(((1-xdisplaymargin)*x.min()),zeropoint),  # point of origin.
        width=abs(((1-xdisplaymargin)*x.min()))-abs(lowest_line),
        height=abs(((1+ydisplaymargin)*y.max()))-zeropoint,
        linewidth=1,
        color='yellow',
        fill=True,
        alpha=0.5,
        zorder=-100))
    ax.add_patch(patches.Rectangle(
        xy=(lowest_line,((1-ydisplaymargin)*y.min())),  # point of origin.
        width=abs(lowest_line),
        height=abs(((1-ydisplaymargin)*y.min())),
        linewidth=1,
        color='orange',
        fill=True,
        alpha=0.5,
        zorder=-100))
    ax.add_patch(patches.Rectangle(
        xy=(((1-xdisplaymargin)*x.min()),((1-ydisplaymargin)*y.min())),  # point of origin.
        width=abs(((1-xdisplaymargin)*x.min()))-abs(lowest_line),
        height=abs(((1-ydisplaymargin)*y.min())),
        linewidth=2,
        color='red',
        fill=True,
        alpha=0.5,
        zorder=-100))
    ax.add_patch(patches.Rectangle(
        xy=(lowest_line,zeropoint),  # point of origin.
        width=abs(lowest_line),
        height=((1+ydisplaymargin)*y.max()),
        linewidth=2,
        color='green',
        fill=True,
        alpha=0.5,
        zorder=-100))
    
    ax_histx = fig.add_axes(rect_histx, sharex=ax, label='min')
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    
    #ax.plot([x.min(), zeropoint], [zeropoint, zeropoint], color='green')
    #ax.plot([lowest_line, lowest_line], [(1-ydisplaymargin)*y.min(), (1+ydisplaymargin)*y.max()], color='green')
    
    # use the previously defined function
    scatter_hist(x, y, ax, ax_histx, ax_histy)
    
    plt.title('min < 0')
    
    plt.show()
def scatterHistoMax():
    max_is_zeropoint =  stats[stats['max'] == zeropoint]
    max_above_zeropoint =  stats[stats['max'] > zeropoint]
    max_above_zeropoint = max_above_zeropoint[~max_above_zeropoint['max'].isin(max_is_zeropoint['max'])]
    
    x = max_above_zeropoint['max']
    y = max_above_zeropoint['close']
    
    closes_below_zeropoint = max_above_zeropoint[(max_above_zeropoint['close'] < zeropoint)]
    highest_max_with_close_below_zeropoint = closes_below_zeropoint['max'].max()
    lowest_line = (int(highest_max_with_close_below_zeropoint / binWidth) + (math.copysign(1, highest_max_with_close_below_zeropoint)*1)) * binWidth
    
    xdisplaymargin = 0.05
    ydisplaymargin = 0.05
    
    def scatter_hist(x, y, ax, ax_histx, ax_histy):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=True)
        ax_histy.tick_params(axis="y", labelleft=True)
        
        # the scatter plot:
        ax.scatter(x, y)
        
        # now determine nice limits by hand:
        binwidth = binWidth
        xymax = rangelimit
        lim = (int(xymax/binwidth) + 1) * binwidth
        
        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=bins, label='max')
        ax_histy.hist(y, bins=bins, label='close', orientation='horizontal')
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    
    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))
    
    ax = fig.add_axes(rect_scatter)
    ax.set_xlim(left=zeropoint, right=(1+xdisplaymargin)*x.max())
    ax.set_xlabel("max")
    ax.set_ylim(top=(1+ydisplaymargin)*y.max(), bottom=(1+ydisplaymargin)*y.min())
    ax.set_ylabel("close")
    
    ax.add_patch(patches.Rectangle(
        xy=(zeropoint,zeropoint),  # point of origin.
        width=abs(abs(lowest_line)-abs(zeropoint)), 
        height=abs(((1+ydisplaymargin)*y.max())),
        linewidth=1,
        color='yellow',
        fill=True,
        alpha=0.5,
        zorder=-100))
    ax.add_patch(patches.Rectangle(
        xy=(lowest_line,((1+ydisplaymargin)*y.min())),  # point of origin.
        width=abs(((1+xdisplaymargin)*x.max()))-abs(lowest_line),
        height=abs(((1+ydisplaymargin)*y.min())),
        linewidth=1,
        color='orange',
        fill=True,
        alpha=0.5,
        zorder=-100))
    ax.add_patch(patches.Rectangle(
        xy=(zeropoint,((1+ydisplaymargin)*y.min())),  # point of origin.
        width=abs(abs(lowest_line)-abs(zeropoint)),
        height=abs(((1+ydisplaymargin)*y.min())),
        linewidth=2,
        color='red',
        fill=True,
        alpha=0.5,
        zorder=-100))
    ax.add_patch(patches.Rectangle(
        xy=(lowest_line,zeropoint),  # point of origin.
        width=abs(((1+xdisplaymargin)*x.max()))-abs(lowest_line),
        height=((1+ydisplaymargin)*y.max()),
        linewidth=2,
        color='green',
        fill=True,
        alpha=0.5,
        zorder=-100))
    
    ax_histx = fig.add_axes(rect_histx, sharex=ax, label='max')
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    
    ax.plot([zeropoint, (1+xdisplaymargin)*x.max()], [zeropoint, zeropoint], color='green')
    ax.plot([lowest_line, lowest_line], [(1+ydisplaymargin)*y.min(), (1+ydisplaymargin)*y.max()], color='green')
    
    # use the previously defined function
    scatter_hist(x, y, ax, ax_histx, ax_histy)
    
    plt.title('max > 0')
    
    plt.show()

#scatterHistoMin()
#scatterHistoMax()

def scatterHistoMinMax():
    never_ran =  stats[stats['close'] == zeropoint]
    
    never_lost =  stats[stats['min'] == zeropoint]
    never_lost =  never_lost[~never_lost['min'].isin(never_ran['min'])]
    
    never_won =  stats[stats['max'] == zeropoint]
    never_won =  never_won[~never_won['max'].isin(never_ran['max'])]
    
    min_below_zeropoint =  stats[stats['min'] < zeropoint]
    max_above_zeropoint =  stats[stats['max'] > zeropoint]
    min_below_zeropoint = min_below_zeropoint[~min_below_zeropoint['min'].isin(never_lost['min'])]
    max_above_zeropoint = max_above_zeropoint[~max_above_zeropoint['max'].isin(never_won['max'])]
    dropped_lost = min_below_zeropoint[(min_below_zeropoint['close'] < zeropoint)]
    dropped_lost = dropped_lost[~dropped_lost['max'].isin(never_won['max'])]
    dropped_won = min_below_zeropoint[(min_below_zeropoint['close'] > zeropoint)]
    dropped_won = dropped_won[~dropped_won['min'].isin(never_lost['min'])]
    rose_lost = max_above_zeropoint[(max_above_zeropoint['close'] < zeropoint)]
    rose_lost = rose_lost[~rose_lost['max'].isin(never_won['max'])]
    rose_won = max_above_zeropoint[(max_above_zeropoint['close'] > zeropoint)]
    rose_won = rose_won[~rose_won['min'].isin(never_lost['min'])]
    
    min_closes_above_zeropoint = min_below_zeropoint[(min_below_zeropoint['close'] > zeropoint)]
    lowest_min_with_close_above_zeropoint = min_closes_above_zeropoint['min'].min()
    lowest_line_min = (int(lowest_min_with_close_above_zeropoint / binWidth) + (math.copysign(1, lowest_min_with_close_above_zeropoint)*1)) * binWidth
    closes_below_zeropoint = max_above_zeropoint[(max_above_zeropoint['close'] < zeropoint)]
    highest_max_with_close_below_zeropoint = closes_below_zeropoint['max'].max()
    lowest_line_max = (int(highest_max_with_close_below_zeropoint / binWidth) + (math.copysign(1, highest_max_with_close_below_zeropoint)*1)) * binWidth
    
    XofMin = never_won['min']
    YofMin = never_won['close']
    XofMax = never_lost['max']
    YofMax = never_lost['close']
    
    displaymargin = 0.05
    
    def scatter_hist(x, y, ax, ax_histx, ax_histy, scatterHistXLabel):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=True)
        ax_histy.tick_params(axis="y", labelleft=True)
        
        # the scatter plot:
        ax.scatter(x, y, label=scatterHistXLabel)
        
        # now determine nice limits by hand:
        binwidth = binWidth
        xymax = rangelimit
        lim = (int(xymax/binwidth) + 1) * binwidth
        
        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=bins, label=scatterHistXLabel, histtype='bar', stacked=True)
        ax_histy.hist(y, bins=bins, label='close', orientation='horizontal')
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    
    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))
    
    ax = fig.add_axes(rect_scatter)
    #ax.set_xlim(left=(1-displaymargin)*XofMin.min(), right=(1+displaymargin)*XofMax.max())
    #ax.set_xlabel("min")
    #ax.set_ylim(top=(1+displaymargin)*YofMax.max(), bottom=(1-displaymargin)*YofMin.min())
    #ax.set_ylabel("close")
    
    ax.add_patch(patches.Rectangle(
        xy=(((1-displaymargin)*XofMin.min()),((1-displaymargin)*YofMin.min())),  # point of origin.
        width=(abs(abs(lowest_line_max)-abs(zeropoint)))+abs(lowest_line_min)+abs(((1-displaymargin)*XofMin.min()))-abs(lowest_line_min),
        height=abs(((1+displaymargin)*YofMin.min())),
        linewidth=2,
        color='red',
        fill=True,
        alpha=0.5,
        zorder=-100))
    ax.add_patch(patches.Rectangle(
        xy=(((1+displaymargin)*XofMin.min()),((1+displaymargin)*YofMin.min())),  # point of origin.
        width=abs(((1-displaymargin)*XofMin.min()))-abs(lowest_line_min),
        height=abs(((1+displaymargin)*YofMin.min())),
        linewidth=2,
        color='red',
        fill=True,
        alpha=0.5,
        zorder=-100))
    ax.add_patch(patches.Rectangle(
        xy=(lowest_line_min,zeropoint),  # point of origin.
        width=abs(lowest_line_min),
        height=((1+displaymargin)*YofMin.max()),
        linewidth=2,
        color='green',
        fill=True,
        alpha=0.5,
        zorder=-100))
    ax.add_patch(patches.Rectangle(
        xy=(zeropoint,((1+displaymargin)*YofMax.min())),  # point of origin.
        width=abs(abs(lowest_line_max)-abs(zeropoint)),
        height=abs(((1+displaymargin)*YofMax.min())),
        linewidth=2,
        color='red',
        fill=True,
        alpha=0.5,
        zorder=-100))
    ax.add_patch(patches.Rectangle(
        xy=(lowest_line_max,zeropoint),  # point of origin.
        width=abs(lowest_line_max),
        height=((1+displaymargin)*YofMax.max()),
        linewidth=2,
        color='green',
        fill=True,
        alpha=0.5,
        zorder=-100))
    
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    
    #ax.plot([x.min(), zeropoint], [zeropoint, zeropoint], color='green')
    #ax.plot([lowest_line, lowest_line], [(1-ydisplaymargin)*y.min(), (1+ydisplaymargin)*y.max()], color='green')
    
    # use the previously defined function
    #scatter_hist(XofMin, YofMin, ax, ax_histx, ax_histy)
    #scatter_hist(XofMax, YofMax, ax, ax_histx, ax_histy)
    
    p1 = scatter_hist(never_won['max'], never_won['close'], ax, ax_histx, ax_histy, 'never_won')
    p2 = scatter_hist(dropped_lost['min'], dropped_lost['close'], ax, ax_histx, ax_histy, 'dropped_lost')
    p3 = scatter_hist(rose_lost['min'], rose_lost['close'], ax, ax_histx, ax_histy, 'rose_lost')
    p4 = scatter_hist(never_lost['min'], never_lost['close'], ax, ax_histx, ax_histy, 'never_lost')
    p5 = scatter_hist(dropped_won['max'], dropped_won['close'], ax, ax_histx, ax_histy, 'dropped_won')
    p6 = scatter_hist(rose_won['max'], rose_won['close'], ax, ax_histx, ax_histy, 'rose_won')
    
    #ax.legend([p1,p2,p3,p4,p5,p6],['never_won','dropped_lost','rose_lost','never_lost','dropped_won','rose_won'])
    #plt.title('min < 0')
    
    #ax.legend(handles=[p1,p2,p3,p4,p5,p6], title='title', bbox_to_anchor=(1.05, 1), loc='upper right')
    
    plt.show()
#scatterHistoMinMax()


def aniPctChg():
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    # Fixing bin edges
    #HIST_BINS = np.linspace(-4, 4, 100)
    #numBins = int(len(simresults) / 4)
    #numBins = int(round(math.sqrt(len(simresults)),0)*1.5)
    #HIST_BINS = np.linspace(stats['min'].min(), stats['max'].max(), numBins)
    HIST_BINS = np.linspace(-1,1, numBins)
    # histogram our data with numpy
    data = np.random.randn(1000)
    #n, _ = np.histogram(data, HIST_BINS)
    
    def prepare_animation(bar_container):
        def animate(frame_number):
            data = []
            for i in range(0,len(simresultsPctChg)):
                data.append(simresultsPctChg['day '+str(frame_number)].iloc[i])
            #data.sort()
            # simulate new data coming in
            #data = np.random.randn(1000)
            n, _ = np.histogram(data, HIST_BINS)
            #print(HIST_BINS)
            for count, rect in zip(n, bar_container.patches):
                if rect.xy[0] < 0:
                    rect.set_facecolor('red')
                else:
                    rect.set_facecolor('green')                 
                rect.set_height(count)
            return bar_container.patches
        return animate
    
    fig, ax = plt.subplots()
    _, _, bar_container = ax.hist(data, HIST_BINS, lw=0.5, ec='gray', fc='none', alpha=0.5, align='right')
    ax.set_ylim(bottom=-1,top=1)
    
    ani = animation.FuncAnimation(fig, prepare_animation(bar_container), len(simresultsPctChg.columns), repeat=False, blit=True)
    plt.show()
#aniPctChg()


#%%
"""
def aniScatterXYZ():
    # without blitting
    for i in range(0,1):
        period = 7
        runs = 10
        a = np.random.rand(period*runs)
        b = np.random.rand(period*runs)
        c = np.random.rand(period*runs)
        t = np.array([np.ones(runs)*int(i) for i in range(period)]).flatten()
        df = pd.DataFrame({"day": t ,"x" : a, "y" : b, "z" : c})
        
        
        
        df['dayindex'] = df['day']
        for i in range(0,len(df['dayindex'])):
            df['dayindex'].iloc[i] = 'day '+str(df['day'].iloc[i]).replace('.0', '')
        
        print("a")
        print(a)
        print("t")
        print(t)
        print("df")
        print(df)
        print("df['dayindex']")
        print(df['dayindex'])
        #print("df['runindex']")
        #print(df['runindex'])
        
        def update_graph(num):
            data=df[df['day']==num]
            graph._offsets3d = (data.x, data.y, data.z)
            title.set_text('3D Test, day={}'.format(num))
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        title = ax.set_title('3D Test')
        
        data=df[df['day']==0]
        graph = ax.scatter(data.x, data.y, data.z)
        
        ani = matplotlib.animation.FuncAnimation(fig, update_graph, period-1, interval=500, blit=False)
        
        plt.show()

#aniScatterXYZ()
t = np.array([np.ones(iters)*int(i) for i in range(period+1)]).flatten()

df = pd.DataFrame({"abla": t})

df['dayindex'] = df['abla']
for i in range(0,len(df['dayindex'])):
    df['dayindex'].iloc[i] = 'day '+str(df['abla'].iloc[i]).replace('.0', '')

print(df)


t = np.array([np.ones(period+1)*int(i) for i in range(iters)]).flatten()

dh = pd.DataFrame({"abla": t})

dh['runindex'] = dh['abla']
for i in range(0,10):
    for j in range(0,iters):
        helper = 'run '+str(j)
        dh['runindex'].iloc[i] = helper


print(dh)


dg = pd.DataFrame({"abla": t})
dg['runindex'] = dh['runindex']
dg['dayindex'] = df['dayindex']

print(dg)
"""
#%%

displaymargin = 0.05

dayno = np.array([np.ones(iters)*int(i) for i in range(period+1)]).flatten()
runlist = [i for i in range(0,iters)]*(period+1)

a = np.random.rand((period+1)*iters)
b = np.random.rand((period+1)*iters)
c = np.random.rand((period+1)*iters)

df = pd.DataFrame({"day": dayno ,"run" : runlist, "x" : a, "y" : b, "z" : c})

for i in range(0,len(df['day'])):
    dayno = str(df['day'].iloc[i]).replace('.0', '')
    df['day'].iloc[i] = 'day '+str(dayno)

for i in range(0,len(df['run'])):
    runno = str(df['run'].iloc[i]).replace('.0', '')
    df['run'].iloc[i] = 'run '+str(runno)

print(df)

# load data
for r in range(0,1):
    xDataForScatter3D = pd.DataFrame(index=simresults.index, columns=[simresults.columns], data=[])
    yDataForScatter3D = pd.DataFrame(index=simresults.index, columns=[simresults.columns], data=[])
    zDataForScatter3D = pd.DataFrame(index=simresults.index, columns=[simresults.columns], data=[])
    for q in range(0,len(simresults.index)):
        for i in range(0,period+1):
            list = []
            for j in range(0,i+1):
                list.append(simresults['day '+str(j)]['run '+str(q)])
            xDataForScatter3D.iloc[q][i] = min(list)
            yDataForScatter3D.iloc[q][i] = max(list)
            zDataForScatter3D.iloc[q][i] = simresults.iloc[q][i]
    print("xDataForScatter3D")
    print(xDataForScatter3D)
    print("yDataForScatter3D")
    print(yDataForScatter3D)
    print("zDataForScatter3D")
    print(zDataForScatter3D)
    
    never_ran =  zDataForScatter3D[zDataForScatter3D['day '+str(period)] == zeropoint]
    actual_runs = zDataForScatter3D[~zDataForScatter3D['day '+str(period)].isin(never_ran['day '+str(period)])]
    #print(never_ran)
    
    # clean-up and separation
    for i in range(0,1):
        closed_at_zeropoint = zDataForScatter3D[zDataForScatter3D['day '+str(period)] == zeropoint]
        
        closed_above_zeropoint = zDataForScatter3D[zDataForScatter3D['day '+str(period)] > zeropoint]
        closed_above_zeropoint = closed_above_zeropoint[~closed_above_zeropoint['day '+str(period)].isin(closed_at_zeropoint['day '+str(period)])]
        closed_above_zeropoint = closed_above_zeropoint.dropna(how='all')
        
        closed_below_zeropoint = zDataForScatter3D[zDataForScatter3D['day '+str(period)] < zeropoint]
        closed_below_zeropoint = closed_below_zeropoint[~closed_below_zeropoint['day '+str(period)].isin(closed_at_zeropoint['day '+str(period)])]
        closed_below_zeropoint = closed_below_zeropoint.dropna(how='all')
        
        xWinners = xDataForScatter3D[zDataForScatter3D['day '+str(period)] > zeropoint]
        xWinners = xWinners[~xWinners['day '+str(period)].isin(closed_at_zeropoint['day '+str(period)])]
        xWinners = xWinners.dropna(how='all')
        
        yLosers = yDataForScatter3D[zDataForScatter3D['day '+str(period)] < zeropoint]
        yLosers = yLosers[~yLosers['day '+str(period)].isin(closed_at_zeropoint['day '+str(period)])]
        yLosers = yLosers.dropna(how='all')
        
        for i in range(0,3):
            if i == 0:
                firstLetter = 'x'
            elif i == 1:
                firstLetter = 'y'
            elif i == 2:
                firstLetter = 'z'
            vars()[str(firstLetter)+'Winners'] = vars()[str(firstLetter)+'DataForScatter3D'][zDataForScatter3D['day '+str(period)] > zeropoint]
            vars()[str(firstLetter)+'Winners'] = vars()[str(firstLetter)+'Winners'][~vars()[str(firstLetter)+'Winners']['day '+str(period)].isin(closed_at_zeropoint['day '+str(period)])]
            vars()[str(firstLetter)+'Winners'] = vars()[str(firstLetter)+'Winners'].dropna(how='all')
        
        for i in range(0,3):
            if i == 0:
                firstLetter = 'x'
            elif i == 1:
                firstLetter = 'y'
            elif i == 2:
                firstLetter = 'z'
            vars()[str(firstLetter)+'Losers'] = vars()[str(firstLetter)+'DataForScatter3D'][zDataForScatter3D['day '+str(period)] < zeropoint]
            vars()[str(firstLetter)+'Losers'] = vars()[str(firstLetter)+'Losers'][~vars()[str(firstLetter)+'Losers']['day '+str(period)].isin(closed_at_zeropoint['day '+str(period)])]
            vars()[str(firstLetter)+'Losers'] = vars()[str(firstLetter)+'Losers'].dropna(how='all')
        #xWinners = vars()[firstLetter+type]
        #xDataForScatter3D = vars()[firstLetter+'DataForScatter3D']
        
        #vars()[ticker]['close']

for i in range(0,len(df)):
    df['x'].iloc[i] = xDataForScatter3D.loc[df['run'].iloc[i]][df['day'].iloc[i]]
    df['y'].iloc[i] = yDataForScatter3D.loc[df['run'].iloc[i]][df['day'].iloc[i]]
    df['z'].iloc[i] = zDataForScatter3D.loc[df['run'].iloc[i]][df['day'].iloc[i]]

print(df)



def update_graph(num):
    data=df[df['day']=='day '+str(num)]
    graph._offsets3d = (data.x, data.y, data.z)
    title.set_text('3D Test, day={}'.format(num))
    #print(data.z[num])
    if data.z[num] > 0:
        kleur = 'green'
        markeer = '^'
    elif data.z[num] < 0:
        kleur = 'red'
        markeer = 'v'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

kleur = 'blue'
markeer = 'o'

data=df[df['day']=='day 0']
graph = ax.scatter(data.x, data.y, data.z, color=kleur, marker=markeer)

#print(xDataForScatter3D['day 28'].min()[0])

ax.set_xlim(left=(1+displaymargin)*xDataForScatter3D['day '+str(period)].min()[0], right=zeropoint)
ax.set_xlabel("min")
ax.set_ylim(top=(1+displaymargin)*yDataForScatter3D['day '+str(period)].max()[0], bottom=zeropoint)
ax.set_ylabel("max")
ax.set_zlim(bottom=(1+displaymargin)*zDataForScatter3D['day '+str(period)].min()[0], top=(1+displaymargin)*zDataForScatter3D['day '+str(period)].max()[0])
ax.set_zlabel("close")

ani = matplotlib.animation.FuncAnimation(fig, update_graph, period-1, interval=500, blit=False)

plt.show()

"""
#%%

tic = time.time()
#def aniScatter3D():
# example helpers
for i in range(0,1):
        # Fixing random state for reproducibility
        np.random.seed(19680801)
        
        # Helper function to make an array of random numbers having shape (n, ) with each number distributed Uniform(vmin, vmax).
        def randrange(n, vmin, vmax):
            return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# load data
for r in range(0,1):
    xDataForScatter3D = pd.DataFrame(index=simresults.index, columns=[simresults.columns], data=[])
    yDataForScatter3D = pd.DataFrame(index=simresults.index, columns=[simresults.columns], data=[])
    zDataForScatter3D = pd.DataFrame(index=simresults.index, columns=[simresults.columns], data=[])
    for q in range(0,len(simresults.index)):
        for i in range(0,period+1):
            list = []
            for j in range(0,i+1):
                list.append(simresults['day '+str(j)]['run '+str(q)])
            xDataForScatter3D.iloc[q][i] = min(list)
            yDataForScatter3D.iloc[q][i] = max(list)
            zDataForScatter3D.iloc[q][i] = simresults.iloc[q][i]
    #print("zDataForScatter3D")
    #print(zDataForScatter3D)
    #print("yDataForScatter3D")
    #print(yDataForScatter3D)
    #print("zDataForScatter3D")
    #print(zDataForScatter3D)
    
    never_ran =  zDataForScatter3D[zDataForScatter3D['day '+str(period)] == zeropoint]
    actual_runs = zDataForScatter3D[~zDataForScatter3D['day '+str(period)].isin(never_ran['day '+str(period)])]
    #print(never_ran)
    
    # clean-up and separation
    for i in range(0,1):
        closed_at_zeropoint = zDataForScatter3D[zDataForScatter3D['day '+str(period)] == zeropoint]
        
        closed_above_zeropoint = zDataForScatter3D[zDataForScatter3D['day '+str(period)] > zeropoint]
        closed_above_zeropoint = closed_above_zeropoint[~closed_above_zeropoint['day '+str(period)].isin(closed_at_zeropoint['day '+str(period)])]
        closed_above_zeropoint = closed_above_zeropoint.dropna(how='all')
        
        closed_below_zeropoint = zDataForScatter3D[zDataForScatter3D['day '+str(period)] < zeropoint]
        closed_below_zeropoint = closed_below_zeropoint[~closed_below_zeropoint['day '+str(period)].isin(closed_at_zeropoint['day '+str(period)])]
        closed_below_zeropoint = closed_below_zeropoint.dropna(how='all')
        
        xWinners = xDataForScatter3D[zDataForScatter3D['day '+str(period)] > zeropoint]
        xWinners = xWinners[~xWinners['day '+str(period)].isin(closed_at_zeropoint['day '+str(period)])]
        xWinners = xWinners.dropna(how='all')
        
        yLosers = yDataForScatter3D[zDataForScatter3D['day '+str(period)] < zeropoint]
        yLosers = yLosers[~yLosers['day '+str(period)].isin(closed_at_zeropoint['day '+str(period)])]
        yLosers = yLosers.dropna(how='all')
        
        for i in range(0,3):
            if i == 0:
                firstLetter = 'x'
            elif i == 1:
                firstLetter = 'y'
            elif i == 2:
                firstLetter = 'z'
            vars()[str(firstLetter)+'Winners'] = vars()[str(firstLetter)+'DataForScatter3D'][zDataForScatter3D['day '+str(period)] > zeropoint]
            vars()[str(firstLetter)+'Winners'] = vars()[str(firstLetter)+'Winners'][~vars()[str(firstLetter)+'Winners']['day '+str(period)].isin(closed_at_zeropoint['day '+str(period)])]
            vars()[str(firstLetter)+'Winners'] = vars()[str(firstLetter)+'Winners'].dropna(how='all')
        
        for i in range(0,3):
            if i == 0:
                firstLetter = 'x'
            elif i == 1:
                firstLetter = 'y'
            elif i == 2:
                firstLetter = 'z'
            vars()[str(firstLetter)+'Losers'] = vars()[str(firstLetter)+'DataForScatter3D'][zDataForScatter3D['day '+str(period)] < zeropoint]
            vars()[str(firstLetter)+'Losers'] = vars()[str(firstLetter)+'Losers'][~vars()[str(firstLetter)+'Losers']['day '+str(period)].isin(closed_at_zeropoint['day '+str(period)])]
            vars()[str(firstLetter)+'Losers'] = vars()[str(firstLetter)+'Losers'].dropna(how='all')
        #xWinners = vars()[firstLetter+type]
        #xDataForScatter3D = vars()[firstLetter+'DataForScatter3D']
        
        #vars()[ticker]['close']

# set initial conditions
for i in range (0,1):
    n = 100
    zeropoint = 0
    displaymargin = 0.05
    xmin, xmax = (1+displaymargin)*xDataForScatter3D['day '+str(period)].min()[0], (1+displaymargin)*xDataForScatter3D['day '+str(period)].max()[0]
    ymin, ymax = (1+displaymargin)*yDataForScatter3D['day '+str(period)].min()[0], (1+displaymargin)*yDataForScatter3D['day '+str(period)].max()[0]
    zmin, zmax = (1+displaymargin)*zDataForScatter3D['day '+str(period)].min()[0], (1+displaymargin)*zDataForScatter3D['day '+str(period)].max()[0]    
"""
"""
# ZEROES
for i in range(0,1):
    # set lo/hi
    zLo, zHi = 0, 0
    # plot scatter
    for m, zlow, zhigh in [('o', zLo, zHi)]:
        x0 = xDataForScatter3D['day '+str(period)]
        y0 = yDataForScatter3D['day '+str(period)]
        z0 = zDataForScatter3D['day '+str(period)]
        ax.scatter(x0, y0, z0, marker=m, color='blue')
"""
"""
# WINNERS
for i in range(0,1):
    # set lo/hi
    zLo, zHi = zeropoint, zDataForScatter3D['day '+str(period)].max()[0]
    # plot scatter
    for m, zlow, zhigh in [('^', zeropoint, zHi)]:
        try:
            xW = xWinners['day '+str(period)]
            yW = yWinners['day '+str(period)]
            zW = zWinners['day '+str(period)]
            ax.scatter(xW, yW, zW, marker=m, color='green')
        except NameError:
            print("NameError")
# LOSERS
for i in range(0,1):
    # chg seed for example
    np.random.seed(14881488)
    # set lo/hi
    zLo, zHi = zDataForScatter3D['day '+str(period)].min()[0], zeropoint
    # plot scatter
    for m, zlow, zhigh in [('v', zLo, zeropoint)]:
        try:
            xL = xLosers['day '+str(period)]
            yL = yLosers['day '+str(period)]
            zL = zLosers['day '+str(period)]
            ax.scatter(xL, yL, zL, marker=m, color='red')
        except NameError:
            print("NameError")
# SET LIMITS, LABELS, AXES
for i in range(0,1):
        ax.set_xlim(left=xmin,right=xmax)
        ax.set_ylim(bottom=ymin,top=ymax)
        ax.set_zlim(bottom=zmin,top=zmax)
        
        ax.set_xlabel('Min')
        ax.set_ylabel('Max')
        ax.set_zlabel('Close')
        
        ax.xaxis._axinfo['juggled'] = (1,0,1)
        ax.yaxis._axinfo['juggled'] = (0,1,0)
        ax.zaxis._axinfo['juggled'] = (2,2,2)
        
        ax.xaxis._axinfo['juggled'] = (1,1,1)
        ax.yaxis._axinfo['juggled'] = (0,0,0)
        #ax.zaxis._axinfo['juggled'] = (2,2,2)
        

plt.show()

toc = time.time()
print("Finished aniScatter3D. ",f"{toc-tic:.3f} sec. elapsed")
"""
#%%


























"""
close = stats['close']
minhisto = stats['min']
maxhisto = stats['max']
minmaxhisto = []
for i in range(0,(len(minhisto))):
    if i < len(minhisto):
        minmaxhisto.append(minhisto[i])
for i in range(0,(len(maxhisto))):
    if i < len(maxhisto):
        minmaxhisto.append(maxhisto[i])
print(len(minmaxhisto))

minhisto = minhisto.sort()
maxhisto = maxhisto.sort()
minmaxhisto = minmaxhisto.sort()

labels = stats.index
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, minhisto, width, label='min')
rects2 = ax.bar(x + width/2, maxhisto, width, label='max')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()
"""



"""
x = []
for i in range(0,len(simresults.columns)):
    x.append(simresults.columns[i])
y = []
for i in range(0,len(simresults.columns)):
    y.append(simresults['day '+str(i)].iat[0])
y2 = []
for i in range(0,len(simresults.columns)):
    y2.append(simresults['day '+str(i)].iat[1])

a = plt.plot(simresults.columns,y)
a = plt.plot(simresults.columns,y2)
"""


"""
print(simresults['day 1'])
print(len(simresults.columns))
print(stats['min'].min())
print(stats['max'].max())

print(len(simresults['day 1']))
data = []
for i in range(0,len(simresults)):
    data.append(simresults['day 1'].iloc[i])
print(data)
"""


















































































toc = time.time()
print(f"{toc-tic:.3f} sec. elapsed")
