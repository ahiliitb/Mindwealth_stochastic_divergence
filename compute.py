
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
from loess.loess_1d import loess_1d
from collections import defaultdict
from collections import deque
from scipy import stats
from constant import *
from tqdm import tqdm
from finta import TA
from yahoo import *
from data import *
from config import *
from util import *
import matplotlib.dates as mdates
import plotly.graph_objects as go
import pandas_ta as ta
import pandas as pd
import numpy as np
import math
import copy
import os

# import matplotlib.pyplot as plt
# import plotly.express as px
# from sklearn.model_selection import train_test_split
# from functools import partial
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader,Dataset

"""
Core logic modules
"""

# Class to store endpoints of a line and handle required requests
class Linear:
    def __init__(self, x1, y1, x2, y2, startIndex = None, endIndex = None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.m = (y1 - y2) / (x1 - x2)
        self.c = y1 - self.m * x1

    # Get Y from X on the line
    def getY(self,x):
        return self.m * x + self.c

    # Check x value if it's in the line segment
    def isInRange(self,x):
        return self.x1 <= x and x <= self.x2

    # Get the tangent of the line
    def getAngle(self, inDegrees = True):
        tanTheta = self.m
        theta = math.atan(tanTheta)

        if not inDegrees:
            return theta
        else:
            return theta * 180 / math.pi

    # Get the length of the line
    def getMangnitude(self):
        return math.sqrt((self.y2 - self.y1) * (self.y2 - self.y1) + (self.x2 - self.x1) * (self.x2 - self.x1))

# Similar to Linear (Only used for Fib Ext)
class Linear2:
    def __init__(self, x1, y1, x2, y2, startIndex, endIndex):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.startIndex = startIndex
        self.endIndex = endIndex

    # Get the tangent of the line
    def getAngle(self, inDegress=True):
        cosTheta = (self.y2 - self.y1)/(math.sqrt((self.y2 - self.y1)*(self.y2 - self.y1) + (self.x2 - self.x1)*(self.x2 - self.x1)))
        theta = math.acos(cosTheta)
        if not inDegress:
            return theta
        else:
            return theta * 180/math.pi

    # Get the length of the line
    def getMangnitude(self):
        return math.sqrt((self.y2 - self.y1)*(self.y2 - self.y1) + (self.x2 - self.x1)*(self.x2 - self.x1))

# Management class for a time range
class Reigon:
    def __init__(self, s, e, c):
        self.start = s
        self.end = e
        self.class_ = c

# Linear scaler class
class Scaler:
    def __init__(self, series):
        self.s = series
        self.max = np.max(series)
        self.min = np.min(series)

    # Get the scaling ratio
    def getScaled(self):
        return np.subtract(self.s, self.min) / (self.max - self.min)
    
    # Return scalar as is
    def getUnscaled(self):
        return self.s

    # Return scaled array
    def getScaledArray(self, a):
        aa = np.asarray(a)
        return np.subtract(aa,self.min)/(self.max-self.min)

    # Return unscaled scalar
    def getUnscaledValue(self, v):
        u = v*(self.max-self.min) + self.min
        return u

    # Return scaled scalar
    def getScaledvalue(self, v):
        return (v- self.min)/(self.max-self.min)

# Returns the closest idex in list of turning points for a given index
def getClosestIndexinStock(Tp,list_):
    dist = 1e10
    point = None
    
    for t in list_:
        d = abs(t - Tp)
        if d <= dist:
            dist = d
            point = t
            
    return point

# Returns a list of linear lines that approximate the stock, which can then be
# used to judge for a rise or a fall
def getLinearModel(x, data, highs, stockHighs):
    linears = []
    i = 0
    while i + 1 < len(highs):
        linears.append(Linear2(x[highs[i]], data[highs[i]], x[highs[i + 1]], data[highs[i + 1]],
            getClosestIndexinStock(highs[i], stockHighs) , getClosestIndexinStock(highs[i + 1], stockHighs)))
        i += 1
    return linears

# Returns high data range
def getReigons(highs, data, stoch = False):
    reigons = []
    i = 15 if stoch else 0

    while i + 1 < len(highs):
        h1 = highs[i]
        h2 = highs[i+1]
        p1 = data[h1]
        p2 = data[h2]
        
        if p2 > p1 and (p2-p1)/p2 > 0.025:
            reigons.append(Reigon(h1, h2, 1))
        elif p2 < p1 and (p1-p2)/p1 > 0.025:
            reigons.append(Reigon(h1, h2, -1))
        else:
            reigons.append(Reigon(h1, h2, 0))
            
        i += 1
    return reigons

# Calculate merged regions
def getFinalReigons(reigons):
    rr = reigons.copy()
    i = 0
    
    while i + 1 < len(rr):
        r1 = rr[i]
        r2 = rr[i+1]

        if not r1.class_ == r2.class_:
            i += 1
        else:
            rr[i].end = r2.end
            rr.remove(r2)
            
    return rr

# Check the existence of overlaps
def isThereOverlap(s1, s2, e1, e2):
    if s1 > s2 > e1 and s1 > e2 > e1:  
        return True
    if s2 > s1 > e2 and s2 > e1 > e2:  
        return True
    if e1 > s2 > s1 and e1 > e2 > s1:  
        return True
    if e2 > s1 > s2 and e2 > e1 > s2:  
        return True
    else:
        return False

# Arrange fibonacci pivot HZ pairs
def removeConflicts(selectedLins1, data):
    selectedLins = []
    
    for s in selectedLins1:
        selectedLins.append(s)
        
    i = 0
    
    while i < len(selectedLins):
        s = selectedLins[i]
        startPrice = data.iloc[s.startIndex].high
        endPrice = data.iloc[s.endIndex].low
        range_ = startPrice - endPrice
        j = i + 1
        
        while j < len(selectedLins):
            s1 = selectedLins[j]
            startPrice1 = data.iloc[s1.startIndex].high
            endPrice1 = data.iloc[s1.endIndex].low
            range1 = startPrice1 - endPrice1
            
            if isThereOverlap(startPrice, startPrice1, endPrice, endPrice1):
                if range_ >= range1:
                    selectedLins.remove(selectedLins[j])
                    j = 0
                    i = -1
                else:
                    selectedLins.remove(selectedLins[i])
                    j = 0
                    i = -1
            j += 1
        i += 1
        
    return selectedLins

# Get local peak points using Zig-Zag algorithm
def get_zigzag(df, final_date):
    pivots = []

    series = df['Close']
    init_date = df.index[0]
    
    win_dur = timedelta(days = zigzag_window)
    pad_dur = timedelta(days = zigzag_padding)

    win_end_date = final_date - pad_dur
    win_start_date = win_end_date - win_dur

    while win_start_date >= init_date:
        if len(series[win_start_date:win_end_date]) > 1:
            max_idx = series[win_start_date:win_end_date].idxmax()
            min_idx = series[win_start_date:win_end_date].idxmin()

            if max_idx < min_idx:
                if len(pivots) > 0:
                    if pivots[-1][1] > 0:
                        pivots.append((min_idx, -1, series[min_idx]))
                        pivots.append((max_idx, 1, series[max_idx]))
                    elif pivots[-1][2] < series[min_idx]:
                        pivots.append((max_idx, 1, series[max_idx]))
                    else:
                        pivots[-1] = (min_idx, -1, series[min_idx])
                        pivots.append((max_idx, 1, series[max_idx]))
                else:
                    pivots.append((min_idx, -1, series[min_idx]))
                    pivots.append((max_idx, 1, series[max_idx]))
            else:
                if len(pivots) > 0:
                    if pivots[-1][1] < 0:
                        pivots.append((max_idx, 1, series[max_idx]))
                        pivots.append((min_idx, -1, series[min_idx]))
                    elif pivots[-1][2] > series[max_idx]:
                        pivots.append((min_idx, -1, series[min_idx]))
                    else:
                        pivots[-1] = (max_idx, 1, series[max_idx])
                        pivots.append((min_idx, -1, series[min_idx]))
                else:
                    pivots.append((max_idx, 1, series[max_idx]))
                    pivots.append((min_idx, -1, series[min_idx]))

        win_end_date -= win_dur
        win_start_date -= win_dur

    pivots = pivots[::-1]

    for _ in range(zigzag_merges):
        merged_pivots = merge_zigzag_pivots(pivots)		
        if len(merged_pivots) < 4: break

        pivots = merged_pivots

    res = pd.DataFrame(columns = ['Date', 'Sign', 'Close'])
    
    for idx, sign, v in pivots:
        r = {'Date': idx, 'Sign': sign, 'Close': v}
        res = pd.concat([res, pd.Series(r).to_frame().T], ignore_index = True)

    res.set_index('Date', inplace = True)
    return res

# Refine peak points by merging Zig-Zag peaks
def merge_zigzag_pivots(pivots):
    if len(pivots) < 3: return pivots	
    res, i = [], 0

    while i < len(pivots) - 3:
        res.append(pivots[i])

        if pivots[i + 3][0] - pivots[i][0] < timedelta(days = zigzag_merge_dur_limit):
            v = [pivots[j][2] for j in range(i, i + 4)]

            if min(v[0], v[3]) < min(v[1], v[2]) and max(v[0], v[3]) > max(v[1], v[2]):
                if zigzag_merge_val_limit * (max(v[0], v[3]) - min(v[0], v[3])) > (max(v[1], v[2]) - min(v[1], v[2])):
                    i += 3
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1

    for j in range(i, len(pivots)):
        res.append(pivots[j])

    return res

# Get recent downfall pivot pairs from Zig-Zag peak points
def get_recent_downfalls(zdf, count):
    res = []

    for i in range(len(zdf) - 1, 1, -1):
        row, prev = zdf.iloc[i], zdf.iloc[i - 1]		

        if row['Sign'] > 0: continue		
        hv, zv = prev['Close'], row['Close']

        if (hv - zv) < hv * fibo_pivot_diff_limit: continue
        res.append((prev.name, row.name))

        if len(res) == count: break

    return res[::-1]

def get_recent_downfalls_old(symbol, from_date, to_date, count):
    #data, highs, lows = getPointsGivenR(None, 1.2, startDate = from_date, endDate = to_date, interval = '1d', oldData = df)
    #days = len(data)
    #years = days / 255
    years = (datetime.strptime(to_date, YMD_FORMAT) - datetime.strptime(from_date, YMD_FORMAT)).days / 255

    if years < 3:
        interval_ = INTERVAL_DAILY
    elif 3 < years < 8:
        interval_ = INTERVAL_WEEKLY
    else:
        interval_ = INTERVAL_MONTHLY
    
    if interval_ == INTERVAL_WEEKLY:
        data, highs, lows, R = getPointsBest(symbol, startDate = from_date, endDate = to_date, getR = True, min_ = 0.5, interval = interval_)
    elif interval_ == INTERVAL_MONTHLY:
        data, highs, lows, R = getPointsBest(symbol, startDate = from_date, endDate = to_date, getR = True, min_ = 0.5, interval = interval_, max_ = 1)
    else:
        data, highs, lows, R = getPointsBest(symbol, startDate = from_date, endDate = to_date, getR = True, min_ = 0.4, interval = interval_)

    data = data.dropna()
 
    # get another copy to get the logData
    logData, highs, lows = getPointsGivenR(symbol, R, startDate = from_date, endDate = to_date, interval = interval_)
    logData = logData.dropna()
    logData = np.log10(logData)
    #logData = np.log10(data)
    
    # Scale the logData to be b/w 0 and 1 for x and y axis
    x = np.linspace(1, len(logData), len(logData))
    y = np.asarray(logData["close"])
    
    ScalerX = Scaler(x)
    ScalerY = Scaler(y)
    
    xs = ScalerX.getScaled()
    ys = ScalerY.getScaled()
 
    xo, yo = xs, ys
    hi, lo = highs, lows

    # Get a list of lines to approximate the loess smooothned data
    lins = getLinearModel(xo, yo, sorted(hi + lo), sorted(highs + lows))
    
    fallingLins = [l for l in lins if l.getAngle() > 90]

    # sorted array of biggest falls 
    sortedLins = sorted(fallingLins, key = lambda l: l.getMangnitude(), reverse = True)

    currentLins = [l for l in sortedLins if l.getMangnitude() > 0.05]
    relevantLins = currentLins

    SelectedLins__ =  relevantLins
    SelectedLins = removeConflicts(SelectedLins__, data)
    #SelectedLins = SelectedLins[-count:]

    res = []

    for s in SelectedLins:
        if data.iloc[s.startIndex].close > data.iloc[s.endIndex].close:
            hd = data.iloc[s.startIndex].name
            zd = data.iloc[s.endIndex].name
            res.append((hd, zd))

    return data, res

# Get Fibonacci extension levels from a given set of downfall pivot pairs
def get_fib_extensions(zdf, downfalls, merge_thres, limit_low, limit_high):
    all_levels = []

    for i, f in enumerate(downfalls):
        hd, zd = f
        hv, zv = zdf.loc[hd]['close'], zdf.loc[zd]['close']
        dv = hv - zv

        for j, l in enumerate(FIB_EXT_LEVELS):
            lv = zv + dv * l
            if lv < limit_low or lv > limit_high: continue

            all_levels.append((i, hd, zd, hv, zv, j, round(lv, 4)))

    all_levels.sort(key = lambda x: x[-1])
    res, flags = [], []

    for i, level in enumerate(all_levels):
        if i in flags: continue

        lv = level[-1]
        th = lv * merge_thres

        flags.append(i)		
        g = [level]

        for j in range(i + 1, len(all_levels)):
            if j in flags: continue
            v = all_levels[j][-1]

            if v - lv <= th:
                flags.append(j)
                g.append(all_levels[j])

                lv = v

        res.append(g)

    return res

# Compute behaviors of Fibonacci extension levels
def get_fib_ext_behaviors(df, extensions, cur_date, merge_thres):
    res = {}
    cur_price = df.iloc[-1]['close']

    for g in extensions:
        lv = (g[0][-1] + g[-1][-1]) / 2
        is_resist = True#(lv >= cur_price)

        behavior, pv, start_date = None, None, None

        #for d in df.loc[cur_date:].iloc:
        for d in df.iloc:
            #v = d.high if is_resist else d.low
            v = max(d.open, d.close) if is_resist else min(d.open, d.close)

            if pv is not None:
                if (pv < lv and v >= lv) or (pv > lv and v <= lv):
                    start_date = d.name
                    break

            pv = d.low if is_resist else d.high

        if start_date is not None:
            milestone_forward = FIB_BEHAVIOR_MILESTONE
            milestone_date = None

            while milestone_forward >= 5 and milestone_date is None:
                milestone_date = get_nearest_forward_date(df, start_date + timedelta(days = milestone_forward))
                milestone_forward //= 2

            if milestone_date is not None:
                mlv = df.loc[milestone_date]['close']
                thres = lv * merge_thres

                has_mid_up, has_mid_down = False, False

                for d in df.loc[df.loc[start_date:milestone_date].index[1:-1]].iloc:
                    if (d.close - lv) >= thres:
                        has_mid_up = True
                    elif (lv - d.close) >= thres:
                        has_mid_down = True

                if (mlv - lv) >= thres:
                    if has_mid_down:
                        behavior = 'Res_Semi_Break' if is_resist else 'Sup_Semi_Sup'
                    else:
                        behavior = 'Res_Break' if is_resist else 'Sup_Sup'
                elif (lv - mlv) >= thres:
                    if has_mid_up:
                        behavior = 'Res_Semi_Res' if is_resist else 'Sup_Semi_Break'
                    else:
                        behavior = 'Res_Res' if is_resist else 'Sup_Break'
                elif has_mid_up == has_mid_down:
                    end_date = get_nearest_forward_date(df, milestone_date + timedelta(days = milestone_forward))

                    if end_date is not None:
                        elv = df.loc[end_date]['close']

                        if (elv - lv) >= thres:
                            behavior = 'Res_Semi_Break' if is_resist else 'Sup_Semi_Sup'
                        elif (lv - elv) >= thres:
                            behavior = 'Res_Semi_Res' if is_resist else 'Sup_Semi_Break'
                        else:
                            behavior = 'Vibration'
                    else:
                        behavior = 'Vibration'
                elif has_mid_up:
                    behavior = 'Res_Break' if is_resist else 'Sup_Sup'
                else:
                    behavior = 'Res_Res' if is_resist else 'Sup_Break'

        res[g[0]] = behavior

    return res

# Generate table-format data for Fibonacci extension analysis
def analyze_fib_extension(df, extensions, behaviors, cur_date, pivot_number, merge_thres, interval, symbol):
    cols = ['ExtID', 'Level', 'Type', 'Width', 'Behavior', 'Description', ' ']
    res = pd.DataFrame(columns = cols)
    
    cur_price = df.iloc[-1]['close']
    i = 0

    for g in extensions:
        lv = (g[0][-1] + g[-1][-1]) / 2
        b = behaviors[g[0]]
        i += 1

        record = [
            i,
            '${:.4f}'.format(lv),
            'Resistance' if lv >= cur_price else 'Support',
            '{:.2f}%'.format(100 * (g[-1][-1] - g[0][-1]) / g[0][-1]) if len(g) > 1 else '',
            FIB_EXT_MARKERS[b][-1] if b is not None else '',
            ' & '.join(['{:.1f}% of {:.4f}-{:.4f}'.format(FIB_EXT_LEVELS[j] * 100, zv, hv) for _, _, _, hv, zv, j, _ in g]),
            ''
        ]
        res = pd.concat([res, pd.Series(dict(zip(cols, record))).to_frame().T], ignore_index = True)

    res = pd.concat([res, pd.Series({}).to_frame().T], ignore_index = True)
    
    res = pd.concat([res, pd.Series({
        'Level': 'Ticker: ' + symbol,
        'Type': 'Current Date:',
        'Width': change_date_format(cur_date, YMD_FORMAT, DBY_FORMAT),
        'Behavior': 'Current Price:',
        'Description': '${:.4f}'.format(cur_price)
    }).to_frame().T], ignore_index = True)

    res = pd.concat([res, pd.Series({
        'Level': 'From: {}'.format(df.index[0].strftime(DBY_FORMAT)),
        'Type': 'To: {}'.format(df.index[-1].strftime(DBY_FORMAT)),
        'Width': 'By: ' + interval,
        'Behavior': 'Merge: {:.1f}%'.format(2 * merge_thres * 100),
        #'Description': 'Recent Pivots: {}'.format(pivot_number)
    }).to_frame().T], ignore_index = True)

    res = pd.concat([res, pd.Series({}).to_frame().T], ignore_index = True)
    return res

def backtest_fib_extension(df, interval, pivot_number, merge_thres, symbol, from_date, to_date):
    cols = ['TransID', 'Position', 'EnterDate', 'EnterPrice', 'ExitDate', 'ExitPrice', 'Offset', 'Profit', 'CumProfit', 'X', ' ']

    enter_date, position = None, None
    trans_count, match_count, cum_profit = 0, 0, 0
    fcounter = 0

    signs = deque(maxlen = 3 if interval == INTERVAL_DAILY else 1)
    fqueue = deque(maxlen = 4)
    res = pd.DataFrame(columns = cols)

    
    has_signal = False

    for cur_date in tqdm(list(df.index), desc = 'backtesting', colour = 'red'):
        cur_candle = df.loc[cur_date]
        signs.append(np.sign(cur_candle['Close'] - cur_candle['Open']))

        if enter_date is not None and (cur_date - enter_date).days < MIN_FIB_EXT_TRANS_DUR: continue

        if signs.count(1) == len(signs):
            cur_sign = 1
        elif signs.count(-1) == len(signs):
            cur_sign = -1
        else:
            cur_sign = 0

        if cur_sign == 0: continue
        if position == cur_sign: continue

        
        if has_signal==False: # checking if we can enter
            min_cur_price = min(cur_candle['Close'], cur_candle['Open'])
            max_cur_price = max(cur_candle['Close'], cur_candle['Open'])

            #zdf = get_zigzag(df, cur_date)
            zdf,downfalls = get_recent_downfalls_old(symbol, from_date, cur_date.strftime(YMD_FORMAT), pivot_number)
            
            if zdf.empty: continue

            #get_recent_downfalls(zdf, pivot_number)
    
            zdf = zdf.rename(columns = {"Open": "open", "High": "high", "Low": "low", "Volume": "volume", "Close": "close"})
            
            extensions = get_fib_extensions(zdf, downfalls, get_safe_num(merge_thres), cur_candle['Close'] * 2 * 0.05, cur_candle['Close'] * 2)
                  
            if len(extensions)==0 : continue
                  
            behaviors = get_fib_ext_behaviors(zdf, extensions, cur_date, get_safe_num(merge_thres)) # Compute behaviors of each extension level
            for g in extensions:
                lv = (g[0][-1] + g[-1][-1]) / 2
                beh = behaviors[g[0]]
                if min_cur_price <= lv and lv <= max_cur_price: # condition for crossing ext level
                    if cur_sign==1 and beh in ['Res_Break','Res_Semi_Break']:
                        has_signal = True
                        enter_date = cur_date
                        position=1
                    elif cur_sign==1 and beh in ['Res_Res','Res_Semi_Res']:
                        has_signal = True
                        enter_date = cur_date
                        position=-1
                    elif cur_sign==-1 and beh in ['Sup_Semi_Break','Sup_Break']:
                        has_signal = True
                        enter_date = cur_date
                        position=-1
                    elif cur_sign==-1 and beh in ['Sup_Sup','Sup_Semi_Sup']:
                        has_signal = True
                        position=1
                        enter_date = cur_date
                    break
        else:       
            #trade is on and check if we want to exit
            if position is None:
                position = cur_sign
                enter_date = cur_date
            else:
                price_offset = cur_candle['Close'] - df.loc[enter_date]['Close']
                true_sign = np.sign(price_offset)				

                if true_sign == position:
                    match_count += 1
                else:
                    fcounter += 1

                if true_sign == position or fcounter % 4 == 3:
                    profit = position * price_offset / df.loc[enter_date]['Close']
                    cum_profit += profit			
                    trans_count += 1

                    record = [
                        trans_count,
                        'Long' if position > 0 else 'Short',
                        enter_date.strftime(DBY_FORMAT),
                        '${:.4f}'.format(df.loc[enter_date]['Close']),
                        cur_date.strftime(DBY_FORMAT),
                        '{:.4f}$'.format(cur_candle['Close']),
                        '{:.2f}%'.format(100 * price_offset / df.loc[enter_date]['Close']),
                        '{:.4f}%'.format(100 * profit),
                        '{:.4f}%'.format(100 * cum_profit),
                        'T' if true_sign == position else 'F',
                        ' '
                    ]
                    res = pd.concat([res, pd.Series(dict(zip(cols, record))).to_frame().T], ignore_index = True)

                enter_date, position = None, None

    success_rate = (match_count / trans_count) if trans_count != 0 else 0
    res = pd.concat([res, pd.Series({}).to_frame().T], ignore_index = True)
    
    res = pd.concat([res, pd.Series({
        'TransID': 'Ticker:',
        'Position': symbol,
        'EnterDate': 'From: {}'.format(df.index[0].strftime(DBY_FORMAT)),
        'EnterPrice': 'To: {}'.format(df.index[-1].strftime(DBY_FORMAT)),
        'ExitDate': 'By: ' + interval,
        #'ExitPrice': 'Recent Pivots: {}'.format(pivot_number),
        'ExitPrice': 'Merge: {:.1f}%'.format(2 * merge_thres * 100)
    }).to_frame().T], ignore_index = True)

    res = pd.concat([res, pd.Series({
        'EnterDate': 'Success Rate:',
        'EnterPrice': '{:.1f}%'.format(success_rate * 100),
        'ExitDate': 'Cumulative Profit:',
        'ExitPrice': '{:.1f}%'.format(cum_profit * 100)
    }).to_frame().T], ignore_index = True)

    res = pd.concat([res, pd.Series({}).to_frame().T], ignore_index = True)
    return res, success_rate, cum_profit


def get_rank_info():
    # import yfinance as yf
    # data1 = yf.download('AAPL', start='2005-01-01', end='2023-12-15', interval='1d')
    
    # ## Reading data of Apple stock from a comma-separated .txt file into a pandas dataframe and removing the header 
    # apple_stock=data1.dropna()
    
    # ## Separating the date and time using str.split() for better interpretation
    # apple_stock.columns = ['Open','High','Low','Close','Adj Close','Volume']
    # ## Dropping the original column and retaining the newly created columns
    # apple_stock = apple_stock.drop('Adj Close',axis=1)
    
    # ## As we are interested in predicting stock prices for the next 5 days we only require day-wise data and not minute wise data
    # ## Thus, we group the data by 'Date' setting values for all the columns appropriately
    # apple_stock = apple_stock.groupby('Date').agg({
    #         'Close' : 'last',
    #         'Open' : 'first',
    #         'High' : 'max',
    #         'Low' : 'min',
    #         'Volume' : 'sum'
    # }).reset_index()
    # ## Creating features/indicators that are suitable for financial data and aid in stock price predictions as they help in determining the trends
    # ## Creating a separate column for stochastic oscillator given by (C-L)/(H-L)
    # apple_stock['Stochastic Oscillator'] = (apple_stock['Close'] - apple_stock['Low'])/(apple_stock['High']-apple_stock['Low'])
    # ## Defining absolute returns as c_t - c_t-1
    # apple_stock['Absolute Returns'] = apple_stock['Close'] - apple_stock['Close'].shift(1)
    # ## Normalizing opening, closing, high and low prices using the previous days' closing price
    # apple_stock['Closing Price Normalized'] = apple_stock['Close']/apple_stock['Close'].shift(1)
    # apple_stock['Opening Price Normalized'] = apple_stock['Open']/apple_stock['Close'].shift(1)
    # apple_stock['High Value Normalized'] = apple_stock['High']/apple_stock['Close'].shift(1)
    # apple_stock['Low Value Normalized'] = apple_stock['Low']/apple_stock['Close'].shift(1)
    # ## Normalizing volume of stocks traded by a 5-day rolling mean of the volume
    # apple_stock['Volume Normalized'] =  apple_stock['Volume']/apple_stock['Volume'].shift(1).rolling(window=5).mean()
    # ## Defining volatility by the variance of volume of stocks traded over a window of 9 days
    # apple_stock['Volatility'] = apple_stock['Volume'].rolling(window=9).var()/1e16
    # ## Removing unnecessary rows and resetting the index
    # apple_stock.dropna(inplace = True)
    # apple_stock.reset_index(drop = True, inplace = True)
    # ## Defining MACD () as  5 day EMA - 9 day EMA
    # ## Defining the periods over which EMA is to be calculated
    # period1 = 9
    # period2 = 5
    # # Calculate the smoothing factor (alpha)
    # alpha1 = 2 / (period1 + 1)
    # alpha2 = 2/(period2 +1 )
    # # Calculate 9-day EMA and 5-day EMA using the pandas `ewm` method
    # EMA9day = apple_stock['Close'].ewm(span=period1, adjust=False).mean()
    # EMA5day = apple_stock['Close'].ewm(span=period2, adjust=False).mean()
    # apple_stock['MACD'] = EMA5day - EMA9day
    
    # ## Defining Gains/Losses for a particular day depending on whether O>C or C>O
    # apple_stock['Gains'] = (apple_stock['Close'] - apple_stock['Open'])*100/apple_stock['Open']
    # apple_stock['Gains'] = apple_stock['Gains'].apply(lambda x: max(0, x))
    # apple_stock['Losses'] = (apple_stock['Open'] - apple_stock['Close'])*100/apple_stock['Open']
    # apple_stock['Losses'] = apple_stock['Losses'].apply(lambda x: max(0, x))
    
    # ## Computing the Relative Strength Index (RSI) using the gains/losses which is dependent on average gains and average losses
    # rsi_data = []
    # ## Defining epsilon to avoid nan related issues (small value)
    # eps = 1e-8
    # for i in range(9,len(apple_stock)):
    #     ## Calculating the RSI gains for the last 9 days
    #     rsi_gains = np.array(apple_stock.iloc[i-9:i]['Gains'])
    #     ## Calculating average positive gains over the last 9 days
    #     average_gains = np.mean(rsi_gains[rsi_gains > 0])
    #     average_gains = 0 if np.isnan(average_gains) else average_gains
    #     ## Calculating RSI losses for the last 9 days
    #     rsi_losses = np.array(apple_stock.iloc[i-9:i]['Losses'])
    #     average_losses = np.mean(rsi_losses[rsi_losses > 0])
    #     average_losses = 0 if np.isnan(average_losses) else average_losses
    #     ## Computing the RSI index
    #     den = 1+ average_gains/(average_losses + eps)
    #     rsi_data.append(100-(100/den))
    
    
    # #Removing the first 9 rows, due to rolling mean
    # apple_stock = apple_stock.iloc[9:]
    # apple_stock['RSI'] = rsi_data
    # apple_stock.reset_index(inplace = True, drop = True)
    # ## Displaying the final dataset with all features
    # import pandas as pd
    # apple_stock['Sine'] = np.sin(2*np.pi/20*pd.DatetimeIndex(data = apple_stock['Date'], yearfirst = True).day)
    # apple_stock['Cosine'] = np.cos(2*np.pi/20*pd.DatetimeIndex(data = apple_stock['Date'], yearfirst = True).day)
    # apple_stock = apple_stock.drop('Date',axis=1)
    
    # ## Defining a Sequencer which converts the data into a format suitable for training
    # class StockDataset(Dataset):
    #     def __init__(self,data,sequence_length,prediction_length):
    #         ## Initializing the dataframe and the window length i.e. number of previous days which will be used at a time to predict
    #         ## today's closing price
    #         self.data = data
    #         self.sequence_length = sequence_length
    #         self.prediction_length = prediction_length
            
    #     def __len__(self):
    #         ## As it picks indices randomly from [0,len], we keep len =  len(df) - seq_len which denotes the last index which can be
    #         ## used to create a batch as we need seq_len rows ahead of it
    #         return len(self.data) - self.sequence_length - self.prediction_length
            
    #     def __getitem__(self,index):
    #         ## Slicing the dataframe from input index to input index + seq_len to get the input data 
    #         input_data = self.data[index : index + self.sequence_length]
    #         input_list = input_data.values.tolist()
    #         input = torch.Tensor(input_list)
            
    #         ## Returning the closing prices of next day as the output for each day in the input
    #         ## Converting both the input and output to tensors before returning
    #         output = self.data.loc[index + self.sequence_length : index + self.sequence_length + self.prediction_length-1, 'Closing Price Normalized'].values.tolist()
    #         output = torch.Tensor(output)
             
    #         return input,output

    # input_features = ['Closing Price Normalized', 'Opening Price Normalized','High Value Normalized','Low Value Normalized','Volume Normalized',
    #              'Stochastic Oscillator', 'Absolute Returns','Volatility','MACD','RSI','Sine','Cosine']
    # df_app = apple_stock[input_features].copy()
    # sequence_length = 12
    # prediction_length = 5
    # sequenced_data = StockDataset(df_app,sequence_length,prediction_length)

    # #Splitting the data to 80% Training, 10% Validaiton and 10% Testing
    # split=0.8
    # #Splitting the indices of the sequences, so as to maintain order of time series
    # indices = list(range(len(sequenced_data)))
    
    # #splitting the indices according to the decided split 
    # train_indices, test_indices = train_test_split(indices, train_size=split, shuffle=False)
    # val_indices, test_indices = train_test_split(test_indices, train_size=0.5, shuffle=False)
    
    # # Create the training , validation and test datasets
    # train_dataset = torch.utils.data.Subset(sequenced_data, train_indices)
    # val_dataset= torch.utils.data.Subset(sequenced_data, val_indices)
    # test_dataset = torch.utils.data.Subset(sequenced_data, test_indices)
    # train_size=len(train_dataset)
    # test_size=len(val_dataset)
    # val_size=len(test_dataset)

    # train_dataloader=DataLoader(train_dataset,batch_size=16,shuffle=False)
    # val_dataloader=DataLoader(val_dataset,batch_size=16,shuffle=False)
    # test_dataloader=DataLoader(test_dataset,batch_size=16,shuffle=False)
    # entire_dataloader=DataLoader(sequenced_data,batch_size=16,shuffle=False)

    # ## Defining the class for diffusion module which keeps on adding gaussian noise with a fixed variance schedule to both input as wel as output
    # class DiffusionProcess(nn.Module):
    #     def __init__(self, num_diff_steps, vae, beta_start, beta_end, scale):
    #         super().__init__()
    #         to_torch = partial(torch.tensor, dtype = torch.float32)
    #         ## Initializing variables like number of time stamps, the Hierarchial VAE to make predictions, start and end values
    #         ## for beta, which governs the variance schedule
    #         self.num_diff_steps = num_diff_steps
    #         self.vae = vae
    #         self.beta_start = beta_start
    #         self.beta_end = beta_end
    #         ## Defining a linearly varying variance schedule for the conditional noise at every timestamp 
    #         betas = np.linspace(beta_start, beta_end,  num_diff_steps, dtype = np.float32)
            
    #         ## Performing reparametrization to calculate output at time t directly using x_start
    #         alphas = 1 - betas
    #         alphas_target = 1 - betas*scale
    #         ## Computing the cumulative product for the input as well as output noise schedule
    #         alphas_cumprod = np.cumprod(alphas, axis = 0)
    #         alphas_target_cumprod = np.cumprod(alphas_target, axis = 0)
            
    #         ## Converting all the computed quantities to tensors and detaching them from the computation graph (setting requires_grad to False)
    #         betas = torch.tensor(betas, requires_grad = False)
    #         alphas_cumprod = torch.tensor(alphas_cumprod, requires_grad = False)
    #         alphas_target_cumprod = torch.tensor(alphas_target_cumprod, requires_grad = False)
            
    #         ## Computing scaling factors for mean and variance respectively
    #         self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).detach().requires_grad_(False)
    #         self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).detach().requires_grad_(False)
    #         self.sqrt_alphas_target_cumprod = torch.sqrt(alphas_target_cumprod).detach().requires_grad_(False)
    #         self.sqrt_one_minus_alphas_target_cumprod = torch.sqrt(1 - alphas_target_cumprod).detach().requires_grad_(False)
            
    #     ## Defining the forward pass
    #     def diffuse(self, x_start, y_target, timestamp):
    #         ## Generating a random noise vector sampled from a standard normal of the size x_start and y_target respectively
    #         noise = torch.randn_like(x_start)
    #         noise_target = torch.randn_like(y_target)
            
    #         ## Computing the sampled value using the reparametrization trick and using that to calculate x_noisy and y_noisy
    #         x_noisy = self.sqrt_alphas_cumprod[timestamp - 1]*x_start + self.sqrt_one_minus_alphas_cumprod[timestamp - 1]*noise
    #         y_noisy = self.sqrt_alphas_target_cumprod[timestamp - 1]*y_target + self.sqrt_one_minus_alphas_target_cumprod[timestamp - 1]*noise_target
        
    #         ## Performing a forward pass through the Hierarchial VAE to generate noisy predictions
    #         output = self.vae(x_noisy)
    #         return output, y_noisy

    # import torch.nn.init as init
    
    # ## Initializing weights using Xavier Initialization
    # def init_weights(layer):
    #     init.xavier_uniform_(layer.weight)
    #     layer_name=layer._class.name_
    #     if layer.find("Conv")!=-1:
    #         layer.weight.data.normal_(0.0,0.25)
    #     elif layer.find("BatchNorm")!=-1:
    #         layer.weight.data.normal(1.00,0.25)
    #         layer.bias.data.fill_(0.00)
    
    # ## Defining a custom Conv2D class with the padding size such that the input size and output size remain the same
    # class Conv2D(nn.Module):
    #     def __init__(self,input_dim,output_dim,kernel_size,stride):
    #         super(Conv2D,self).__init__()
    #         ## Required padding size = kernel_size - 1/2
    #         padding=int((kernel_size-1)/2)
    #         self.layer=nn.Conv2d(input_dim,output_dim,kernel_size,stride=stride,padding=padding,bias=True)
    #     ## Performing the forward pass
    #     def forward(self,input):
    #         return self.layer(input)
    
    # ## Defining the module for Swish Activation or Sigmoid Linear Unit
    # class Swish(nn.Module):
    #     def __init__(self):
    #         super(Swish,self).__init__()
    #         self.layer=nn.SiLU()
    #     def forward(self,input):
    #         return self.layer(input)
    
    # ## Performing Batch Normalization by inherting it from torch.nn
    # class BatchNorm(nn.Module):
    #     def __init__(self,batch_dim,size):
    #         super(BatchNorm,self).__init__()
    #         ## Equivalent to BatchNorm as first dimension is batch_size
    #         self.layer=nn.LayerNorm([batch_dim,size,size])
    
    #     def forward(self,input):
    #         return self.layer(input)
            
    # class SE(nn.Module):
    #     def __init__(self,channels_in,channels_out):
    #         super(SE,self).__init__()
    #         ## Defining number of units to be compressed into
    #         num_hidden=max(channels_out//16,4)
            
    #         ## Defining the network which compresses and expands to focus on features rather than noise
    #         ## 2 networks req as 2 different input output dimensions are present in the Hierarchial VAE
    #         self.se=nn.Sequential(nn.Linear(1,num_hidden),nn.ReLU(inplace=True),
    #                                 nn.Linear(num_hidden, 144), nn.Sigmoid())
    #         self.se2=nn.Sequential(nn.Linear(1,num_hidden),nn.ReLU(inplace=True),
    #                                 nn.Linear(num_hidden, 36), nn.Sigmoid())
    
    #     def forward(self,input):
                
    #         ## Getting compressed vector
    #         se=torch.mean(input,dim=[1,2])
    #         ## Flattening out the layer
    #         se=se.view(se.size(0),-1)
            
    #         if(input.size(1)==12):
    #             se=self.se(se)
    #             se=se.view(se.size(0),12,12)
    #         else:
    #             se=self.se2(se)
    #             se=se.view(se.size(0),6,6)
    #         ## Returning appropriate mapped feature
    #         return input*se
    
    # ## Performing pooling for downsampling using nn.AvgPool2D and using a kernel of size 2 to ensure that output size is halved
    # class Pooling(nn.Module):
    #     def __init__(self):
    #         super(Pooling,self).__init__()
    #         ## Using a 2x2 kernel and a stride of 2 in both directions
    #         self.mean_pool = nn.AvgPool2d(kernel_size=(2, 2),padding=0,stride=(2,2))
    #     def forward(self,input):
    #         return self.mean_pool(input)
    
    # ## Defining a class to compute square of a quantity
    # class Square(nn.Module):
    #     def __init__(self):
    #         super(Square,self).__init__()
    #         pass
    #     def forward(self,input):
    #         return input**2

    # ## Defining the encoder block to be used in Hierarchial VAE to convert to input into its latent space representation
    # class Encoder_Block(nn.Module):
    #     def __init__(self,input_dim,size,output_dim):
    #         super().__init__()
    #         ## Initializing the in and out dimensions of the conv layers and SE block
    #         self.input_dim = input_dim
    #         self.output_dim = output_dim
    #         self.size = size
    #         ## Defining the encoder layers i.e 2 Conv2D layers followed by Batch Normalization, a Conv2D layer of kernel size 1 and Squeeze and excitation at the end
    #         self.seq=nn.Sequential(Conv2D(input_dim,input_dim,kernel_size=5,stride=1),
    #                                Conv2D(input_dim,input_dim,kernel_size=1,stride=1),
    #                                BatchNorm(input_dim,size),Swish(),
    #                                Conv2D(input_dim,input_dim,kernel_size=3,stride=1),
    #                                SE(input_dim,output_dim))
    #     def forward(self,input):
    #         ## Computing the final output as the sum of scaled encoded output and original input (result of skip connection i.e. residual encoder)
    #         return input +0.1*self.seq(input)

    # ## Defining the decoder to be used in Hierarchial VAE to convert from latent space representations to noisy outputs 
    # class Decoder_Block(nn.Module):
    #     def __init__(self,dim,size,output_dim):
    #         super().__init__()
    #         ## Defining the decoder net which comprises of Conv2D layers, BatchNorm and SE Blocks
    #         ## We ensure that the dimension of the input and output stays the same at all instants as down/up sampling is done in a separate block 
    #         self.seq = nn.Sequential(
    #             BatchNorm(dim,size),
    #             Conv2D(dim,dim,kernel_size=1,stride=1),
    #             BatchNorm(dim,size), Swish(),
    #             Conv2D(dim,dim, kernel_size=5, stride=1),
    #             BatchNorm(dim,size), Swish(),
    #             Conv2D(dim, dim, kernel_size=1, stride = 1),
    #             BatchNorm(dim,size),
    #             ## SE Block just compresses and expands which allows it to ignore noise and focus on actual indicators
    #             SE(dim,output_dim))
    #     ## Computing the final output similar to encoder taking into account the skip connection
    #     def forward(self,input):
    #         return input+0.1*self.seq(input)

    # ## Defining the class for the Hierarchial VAE which takes as input various hyperparameters and the classes for encoder and decoder blocks
    # class HierarchialVAE(nn.Module):
    #     def __init__(self, Encoder_Block, Decoder_Block, latent_dim2 = 5, latent_dim1 = 2, feature_size2 = 36, 
    #                  feature_size1 = 9, hidden_size = 2, pred_length = 5, num_features = 12, seq_length = 12, batch_size = 16):
    #         super().__init__()
    #         ## Initializing the encoder at the beginning when x_start has 12 features
    #         self.Encoder1 = Encoder_Block(input_dim = batch_size, output_dim = batch_size, size = 12)
    #         ## Initializing the encoder reqd after downsampling when input has 6 features 
    #         self.Encoder2 = Encoder_Block(input_dim = batch_size, output_dim = batch_size, size = 6)
    #         ## Initializing the decoder reqd after upsampling which gives y_noisy at the output
    #         self.Decoder1 = Decoder_Block(dim = batch_size,size = 12,output_dim = batch_size)
    #         ## Initializing the first decoder which obtains an input of size batchx6x6
    #         self.Decoder2 = Decoder_Block(dim = batch_size,size = 6,output_dim = batch_size)
            
    #         ## Initializing dimensions of both latent vectors, feature size of both the intermediate feature maps 
    #         self.latent_dim2 = latent_dim2
    #         self.latent_dim1 = latent_dim1
    #         self.feature_size2 = feature_size2
    #         self.feature_size1 = feature_size1
    #         ## Initializing the initial hidden state with a tensor of zeros with dimension equal to that of the final latent vector
    #         self.hidden_size = hidden_size
    #         self.hidden_state = torch.zeros(self.latent_dim1)
    #         ## Initializing batch_size
    #         self.batch_size= batch_size
            
    #         ## Defining the upsampling blocks required at 2 different stages in the entire network (2 networks reqd as size of input feature map varies throughout the network)
    #         self.upsample1 = nn.Upsample(size=(6, 6), mode='bilinear', align_corners=False)
    #         self.upsample2 = nn.Upsample(size=(12, 12), mode='bilinear', align_corners=False)
    #         ## Defining linear layers that map flattened feature maps to latent space dimensions and vice versa
    #         self.fc12 = nn.Linear(feature_size2,2*latent_dim2)
    #         self.fc11 = nn.Linear(feature_size1,2*latent_dim1)
    #         self.fc22 = nn.Linear(latent_dim2, feature_size2)
    #         self.fc21 = nn.Linear(latent_dim1, feature_size1)
    #         ## Defining pooling layer for downsampling
    #         self.mean_pool = nn.AvgPool2d(kernel_size=(2, 2),padding=0,stride=(2,2))
    #         ## The final linear layer which maps the VAE output to the output dimension
    #         self.fc_final = nn.Linear(num_features*seq_length, pred_length)
            
        
    #     def forward(self,x_start):
    #         ## We pass the input through two encoder blocks followed by pooling which reduces the feature map size to 6x6
    #         out = self.Encoder1(x_start)
    #         out = self.Encoder1(out)
    #         out = self.mean_pool(out)
    #         ## Reshaping the feature map and storing as it is required for sampling 
    #         feature_map2 = out.view(out.size(0),6,6)
    #         ## Encoding and Pooling the output once again which reduces the feature map size to 3x3 
    #         out = self.Encoder2(out)
    #         out = self.mean_pool(out)
    #         ## Flattening the final feature map and passing it through the linear layer which maps it to a latent vector of 
    #         ## dimension 4 (latent vector is dimension 2, but we predict both the mean and variances)
    #         feature_map1 = out.view(out.size(0),-1)
    #         z1 = self.fc11(feature_map1)
    #         ## Randomly sampling noise from a standard normal
    #         noise1 = torch.randn((out.size(0),self.latent_dim1))
    #         ## Applying the reparametrization trick to get the sampled value
    #         sampled_z1 = self.reparametrize(noise1,z1)
    #         ## Adding the initial hidden vector to the sampled output and converting it back to 3x3 feature map using a linear layer
    #         out = sampled_z1 + self.hidden_state
    #         out = self.fc21(out)
    #         out = out.view(out.size(0),3,3)
    #         ## Upsampling to dimension 6x6
    #         out = self.upsample1(out.unsqueeze(0)).squeeze(0)
    #         ## Passing it through the decoder and combining it with feature map 2 to sample from the 2nd latent vector
    #         out = self.Decoder2(out)
    #         ## Maps to a dimension of 10 after flattening the vector which means means and variances of a latent vector of dim = 5
    #         z_decoder = (feature_map2 + out).view(out.size(0),-1)
    #         z2 = self.fc12(z_decoder)
    #         ## In a similar fashion, we get the sampled value from z2
    #         noise2 = torch.randn((out.size(0),self.latent_dim2))
    #         sampled_z2 = self.reparametrize(noise2,z2)
    #         ## We convert it back to dim = 36 using a linear layer followed by reshaping it to 6x6
    #         z2_upsampled = self.fc22(sampled_z2).view(out.size(0),6,6)
    #         ## Upsampling to the original dimension of 12x12
    #         out = out + z2_upsampled
    #         out = self.upsample2(out.unsqueeze(0)).squeeze(0)
    #         out = self.Decoder1(out)
    #         out = self.Decoder1(out)
    #         ## Passing it through the final linear layer to map it to the shape of output
    #         out = self.fc_final(out.view(out.size(0),-1))
    #         return out
            
    #     def reparametrize(self,noise,z):
    #         ## Getting the batch_size
    #         zsize=int(z.size(1))
    #         ## Initializing tensors for mean and variances
    #         sampled_z = torch.zeros((noise.size(0),zsize//2))
    #         mu=torch.zeros((noise.size(0),zsize//2))
    #         sig=torch.zeros((noise.size(0),zsize//2))
    #         for i in range(0,zsize//2):
    #             mu[:,i]=z[:,i]
    #             sig[:,i]=z[:,zsize//2+i]
    #             ## Computing the sampled value
    #             sampled_z[:,i]=mu[:,i] + noise[:,i]*sig[:,i]
    #         return sampled_z

    # ## Defining the network for denoising score matching 
    # class Denoise_net(nn.Module):
    #     def __init__(self,in_channels,dim,size,number=5):
    #         super().__init__()
    #         ## 2*number is number of diffusion samples used for denoise calculation
    #         ## Initializing the input dimension (actually prediction length in this case)
    #         hw = size
    #         self.dim=dim
    #         ## Number of input channels (batched mapping)
    #         self.channels=in_channels
    #         ## Defining the network for energy calculation
    #         self.conv=Conv2D(self.channels,dim,3,1)
    #         self.conv1=Conv2D(dim,dim,3,1)
    #         self.relu1=nn.ELU()
    #         self.pool1=Pooling()
    #         self.conv2=Conv2D(dim,dim,3,1)
    #         self.relu2=nn.ELU()
    #         self.conv3=Conv2D(dim,dim,3,1)
    #         self.relu3=nn.ELU()
    #         ## Getting interaction energy and self energy component field terms
    #         self.f1=nn.Linear((int(hw/2)*number),1)
    #         self.f2=nn.Linear((int(hw/2)*number),1)
    #         self.fq=nn.Linear((int(hw/2)*number),1)
    #         self.square=Square()
        
    #     def forward(self,input):
    #         output=self.conv(input)
    #         output1=self.conv1(output)
    #         output2=self.relu1(output1)
    #         ## Resnet type output computation for stable gradient flow
    #         output2=output2+output1
    #         ## Pooling to increase the receptive field 
    #         output3=self.pool1(output2)
    #         output4=self.conv2(output3)
    #         output5=self.relu2(output4)
    #         output5=output5+output4
    #         output7=self.conv3(output5)
    #         output8=self.relu3(output7)
    #         l1=self.f1(output8.view(input.size(0),-1))
    #         l2=self.f2(output8.view(input.size(0),-1))
    #         lq=self.fq(self.square(output8.view(input.size(0),-1)))
    #         ## Getting gradient of energy term per sample (gradient of energy term is what we are concerned with)
    #         out=l1*l2 +lq
    #         out=out.view(-1)
    #         return out

    # batch_size = 16
    # ## Initializing the VAE and Diffusion block with appropriate hyperparameters
    # VAE = HierarchialVAE(Encoder_Block = Encoder_Block, Decoder_Block = Decoder_Block , latent_dim2 = 5, latent_dim1 = 2, feature_size2 = 36, 
    #                  feature_size1 = 9, hidden_size = 2, pred_length = 5, num_features = 12, seq_length = 12)
    # Diffusion_Process = DiffusionProcess(num_diff_steps = 10, vae = VAE, beta_start = 0.01, beta_end = 0.1, scale = 0.5)

    # ## Initializing the Denoising network with appropriate hyperparameters
    # Denoise_Net = Denoise_net(in_channels = 16,dim = 16, size = 5)

    # ## Defining the MSE Loss and optimizers for parameters of the VAE and denoising net
    # from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
    # criterion=nn.MSELoss()
    # optimizer1=optim.Adam(VAE.parameters(),lr=3e-3)
    # optimizer2=optim.Adam(Denoise_Net.parameters(),lr=3e-3)
    # ## Using Step learning rate scheduler for both the optimizers to ensure stable convergence
    # scheduler1= StepLR(optimizer1, step_size=2, gamma=0.5)
    # scheduler2 = StepLR(optimizer2, step_size=2, gamma=0.5)

    # ## Defining the training loop with number of epochs, VAE and Dnet's and dataloaders as the inputs
    # def train(epochs,train_dataloader,val_dataloader,VAE,dnet,num_diff_steps):
        
    #     ## List for accumulating training and validation losses
    #     train_loss=[]
    #     val_loss=[]
        
    #     ## Iterating over number of epochs
    #     for epoch in range(0,epochs):
            
    #         total_loss=0
    #         ## Setting the both the models into training mode
    #         VAE.train()
    #         dnet.train()
    #         for i,(x,y) in enumerate(train_dataloader):
    #             if(x.size(0)!=16):
    #                 break
    #             ## Initializing the VAE and diffusion outputs
    #             vae_out = torch.zeros((y.size(0), y.size(1),num_diff_steps))
    #             diff_out = torch.zeros((y.size(0), y.size(1),num_diff_steps))
                
    #             ## For number of diffusion timestamps..
    #             for time in range(1,num_diff_steps + 1):
    #                 ## We compute the diffused target as well as target predicted by the VAE
    #                 output, y_noisy = Diffusion_Process.diffuse(x,y,time)
    #                 vae_out[:,:,time-1] = output
    #                 diff_out[:,:,time-1] = y_noisy 
    #             ## To get a approximate distribution of the outputs of the VAE and those produced by the diffusion net by adding noise
    #             ## we use the mean and variances of all outputs of all timestamps (assuming the distribution to be normal)
    #             mean_vae = torch.mean(vae_out, dim = 2)
    #             mean_diff = torch.mean(diff_out, dim = 2)
    #             var_vae = torch.std(vae_out, dim = 2)
    #             var_diff = torch.std(diff_out, dim = 2)
    #             optimizer1.zero_grad()
    #             optimizer2.zero_grad()
    #             ## Computing the MSE loss between mean values of both the outputs
    #             mse_loss = criterion(mean_vae, mean_diff)
    #             ## Computing the KL divergence between both the distributions
    #             ## We used the standard formula for KL divergence between 2 multivariate gaussians
    #             term1 = (mean_vae - mean_diff) / var_diff
    #             term2 = var_vae / var_diff
    #             kl_loss =  0.5 * ((term1 * term1).sum() + (term2 * term2).sum()) - 40 - torch.log(term2).sum()
    #             kl_loss = kl_loss.sum()
                
    #             ran=torch.randint(low=1,high=num_diff_steps + 1,size=(1,))
    #             y_nn=vae_out[:,:,:]
                
    #             E = Denoise_Net(y_nn).sum()
    #             grad_x = torch.autograd.grad(E, y_nn, create_graph=True)[0] 
    #             dsm_loss = torch.mean(torch.sum((y.unsqueeze(2)-y_nn+grad_x*1)**2, [0,1,2])).float()
    #             ## Combining all the 3 losses with appropriate weights which are hyperparameters
    #             loss = 4*mse_loss+0.01*kl_loss+ 0.1*dsm_loss
    #             total_loss+=loss
    #             ## Performing backpropogation and gradient descent
    #             loss.backward()
    #             optimizer1.step()
    #             optimizer2.step()
            
    #         ## Updating the learning rate of both the optimizers according to the specified schedule
    #         scheduler1.step()
    #         scheduler2.step()
            
    #         totalval_loss=0
            
    #         ## Setting the model to evaluation mode
    #         VAE.eval()
    #         dnet.eval()
    #         for i,(x,y) in enumerate(val_dataloader):
    #             if(x.size(0)!=16):
    #                 break
    #             ## Initializing the VAE and diffusion outputs
    #             vae_out = torch.zeros((y.size(0), y.size(1),num_diff_steps))
    #             diff_out = torch.zeros((y.size(0), y.size(1),num_diff_steps))
                    
    #             ## Performing forward pass and computing the losses in a fashion similar to training
    #             for time in range(1,num_diff_steps + 1):
    #                 output, y_noisy = Diffusion_Process.diffuse(x,y,time)
    #                 vae_out[:,:,time-1] = output
    #                 diff_out[:,:,time-1] = y_noisy 
    #             mean_vae = torch.mean(vae_out, dim = 2)
    #             mean_diff = torch.mean(diff_out, dim = 2)
    #             var_vae = torch.std(vae_out, dim = 2)
    #             var_diff = torch.std(diff_out, dim = 2)
    #             mse_loss = criterion(mean_vae, mean_diff)
    #             term1 = (mean_vae - mean_diff) / var_diff
    #             term2 = var_vae / var_diff
    #             kl_loss =  0.5 * ((term1 * term1).sum() + (term2 * term2).sum()) - 40 - torch.log(term2).sum()
    #             kl_loss = kl_loss.sum()
    #             ran=torch.randint(low=1,high=num_diff_steps + 1,size=(1,))
    #             y_nn=vae_out[:,:,:]
    #             E = Denoise_Net(y_nn).sum()
    #             grad_x = torch.autograd.grad(E, y_nn, create_graph=True)[0] 
    #             dsm_loss = torch.mean(torch.sum((y.unsqueeze(2)-y_nn+grad_x*1)**2, [0,1,2])).float()
    #             ## Computing the total validation loss
    #             valloss = 4*mse_loss+0.01*kl_loss+ 0.1*dsm_loss
    #             totalval_loss+=valloss
            
    #         ## Averaging out the training loss over all batches and printing the losses after every epoch
    #         train_loss.append(total_loss/(len(train_dataloader)))
    #         val_loss.append(totalval_loss/(len(val_dataloader)))
    #     return train_loss,val_loss

    # train_loss,val_loss = train(epochs =1 ,train_dataloader = train_dataloader, val_dataloader = val_dataloader, VAE = VAE,dnet = Denoise_Net, num_diff_steps = 10)

    # # As can be seen our model predicts the stock prices for the next 5 days quiet appreciably!
    
    # # Get the list of S&P 500 companies
    # table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    # sp500 = table[0]
    # symbols = sp500['Symbol'].tolist()

    # symbols=load_symbols()
    # Define the date range
    # Calculate the difference between today and the last Monday
    today = datetime.today()
    days_since_monday = today.weekday()  # 0 is Monday, 1 is Tuesday, ..., 6 is Sunday
    days_to_subtract = (days_since_monday + 6) % 7 + 1  # Calculate days to subtract to get to the last Monday
    
    # Calculate the date of the last Monday
    last_monday = today - timedelta(days=days_to_subtract)-timedelta(days=7)
    end_date = last_monday
    start_date = end_date - timedelta(days=69)
    
    # Fetch data for each stock
    # stock_data = {}
    # for symbol in symbols:
    #     stock = yf.download(symbol, start=start_date, end=end_date)
    #     stock_data[symbol] = stock

    # # Defining a Sequencer which converts the data into a format suitable for training
    # class StockDataset2(Dataset):
    #     def __init__(self,data,sequence_length,prediction_length):
    #         ## Initializing the dataframe and the window length i.e. number of previous days which will be used at a time to predict
    #         ## today's closing price
    #         self.data = data
    #         self.sequence_length = sequence_length
    #         self.prediction_length = prediction_length
            
    #     def __len__(self):
    #         ## As it picks indices randomly from [0,len], we keep len =  len(df) - seq_len which denotes the last index which can be
    #         ## used to create a batch as we need seq_len rows ahead of it
    #         return len(self.data) - self.sequence_length
            
    #     def __getitem__(self,index):
    #         ## Slicing the dataframe from input index to input index + seq_len to get the input data 
    #         input_data = self.data[index : index + self.sequence_length]
    #         input_list = input_data.values.tolist()
    #         input = torch.Tensor(input_list)
            
    #         ## Returning the closing prices of next day as the output for each day in the input
    #         ## Converting both the input and output to tensors before returning
    #         output = torch.zeros(self.prediction_length)
    #         output = torch.Tensor(output)
             
    #         return input,output

    # predicted_seq=[]
    # omega=[]
    # downside=[]
    # mean_vae_list=[]
    # std_vae_list=[]
    # ss=[]
    
    # for k in range(len(symbols)):
    #     apple_stock=stock_data[symbols[k]]
    #     apple_stock = apple_stock.groupby('Date').agg({
    #             'Close' : 'last',
    #             'Open' : 'first',
    #             'High' : 'max',
    #             'Low' : 'min',
    #             'Volume' : 'sum'
    #     }).reset_index()
    #     ## Creating features/indicators that are suitable for financial data and aid in stock price predictions as they help in determining the trends
    #     ## Creating a separate column for stochastic oscillator given by (C-L)/(H-L)
    #     apple_stock['Stochastic Oscillator'] = (apple_stock['Close'] - apple_stock['Low'])/(apple_stock['High']-apple_stock['Low'])
    #     ## Defining absolute returns as c_t - c_t-1
    #     apple_stock['Absolute Returns'] = apple_stock['Close'] - apple_stock['Close'].shift(1)
    #     ## Normalizing opening, closing, high and low prices using the previous days' closing price
    #     apple_stock['Closing Price Normalized'] = apple_stock['Close']/apple_stock['Close'].shift(1)
    #     apple_stock['Opening Price Normalized'] = apple_stock['Open']/apple_stock['Close'].shift(1)
    #     apple_stock['High Value Normalized'] = apple_stock['High']/apple_stock['Close'].shift(1)
    #     apple_stock['Low Value Normalized'] = apple_stock['Low']/apple_stock['Close'].shift(1)
    #     ## Normalizing volume of stocks traded by a 5-day rolling mean of the volume
    #     apple_stock['Volume Normalized'] =  apple_stock['Volume']/apple_stock['Volume'].shift(1).rolling(window=5).mean()
    #     ## Defining volatility by the variance of volume of stocks traded over a window of 9 days
    #     apple_stock['Volatility'] = apple_stock['Volume'].rolling(window=9).var()/1e16
    #     ## Removing unnecessary rows and resetting the index
    #     apple_stock.dropna(inplace = True)
    #     apple_stock.reset_index(drop = True, inplace = True)
    #     ## Defining MACD () as  5 day EMA - 9 day EMA
    #     ## Defining the periods over which EMA is to be calculated
    #     period1 = 9
    #     period2 = 5
    #     # Calculate the smoothing factor (alpha)
    #     alpha1 = 2 / (period1 + 1)
    #     alpha2 = 2/(period2 +1 )
    #     # Calculate 9-day EMA and 5-day EMA using the pandas `ewm` method
    #     EMA9day = apple_stock['Close'].ewm(span=period1, adjust=False).mean()
    #     EMA5day = apple_stock['Close'].ewm(span=period2, adjust=False).mean()
    #     apple_stock['MACD'] = EMA5day - EMA9day
    #     ## Defining Gains/Losses for a particular day depending on whether O>C or C>O
    #     apple_stock['Gains'] = (apple_stock['Close'] - apple_stock['Open'])*100/apple_stock['Open']
    #     apple_stock['Gains'] = apple_stock['Gains'].apply(lambda x: max(0, x))
    #     apple_stock['Losses'] = (apple_stock['Open'] - apple_stock['Close'])*100/apple_stock['Open']
    #     apple_stock['Losses'] = apple_stock['Losses'].apply(lambda x: max(0, x))
    #     ## Computing the Relative Strength Index (RSI) using the gains/losses which is dependent on average gains and average losses
    #     rsi_data = []
    #     ## Defining epsilon to avoid nan related issues (small value)
    #     eps = 1e-8
    #     for i in range(9,len(apple_stock)):
    #         ## Calculating the RSI gains for the last 9 days
    #         rsi_gains = np.array(apple_stock.iloc[i-9:i]['Gains'])
    #         ## Calculating average positive gains over the last 9 days
    #         average_gains = np.mean(rsi_gains[rsi_gains > 0])
    #         average_gains = 0 if np.isnan(average_gains) else average_gains
    #         ## Calculating RSI losses for the last 9 days
    #         rsi_losses = np.array(apple_stock.iloc[i-9:i]['Losses'])
    #         average_losses = np.mean(rsi_losses[rsi_losses > 0])
    #         average_losses = 0 if np.isnan(average_losses) else average_losses
    #         ## Computing the RSI index
    #         den = 1+ average_gains/(average_losses + eps)
    #         rsi_data.append(100-(100/den))
        
    #     #Removing the first 9 rows, due to rolling mean
    #     apple_stock = apple_stock.iloc[9:]
    #     apple_stock['RSI'] = rsi_data
    #     # Using periodic features that help to discover repetitive patterns or cycles within the data
    #     # These cycles aid in predicting future price movements
    #     apple_stock.reset_index(inplace = True, drop = True)
    #     apple_stock['Sine'] = np.sin(2*np.pi/20*pd.DatetimeIndex(data = apple_stock['Date'], yearfirst = True).day)
    #     apple_stock['Cosine'] = np.cos(2*np.pi/20*pd.DatetimeIndex(data = apple_stock['Date'], yearfirst = True).day)
    #     apple_stock.drop(['Date'],axis=1,inplace=True)
        
        
        
        
    #     input_features = ['Closing Price Normalized', 'Opening Price Normalized','High Value Normalized','Low Value Normalized','Volume Normalized',
    #                      'Stochastic Oscillator', 'Absolute Returns','Volatility','MACD','RSI','Sine','Cosine']
    #     df_app = apple_stock[input_features].copy()
    #     print(len(df_app))
    #     if(len(df_app)!=28):
    #         continue
    #     else:
    #         ss.append(symbols[k])
    #     sequence_length = 12
    #     prediction_length = 5
    #     sequenced_data = StockDataset2(df_app,sequence_length,prediction_length)
        
    #     #Splitting the indices of the sequences, so as to maintain order of time series
    #     indices = list(range(len(sequenced_data)))
    #     test_dataset = torch.utils.data.Subset(sequenced_data, indices)
    #     test_dataloader=DataLoader(test_dataset,batch_size=16,shuffle=False)
    #     print(len(sequenced_data))
        
    
    #     num_diff_steps=10
        
    #     for j,(x,y) in enumerate(test_dataloader):
    #         if(x.size(0)!=16):
    #             break
    #         vae_out = torch.zeros((y.size(0), y.size(1),num_diff_steps))
    #         diff_out = torch.zeros((y.size(0), y.size(1),num_diff_steps))
        
    #         #Similar to the training loop
    #         for time in range(1,num_diff_steps + 1):
    #             output, y_noisy = Diffusion_Process.diffuse(x,y,time)
    #             vae_out[:,:,time-1] = output
    #             diff_out[:,:,time-1] = y_noisy 
    #         mean_vae = torch.mean(vae_out, dim = 2)
    #         mean_diff = torch.mean(diff_out, dim = 2)
    #         var_vae = torch.std(vae_out, dim = 2)
    #         var_diff = torch.std(diff_out, dim = 2)
    #         #mse_loss = criterion(mean_vae, mean_diff)
    #         term1 = (mean_vae - mean_diff) / var_diff
    #         term2 = var_vae / var_diff
    #         y_nn=vae_out[:,:,:]
    #         E = Denoise_Net(y_nn).sum()
    #         grad_x = torch.autograd.grad(E, y_nn, create_graph=True)[0] 
    #         p=var_vae[-1]
    #         q=(mean_vae - torch.mean(grad_x,dim=2))[-1]
    #         predicted_seq.append((mean_vae - torch.mean(grad_x,dim=2))[-1])
    #         # omega.append(min((torch.sum((mean_vae - torch.mean(grad_x,dim=2))[mean_vae - torch.mean(grad_x,dim=2)>0].reshape(mean_vae.size(0),-1),dim=1)[-1])/
    #         #                  torch.sum(torch.abs((mean_vae - torch.mean(grad_x,dim=2))[mean_vae - torch.mean(grad_x,dim=2)<0]).reshape(mean_vae.size(0),-1),dim=1)[-1],7))
    #         omega.append(0)
    #         downside.append(torch.sum(p[q<1]))
    #         mean_vae_list.append(torch.prod(mean_vae - torch.mean(grad_x,dim=2),dim=1)[-1])
    #         std_vae_list.append(torch.sum(var_vae,dim=1)[-1])

    # print(len(mean_vae_list),len(std_vae_list),len(downside))
    # def rank_assets(symbols, predicted_seq, mean_vae, var_vae,omega, downside, weights):
    #     sharpe_ratios=np.array(omega)
    #     sortino_ratios=np.array(omega)
    #     omega_ratios=np.array(omega)
        
    #     for i in range(len(mean_vae)):
    #         # Calculate Sharpe ratios
    #         sharpe_ratios[i] = min(abs(mean_vae[i]) / var_vae[i],10)
            
    #         # Calculate Sortino ratios
    #         sortino_ratios[i] = min(abs(mean_vae[i]) / downside[i],10)
    #         #sortino_ratios[i]=0
    #         # Calculate Omega Ratios
    #         omega_ratios[i] = 0
            
    #     # Weighted Combined scores for overall ranking
    #     combined_scores = (
    #         weights[0] * sharpe_ratios +
    #         weights[1] * sortino_ratios +
    #         weights[2] * omega_ratios
    #     )    
    #     # Rank assets based on weighted combined scores
    #     ranked_assets = symbols[np.argsort(combined_scores)[::-1]]  # Reverse the sorting to get highest first
    #     return ranked_assets,combined_scores,sharpe_ratios,sortino_ratios
    
    # # Example weights (adjust based on preference or analysis)
    # weights = [0.4, 0.35, 0.25]  # Sharpe, Sortino, Omega, Momentum
    
    # # Example usage
    # ranked_assets,score,sharpe,sortino = rank_assets(np.array(ss), predicted_seq, mean_vae_list, std_vae_list,omega, downside, weights)

    # ordered=np.argsort(score)[::-1]
    # sym=[]
    # returns=[]
    # lli=[]
    # k=0
    # i=0
    # while k<5:
    #     if(abs(mean_vae_list[ordered[i]]-1)>0.4):
    #         i=i+1
    #         continue
    #     else:
    #         i=i+1
    #         k=k+1
    #     sym.append(ss[ordered[i-1]])
    #     returns.append((100/np.sqrt(25))*(mean_vae_list[ordered[i-1]].detach().numpy()-1))
    #     if mean_vae_list[ordered[i-1]]>1:
    #         lli.append("Buy")
    #     elif mean_vae_list[ordered[i-1]]<1:
    #         lli.append("Sell")

    sym=[1]
    returns=[1]
    lli=[1]
    li={'Symbols':sym,'Returns':returns,'Signal':lli}
    ranks=pd.DataFrame(li)
    ranks.to_excel('out/ranking.xlsx',index=False)
    return ranks,last_monday




# Get information for dashboard
def get_dashboard_info():
    #cols = ['Symbol', 'State', 'Current Price', 'New Highest']
    cols = ['Symbol', 'Current Price', 'New Highest']
    res = pd.DataFrame(columns = cols)

    for symbol in tqdm(load_stock_symbols(), desc = 'loading', colour = 'green'):
        df = load_yf(symbol, '1800-01-01', '2100-01-01', INTERVAL_DAILY, for_backup = True)

        highs = df['High'].to_numpy()		
        last_date = df.index[-1]

        is_new_highest = (highs.argmax() == len(highs) - 1)
        is_bullish = df.loc[last_date]['Close'] >= df.loc[last_date]['Open']

        record = [
            symbol,
            #'↑ Bullish' if is_bullish else '↓ Bearish',
            '${:.4f}'.format(df.loc[last_date]['Close']),
            '√ ${:.4f}'.format(highs[-1]) if is_new_highest else '--------'
        ]
        res = pd.concat([res, pd.Series(dict(zip(cols, record))).to_frame().T], ignore_index = True)

    res = pd.concat([res, pd.Series({}).to_frame().T], ignore_index = True)
    return res, last_date

def is_pivot(candle, window, df):
    if candle - window < 0 or candle + window >= len(df): return 0
    
    pivot_high = 1
    pivot_low = 2
    
    for i in range(candle - window, candle + window + 1):
        if df.iloc[candle].Low > df.iloc[i].Low: pivot_low = 0
        if df.iloc[candle].High < df.iloc[i].High: pivot_high = 0
    
    if pivot_high and pivot_low:
        return 3
    elif pivot_high:
        return pivot_high
    elif pivot_low:
        return pivot_low
    else:
        return 0

def calculate_point_pos(row):
    if row['isPivot'] == 2:
        return row['Low'] - 1e-3
    elif row['isPivot'] == 1:
        return row['High'] + 1e-3
    else:
        return np.nan

def alert_trendline(df, symbol, cur_date, interval, level, width):
    # df = copy.deepcopy(df)[:cur_date]
    df = copy.deepcopy(df)

    df['ID'] = range(len(df))
    df['Date'] = list(df.index)
    df.set_index('ID', inplace = True)

    atr = ta.atr(high = df['High'], low = df['Low'], close = df['Close'], length = 14)
    atr_multiplier = 2 
    try:
        atr = ta.atr(high = df['High'], low = df['Low'], close = df['Close'], length = 14);
        stop_percentage = 2 * atr.iloc[-1] / df['Close'].iloc[-1]
    except:
        atr = ta.atr(high = df['High'], low = df['Low'], close = df['Close'], length = len(df))
        stop_percentage = 2 * atr.iloc[-1] / df['Close'].iloc[-1]
    profit_percentage = (1 + 3 / 4) * stop_percentage
    
    output = 'No signal today for {} on {} interval.'.format(symbol, interval)

    #for level in range(2, 11, 2):
    # for level in range(2, 3, 2):

    # window = width * level
    # backcandles = 10 * window
    # df['isPivot'] = df.apply(lambda row: is_pivot(row.name, window, df), axis = 1)
    
    # signal = is_breakout(len(df) - 1, backcandles, window, df, stop_percentage)
    signal = df.loc[len(df)-1, 'isBreakOut']

    if signal == 1:
        output = 'Short signal for today for {} on {} interval with Take_Profit={:.1f}% and Stop_Loss={:.1f}%.'.format(
            symbol, interval, profit_percentage, stop_percentage
        )
    elif signal == 2:
        output = 'Long signal for today for {} on {} interval with Take_Profit={:.1f}% and Stop_Loss={:.1f}%.'.format(
                symbol, interval, profit_percentage, stop_percentage
            )
    return output

# Backtest Trendline Program
def backtest_trendline(df, symbol, from_date, to_date, interval):
    combined_trades = pd.DataFrame()
    
    df['ID'] = range(len(df))
    df['Date'] = list(df.index)
    df.set_index('ID', inplace = True)

    try:
        atr = ta.atr(high = df['High'], low = df['Low'], close = df['Close'], length = 14);
        stop_percentage = 2 * atr.iloc[-1] / df['Close'].iloc[-1]
    except:
        atr = ta.atr(high = df['High'], low = df['Low'], close = df['Close'], length = len(df))
        stop_percentage = 2 * atr.iloc[-1] / df['Close'].iloc[-1]
        
    # For a certain date point, we can only have one decision
    # So the loop of levels and arrange the results is nonsense
    # Nikola fixed it so to calculate only once
    for level in tqdm(list(range(2, 11, 2))):
    #for level in range(4, 5, 2):
        window = 3 * level
        backcandles = 10 * window
        
        df['isPivot'] = df.apply(lambda row: is_pivot(row.name, window, df), axis = 1)
        df['isBreakOut'] = 0
        df['BreakoutInfoSlope'] = 0
        df['BreakoutInfoIntercept'] = 0
  
        for i in range(backcandles + window, len(df)):
            df.loc[i, 'isBreakOut'] = is_breakout(i, backcandles, window, df, stop_percentage)
            df.loc[i, 'BreakoutInfoSlope'] = breakout_line_data(i, backcandles, window, df, stop_percentage)[0]
            df.loc[i, 'BreakoutInfoIntercept'] = breakout_line_data(i, backcandles, window, df, stop_percentage)[1]

        trades_data = unit_trendline_backtest(df, level)
        combined_trades = pd.concat([combined_trades, trades_data])

    # Merge the transactions with the same entry dates
    combined_trades = combined_trades.sort_values(by = 'Enter Date')
    combined_trades = combined_trades.drop_duplicates(subset = ['Enter Date', 'Exit Date'], keep = 'first')

    total_trades = len(combined_trades)
    profitable_trades = len(combined_trades[combined_trades['Profit/Loss'] > 0])
    success_rate = profitable_trades / total_trades if total_trades != 0 else 0

    valid_trades = combined_trades.dropna(subset = ['Return']).copy()
    valid_trades['Cumulative Return'] = (1 + valid_trades['Return']/100).cumprod()

    if len(valid_trades) > 0:
        overall_return = valid_trades['Cumulative Return'].iloc[-1] - 1
    else:
        overall_return = 0
    
    combined_trades = combined_trades.drop('Profit/Loss', axis = 1)
    combined_trades = combined_trades.drop('Level', axis = 1)
    combined_trades = combined_trades.round(2)
    for i in range(len(combined_trades)):
        combined_trades['Return'].iloc[i] = str(combined_trades['Return'].iloc[i]) + "%"

    # Records for visualization dataframe
    last_records = [
        {},
        {
               'Enter Price': f"Ticker: {symbol}",
            'Exit Date': f"From: {change_date_format(from_date, YMD_FORMAT, '%Y-%m-%d')}",
            'Exit Price': f"To: {change_date_format(to_date, YMD_FORMAT, '%Y-%m-%d')}",
            'Signal': f"By: {interval}"
        },
        {
            'Enter Price': 'Success Rate:',
            'Exit Date': '{:.1f}%'.format(success_rate * 100),
            'Exit Price': 'Cumulative Profit:',
               'Signal': '{:.1f}%'.format(overall_return * 100)
        },
        {}
    ]
    for r in last_records:
        combined_trades = pd.concat([combined_trades, pd.Series(r).to_frame().T], ignore_index = True)

    return combined_trades, success_rate, overall_return

def collect_channel(candle, backcandles, window, df):
    best_r_squared_low = 0
    best_r_squared_high = 0
    best_slope_low = 0
    best_intercept_low = 0
    best_slope_high = 0
    best_intercept_high = 0
    best_backcandles_low = 0
    best_backcandles_high = 0
    
    for i in range(backcandles - backcandles // 2, backcandles + backcandles // 2, window):
        local_df = df.iloc[candle - i - window: candle - window]
        
        lows = local_df[local_df['isPivot'] == 2].Low.values[-4:]
        idx_lows = local_df[local_df['isPivot'] == 2].Low.index[-4:]
        highs = local_df[local_df['isPivot'] == 1].High.values[-4:]
        idx_highs = local_df[local_df['isPivot'] == 1].High.index[-4:]

        if len(lows) >= 2:
            slope_low, intercept_low, r_value_l, _, _ = stats.linregress(idx_lows, lows)
            
            if (r_value_l ** 2) * len(lows) > best_r_squared_low and (r_value_l ** 2) > 0.85:
                best_r_squared_low = (r_value_l ** 2) * len(lows)
                best_slope_low = slope_low
                best_intercept_low = intercept_low
                best_backcandles_low = i
        
        if len(highs) >= 2:
            slope_high, intercept_high, r_value_h, _, _ = stats.linregress(idx_highs, highs)
            
            if (r_value_h ** 2) * len(highs) > best_r_squared_high and (r_value_h ** 2)> 0.85:
                best_r_squared_high = (r_value_h ** 2) * len(highs)
                best_slope_high = slope_high
                best_intercept_high = intercept_high
                best_backcandles_high = i
    
    return best_backcandles_low, best_slope_low, best_intercept_low, best_r_squared_low, best_backcandles_high, best_slope_high, best_intercept_high, best_r_squared_high

def is_breakout(candle, backcandles, window, df, stop_percentage):
    if 'isBreakOut' not in df.columns: return 0

    for i in range(1, 2):
        if df['isBreakOut'].iloc[candle - i] != 0: return 0
  
    if candle - backcandles - window < 0: return 0
    best_back_l, sl_lows, interc_lows, r_sq_l, best_back_h, sl_highs, interc_highs, r_sq_h = collect_channel(candle, backcandles, window, df)
    
    thirdback = candle - 2
    thirdback_low = df.iloc[thirdback].Low
    thirdback_high = df.iloc[thirdback].High
    thirdback_volume = df.iloc[thirdback].Volume

    prev_idx = candle - 1
    prev_close = df.iloc[prev_idx].Close
    prev_open = df.iloc[prev_idx].Open
    
    curr_idx = candle
    curr_close = df.iloc[curr_idx].Close
    curr_open = df.iloc[curr_idx].Open
    curr_volume= max(df.iloc[candle].Volume, df.iloc[candle-1].Volume)


    if ( 
        thirdback_high > sl_lows * thirdback + interc_lows and
        curr_volume > thirdback_volume and
        prev_close < prev_open and
        curr_close < curr_open and
        sl_lows > 0 and
        prev_close < sl_lows * prev_idx + interc_lows and
        curr_close < sl_lows * prev_idx + interc_lows):
        return 1
    elif (
        thirdback_low < sl_highs * thirdback + interc_highs and
        curr_volume > thirdback_volume and
        prev_close > prev_open and 
        curr_close > curr_open and
        sl_highs < 0 and
        prev_close > sl_highs * prev_idx + interc_highs and
        curr_close > sl_highs * prev_idx + interc_highs):
        return 2
    else:
        return 0

def breakout_line_data(candle, backcandles, window, df, stop_percentage):
    if 'isBreakOut' not in df.columns: return [0, 0]

    for i in range(1, 2):
        if df['isBreakOut'].iloc[candle - i] != 0: return [0, 0]
  
    if candle - backcandles - window < 0: return 0
    best_back_l, sl_lows, interc_lows, r_sq_l, best_back_h, sl_highs, interc_highs, r_sq_h = collect_channel(candle, backcandles, window, df)
    
    thirdback = candle - 2
    thirdback_low = df.iloc[thirdback].Low
    thirdback_high = df.iloc[thirdback].High
    thirdback_volume = df.iloc[thirdback].Volume

    prev_idx = candle - 1
    prev_close = df.iloc[prev_idx].Close
    prev_open = df.iloc[prev_idx].Open
    
    curr_idx = candle
    curr_close = df.iloc[curr_idx].Close
    curr_open = df.iloc[curr_idx].Open
    curr_volume= max(df.iloc[candle].Volume, df.iloc[candle-1].Volume)


    if ( 
        thirdback_high > sl_lows * thirdback + interc_lows and
        curr_volume > thirdback_volume and
        prev_close < prev_open and
        curr_close < curr_open and
        sl_lows > 0 and
        prev_close < sl_lows * prev_idx + interc_lows and
        curr_close < sl_lows * prev_idx + interc_lows):
        return [sl_lows, interc_lows]
    elif (
        thirdback_low < sl_highs * thirdback + interc_highs and
        curr_volume > thirdback_volume and
        prev_close > prev_open and 
        curr_close > curr_open and
        sl_highs < 0 and
        prev_close > sl_highs * prev_idx + interc_highs and
        curr_close > sl_highs * prev_idx + interc_highs):
        return [sl_highs, interc_highs]
    else:
        return [0, 0]

# Calculate overlaps of two regions
def getOverlap(r1, r2, percent = 0.3):
    s1 = r1.start
    s2 = r2.start
    e1 = r1.end
    e2 = r2.end

    if s2 <= s1 and e2 <= s1:
        return False
    elif s2 >= e1 and e2 >= e1:
        return False
    elif s2 < s1 and e2 > e1:
        return True
    elif s2 > s1 and e2 < e1:
        return True
    elif s1 < s2 and e2 > s1:
        p = (e2 - s1) / (e1 - s1)
        return p > percent
    elif s2 < e1 and e2 > e1:
        p = (e1 - s2) / (e1 - s1)
        return p > percent


def getDivergance_LL_HL(r, rS):
    divs = []
    
    for rr in r:
        for rrs in rS:
            if getOverlap(rr, rrs):
                sc = rr.class_
                dc = rrs.class_

                if sc == -1 or sc == 0:
                    if dc == 1:
                        if not rr.start == rr.end and not rrs.start == rrs.end: divs.append(( (rrs.start, rr.start), (rrs.end, rr.end)))
    return divs

def getDivergance_HH_LH(r, rS):
    divs = []
    
    for rr in r:
        for rrs in rS:
            if getOverlap(rr, rrs):
                sc = rr.class_
                dc = rrs.class_

                if sc == 1 or sc == 0:
                    if dc == -1:
                        if not rr.start == rr.end and not rrs.start == rrs.end: divs.append(( (rrs.start, rr.start), (rrs.end, rr.end)))
    return divs


def calculate_breakpoint_pos(row):
    if row['isBreakOut'] == 2:
        return row['Low'] - 3e-3
    elif row['isBreakOut'] == 1:
        return row['High'] + 3e-3
    else:
        return np.nan

def unit_trendline_backtest(df, level):
    trades = []

    for i in range(1, len(df)):
        signal_type = df['isBreakOut'].iloc[i]
        signal = ""
        
        slope = df['BreakoutInfoSlope'].iloc[i]
        intercept = df['BreakoutInfoIntercept'].iloc[i]
        count = 0
        if signal_type == 2:
            signal = "Long"
            entry_date = df['Date'].iloc[i].strftime(YMD_FORMAT)
            entry_price = df['Close'].iloc[i]
            exit_price = None

            for j in range(i + 1, len(df)):
                if (slope*j+intercept >= df['Close'].iloc[j]):
                    count += 1;
                if count >= 2:
                    exit_date = df['Date'].iloc[j].strftime(YMD_FORMAT)
                    exit_price = df['Close'].iloc[j]
                    break
                if df['isPivot'].iloc[j] != 0:
                    exit_date = df['Date'].iloc[j].strftime(YMD_FORMAT)
                    exit_price = df['Close'].iloc[j]
                    break

            if exit_price is None:
                exit_date = df['Date'].iloc[-1].strftime(YMD_FORMAT)
                exit_price = df['Close'].iloc[-1]

            profit_or_stopped = calculate_profit_or_stopped(entry_price, exit_price, signal_type)
            trades.append((entry_date, entry_price, exit_date, exit_price, profit_or_stopped,signal,level))
        elif signal_type == 1:
              signal = "Short"
              entry_date = df['Date'].iloc[i].strftime(YMD_FORMAT)
              entry_price = df['Close'].iloc[i]
              exit_price = None
              
              for j in range(i + 1, len(df)):
                if (slope*j+intercept <= df['Close'].iloc[j]):
                    count += 1;
                if count >= 2:
                    exit_date = df['Date'].iloc[j].strftime(YMD_FORMAT)
                    exit_price = df['Close'].iloc[j]
                    break
                if df['isPivot'].iloc[j] != 0:
                    exit_date = df['Date'].iloc[j].strftime(YMD_FORMAT)
                    exit_price = df['Close'].iloc[j]
                    break

              if exit_price is None:
                    exit_date = df['Date'].iloc[-1].strftime(YMD_FORMAT)
                    exit_price = df['Close'].iloc[-1]

              profit_or_stopped = calculate_profit_or_stopped(entry_price, exit_price, signal_type)
              trades.append((entry_date, entry_price, exit_date, exit_price, profit_or_stopped, signal, level))

    trade_data = pd.DataFrame(trades, columns = ['Enter Date', 'Enter Price', 'Exit Date', 'Exit Price', 'Profit/Loss', 'Signal', 'Level'])
    trade_data['Return'] = 100 * trade_data['Profit/Loss'] * abs(trade_data['Enter Price'] - trade_data['Exit Price'] ) / trade_data['Enter Price']

    return trade_data

def calculate_profit_or_stopped(entry_price, exit_price, long_or_short):
  if long_or_short == 2:
    if exit_price >= entry_price :
        return 1
    else:
        return -1
  elif long_or_short == 1:
    if exit_price <= entry_price :
        return 1
    else:
        return -1

def get_inter_divergence_lows_general(df1, df2):
    last_low_1 = df1.iloc[-1].Low
    last_low_2 = df2.iloc[-1].Low
    counter_backward = 0
    #print(df1)
    i = len(df1) - 2
    enddd = len(df1) - 1
    starts, ends = -1, -1
    #i = 1
    while i >= 1:
        # datee = df1.iloc[i].name
        datee = i
        counter_backward = counter_backward + 1
        if counter_backward > 8:
            starts = -1
            break
        #print("checking for ",df1.iloc[i].name)
        dir1 = np.sign(last_low_1 - df1.iloc[i].Low)
        dir2 = np.sign(last_low_2 - df2.iloc[i].Low)
        
        if dir1 != dir2:
            if starts == -1: starts = len(df1) - i - 1
            ends = starts
        if starts != -1: break
        i -= 1
    if starts != -1:
        starts = len(df1) - 1 -  starts
    ends = starts
    return starts,datee,datee

def get_inter_divergence_lows(df1, df2):
    last_low_1 = df1.iloc[-1].Low
    last_low_2 = df2.iloc[-1].Low

    i = len(df1) - 2
    starts, ends = -1, -1
    
    while i >= 0:
        dir1 = np.sign(last_low_1 - df1.iloc[i].Low)
        dir2 = np.sign(last_low_2 - df2.iloc[i].Low)
        
        if dir1 != dir2:
            if starts != -1: starts = len(df1) - i - 1
            ends = len(df1) - i - 1

        if starts != -1: break
        i -= 1

    return starts, ends


def get_inter_divergence_highs(df1, df2):
    last_high_1 = df1.iloc[-1].High
    last_high_2 = df2.iloc[-1].High
    counter_backward = 0
        
    i = len(df1) - 2
    starts, ends = -1, -1

    while i >= 0:
        counter_backward = counter_backward + 1
        if counter_backward > 8:
            starts = -1
            break
        dir1 = np.sign(last_high_1 - df1.iloc[i].High)
        dir2 = np.sign(last_high_2 - df2.iloc[i].High)
        if dir1 != dir2:
            if starts == -1: starts = i
            ends = i
        if starts != -1: break
        i -= 1

    return starts, ends

def getPointsBest(STOCK, min_ = 0.23, max_ = 0.8, getR = False, startDate = '2000-01-01', endDate = '2121-01-01',
        increment = 0.005, limit = 100, interval = INTERVAL_DAILY, returnData = 'close'):
    
    data = load_yf(STOCK, startDate, endDate, interval)
    data = data.dropna()
    data = data.rename(columns = {"Open": "open", "High": "high", "Low": "low", "Volume": "volume", "Close": "close"})
    
    date_format = "%Y-%m-%d"
    
    end_ = datetime.strptime(endDate, date_format)
    today = datetime.today()

    d = today - end_
    
    if returnData == 'close':
        Sdata = getScaledY(data["close"])
    elif returnData == 'lows':
        Sdata = getScaledY(data["low"])
    elif returnData == 'highs':
        Sdata = getScaledY(data["high"])
    else:
        print("Wrong data for argument returnData")
        return None
    
    R = 1.1
    satisfied = False
    c = 0
    
    while not satisfied and c < limit and R > 1:
        if returnData == 'close':
            highs, lows = getPointsforArray(data["close"], R)
        elif returnData == 'lows':
            highs, lows = getPointsforArray(data["low"], R)
        else:
            highs, lows = getPointsforArray(data["high"], R)
        
        if not len(highs) <2 and not len(lows) < 2:
            linears = getLinears(Sdata, sorted(highs+lows))
            MSE = getMSE(Sdata, linears)
            c += 1
            
            if min_ < MSE and MSE < max_:
                satisfied = True
            elif MSE > min_:
                R -= increment
            else:
                R += increment                
        else:
            R -= increment
    if R > 1:
        if returnData == 'close':
            h, l = getPointsforArray(data["close"], R)
        elif returnData == 'lows':
            h, l = getPointsforArray(data["low"], R)
        else:
            h, l = getPointsforArray(data["close"], R)
    else:
        if returnData == 'close':
            h, l = getPointsforArray(data["close"], 1.001)
        elif returnData == 'lows':
            h, l = getPointsforArray(data["low"], 1.001)
        else:
            h, l = getPointsforArray(data["close"], 1.001)

    if getR:
        return data, h, l, R
    else:
        return data, h, l

def getScaledY(data):
    return (np.asarray(data) - min(data)) / (max(data) - min(data))

def getScaledX(x, data):
    return np.asarray(x) / len(data)

def getUnscaledX(x, data):
    p =  x * len(data)
    return int(p)

def getLinears(data, Tps):
    linears = []
    i = 0
    
    while i + 1 < len(Tps):
        l = Linear(getScaledX(Tps[i], data), data[Tps[i]], getScaledX(Tps[i + 1], data), data[Tps[i + 1]])
        linears.append(l)
        i += 1
    return linears

def getLinearForX(lins, x):
    for l in lins:
        if l.isInRange(x): return l

    return None

def getMSE(data, lins):
    i = lins[0].x1
    E = 0
    
    while i < lins[-1].x2:
        l = getLinearForX(lins, i)
        p = data[getUnscaledX(i, data)]
        pHat = l.getY(i)
        E += abs((p - pHat)) * 1 / len(data)
        i += 1 / len(data)

    return E * 10

def getPointsforArray(series, R = 1.1):
    highs, lows = getTurningPoints(series, R, combined = False)
    return highs, lows

def getTurningPoints(closeSmall, R, combined = True):    
    markers_on = []
    highs = []
    lows = []
    
    i, markers_on = findFirst(closeSmall, len(closeSmall), R, markers_on)
    
    if i < len(closeSmall) and closeSmall[i] > closeSmall[0]:
        i, highs = finMax(i, closeSmall, len(closeSmall) - 1, R, highs)
    while i < len(closeSmall) - 1 and not math.isnan(closeSmall[i]):
        i, lows = finMin(i, closeSmall, len(closeSmall) - 1, R, lows)
        i, highs = finMax(i, closeSmall, len(closeSmall) - 1, R, highs)
    
    if combined:
        return highs + lows
    else:
        return highs, lows

def findFirst(a, n, R, markers_on):
    iMin = 1
    iMax = 1
    i = 2
    while i<n and a[i] / a[iMin]< R and a[iMax]/a[i]< R:
        if a[i] < a[iMin]:
            iMin = i
        if a[i] > a[iMax]:
            iMax = i
        i += 1
    if iMin < iMax:
        markers_on.append(iMin)
    else:
        markers_on.append(iMax)
    return i, markers_on

def finMin(i, a, n, R, markers_on):
    iMin = i
    
    while i < n and a[iMin]!=0 and a[i] / a[iMin] < R:
        if a[i] < a[iMin]: iMin = i
        i += 1
        
    if i < n or a[iMin] < a[i]:
        markers_on.append(iMin)
        
    return i, markers_on

def finMax(i, a, n, R, markers_on):
    iMax = i
    
    while i < n and a[i]!=0 and a[iMax] / a[i] < R:
        if a[i] > a[iMax]: iMax = i
        i += 1
        
    if i < n or a[iMax] > a[i]:
        markers_on.append(iMax)
        
    return i, markers_on

def getPointsGivenR(STOCK, R, startDate = '2000-01-01', endDate = '2121-01-01', interval = INTERVAL_DAILY, type_ = None, oldData = None):
    if oldData is None:
        data = load_yf(STOCK, startDate, endDate, interval)
        data = data.dropna()
        data = data.rename(columns = {"Open": "open", "High": "high", "Low": "low", "Volume": "volume", "Close": "close"})
    else:
        data = oldData

    date_format = "%Y-%m-%d"
    end_ = datetime.strptime(endDate, date_format)
    today = datetime.today()

    d = today - end_

    if type_ is None:
        highs, lows = getPointsforArray(data["close"], R)
        return data, highs, lows
    elif type_== 'lows':
        _, lows = getPointsforArray(data["low"], R)
        return data, lows
    elif type_== 'highs':
        highs, _ = getPointsforArray(data["high"], R)
        return data, highs
    else:
        return None, None

def runStochDivergance(symbol, from_date = '2000-01-01', to_date = '2022-08-07', return_csv = False, cur_date = None, old_data = None):
    R = 1.02
    data, _, _ = getPointsGivenR(symbol, R, startDate = from_date, endDate = to_date, oldData = old_data)
    _, lows = getPointsGivenR(symbol, R, startDate = from_date, endDate = to_date, type_='lows', oldData = data)
    _, highs = getPointsGivenR(symbol, R, startDate = from_date, endDate = to_date, type_='highs', oldData = data)

    lows = np.asarray(lows)
    lows -= 15
    lows = lows[lows >= 0]
    lows = lows.tolist()

    highs = np.asarray(highs)
    highs -= 15
    highs = highs[highs >= 0]
    highs = highs.tolist()

    #K = TA.STOCH(data, 14) # Why not use?
    D = TA.STOCHD(data)

    data = data[15:]
    D = D[15:]
    x = D.to_numpy()

    highsStoch, lowsStoch = getPointsforArray(x, 1.05)
    highsStoch.append(len(D) - 1)

    rr = getReigons(lows, data['low'])
    fr = getFinalReigons(rr)
    rr1 = getReigons(highs, data['high'])
    fr1 = getFinalReigons(rr1)
    rrS1 = getReigons(highsStoch, D)
    frS1 = getFinalReigons(rrS1)
    rrS1 = getReigons(lowsStoch, D)
    frS2 = getFinalReigons(rrS1)

    type1 = getDivergance_LL_HL(fr, frS2)
    type2 = getDivergance_HH_LH(fr1, frS1)

    df = data

    if not return_csv:
        fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True, vertical_spacing = 0.01, subplot_titles = ('Stock prices', 'Stochastic Indicator'), row_width = [0.29,0.7])
        fig.update_yaxes(type = 'log', row = 1, col = 1)
        fig.add_trace(go.Candlestick(x = df.index, open = df['open'], high = df['high'], low = df['low'], close = df['close'], showlegend = False), row = 1, col = 1)
        fig.update_layout(xaxis_rangeslider_visible = False, yaxis_tickformat = '0')
        
        fig.add_trace(go.Scatter(x = D.index, y = D, showlegend = False), row = 2, col = 1)
        fig.add_trace(go.Scatter(x = df.index, y = df['close'].rolling(10).mean(), name = 'MA-10W'))
        fig.add_trace(go.Scatter(x = df.index, y = df['close'].rolling(40).mean(), name = 'MA-40W'))

    lines_to_draw, typeONEs = [], []

    for t in type1:
        sS, eS = t[0][0], t[1][0]
        sD, eD = t[0][1], t[1][1]
        stockS = data.iloc[t[0][0]].high
        stockE = data.iloc[t[1][0]].high

        if not eS == sS and not sD == eD:
            StockM = (stockE - stockS) / (eS - sS)
            Dm = (eD - sS) / (eD - sD)

            if StockM > 0.2 and Dm > 0.2:
                pass
            elif StockM < -0.2 and Dm < -0.2:
                pass
            else:
                start = max(t[0][1], t[0][0])
                ending = min(t[1])
                stockStart = start
                stockEnd = ending

                dStart = start
                dEnd = ending
                
                if True:#data.iloc[stockStart].low > data.iloc[stockEnd].low: # Nikola Added
                    a1 = dict(
                        x0 = data.iloc[dStart].name,
                        y0 = D.iloc[dStart],
                        x1 = data.iloc[dEnd].name,
                        y1 = D.iloc[dEnd],
                        type = 'line',
                        xref = 'x2',
                        yref = 'y2',
                        line_width = 4,
                        line_color = 'blue'
                    )
                    b1 = dict(
                        x0 = data.iloc[stockStart].name,
                        y0 = data.iloc[stockStart].low,
                        x1 = data.iloc[stockEnd].name,
                        y1 = data.iloc[stockEnd].low,
                        type = 'line',
                        xref = 'x',
                        yref = 'y',
                        line_width = 4,
                        line_color = 'blue'
                    )
                    typeONEs.append((a1, b1))
                    
                    if not return_csv:
                        lines_to_draw.append(a1)
                        lines_to_draw.append(b1)                

    typeTWOs = []

    for t in type2:
        sS, eS = t[0][0], t[1][0]
        sD, eD = t[0][1], t[1][1]
        ss = max(sS, sD)
        ee = min(eS, eD)
        stockS = data.iloc[ss].high
        stockE = data.iloc[ee].high
        dds = D.iloc[ss]
        dde = D.iloc[ee]

        if not eS == sS and not sD == eD:
            StockM = (stockE - stockS)/(eS-sS)
            Dm = (dde - dds)/(eS-sS)

            if StockM > 0.2 and Dm > 0.2:
                pass
            elif StockM < -0.2 and Dm < -0.2:
                pass
            else:
                start = max(t[0][1], t[0][0])
                ending = min(t[1])
                stockStart = start
                stockEnd = ending

                dStart = start
                dEnd = ending

                if True:#data.iloc[stockStart].high < data.iloc[stockEnd].high: # Nikola Added
                    a1 = dict(
                        x0 = data.iloc[dStart].name,
                        y0 = D.iloc[dStart],
                        x1 = data.iloc[dEnd].name,
                        y1 = D.iloc[dEnd],
                        type = 'line',
                        xref = 'x2',
                        yref = 'y2',
                        line_width = 4
                    )
                    a2 = dict(
                        x0 = data.iloc[stockStart].name,
                        y0 = data.iloc[stockStart].high,
                        x1 = data.iloc[stockEnd].name,
                        y1 = data.iloc[stockEnd].high,
                        type = 'line',
                        xref = 'x',
                        yref = 'y',
                        line_width = 4
                    )
                    typeTWOs.append((a1, a2))
                    
                    if not return_csv:
                        lines_to_draw.append(a1)
                        lines_to_draw.append(a2)

    if not return_csv:
        if cur_date is not None: lines_to_draw = [d for d in lines_to_draw if d['x1'] < cur_date]
        fig.update_layout(shapes = lines_to_draw)

        fig.update_xaxes(
            rangeslider_visible = False,
            range = [df.index[0], df.index[-1]],
            row = 1, col = 1
        )
        fig.update_xaxes(
            rangeslider_visible = False,
            range = [df.index[0], df.index[-1]],
            row = 2, col = 1
        )

    if return_csv: return typeONEs, typeTWOs, df
    return fig, data

def append_divergence_record(symbol1, symbol2, sign, start_date, end_date):
    if not os.path.exists(DIVERGENCE_RECORDS_PATH):
        with open(DIVERGENCE_RECORDS_PATH, 'w') as fp:
            fp.write('Symbol1,Symbol2,Type,StartDate,EndDate,RecordDate\n')

    with open(DIVERGENCE_RECORDS_PATH,'a') as fp:
        typeStr = 'Bullish' if sign > 0 else 'Bearish'
        fp.write(f'{symbol1},{symbol2},{typeStr},{start_date.strftime(YMD_FORMAT)},{end_date.strftime(YMD_FORMAT)},{datetime.today().strftime(YMD_FORMAT)}\n')

def diffrenciate(xout, yout,xx,yy):
    h = xout[1] - xout[0]
    yPrime = np.diff(yout)
    yPrime = yPrime/h
    xPrime = xx[1:]
    xPs, yPs,  _ = loess_1d(xPrime, yPrime, xnew=None, degree=1, frac=0.05, npoints=None, rotate=False, sigy=None)
    return xPs, yPs, xPrime, yPrime

def getInbetween(pair, minIndexes, xx,yy):
    allIndex = minIndexes
    if yy[pair[1]] < yy[pair[0]]:
        min_ = yy[pair[1]]
        max_ = yy[pair[0]]
    else:
        min_ = yy[pair[0]]
        max_ = yy[pair[1]]
    c = 0
    for a in allIndex:
        if min_ < yy[a] < max_:
            #print(min_, yy[a], max_)
            c += 1
    if c < 1:
        return False
    return True

def getPairs(xx, yy, loesFraction, primeLoessFraction=0.05):
    xout, yout, _ = loess_1d(xx, yy, xnew=None, degree=1, frac=loesFraction, npoints=None, rotate=False, sigy=None)

    xPs, yPs, xPrime, yPrime = diffrenciate(xout, yout,xx,yy)
    h = xout[1] - xout[0]
    yPrimePrime = np.diff(yPs)
    yPrimePrime = yPrimePrime/h
    xPrimePrime = xx[2:]
    xPPs, yPPs, _ = loess_1d(xPrimePrime, yPrimePrime, xnew=None, degree=1, frac=primeLoessFraction, npoints=None, rotate=False, sigy=None)
    minIndexes = []
    maxIndexes = []
    i = 1
    while i < len(yPs):
        if yPs[i-1] < 0 and yPs[i] > 0:
            minIndexes.append(i)
        elif yPs[i-1] > 0 and yPs[i] < 0:
            maxIndexes.append(i)
        i += 1

    minIndexesCheck = []
    maxIndexesCheck = []
    i = 1
    while i < len(yPs):
        if yPs[i-1] < 0 and yPs[i] > 0:
            minIndexesCheck.append(i)
        elif yPs[i-1] > 0 and yPs[i] < 0:
            maxIndexesCheck.append(i)
        i += 1

    allIndexCheck = sorted(minIndexes+maxIndexes)

    allIndex = sorted(minIndexes+maxIndexes)
    allDiffs = []
    i = 0
    while i+1 < len(allIndex):
        a = xout[allIndex[i]+1]
        b = xout[allIndex[i+1]]
        allDiffs.append( [abs(a-b), (allIndex[i], allIndex[i+1])] )
        i += 1

    pairsInnit = [(minIndex, maxIndex) for minIndex in minIndexes for maxIndex in maxIndexes]

    pairs = []
    for diff in allDiffs:
        if getInbetween(diff[1], allIndexCheck, xx,yy) and abs(diff[1][0] - diff[1][1]) > 1:
            pairs.append(diff[1])

    #print(len(pairs), len(allDiffs))
    return pairs

def plotter1(pairs, xx, yy, xScaler, yScaler, intervalSET, data, y):
    figures=[]
    cols = ['black', 'blue', 'green', "yellow"]
    currentPrice = yScaler.getScaledvalue(y[-1])
    #print(currentPrice)
    #print(data)

    #data = data.copy()
    data['Date'] = list(data.index)
    plotted = []
    data = data.reset_index()    
    data['Date1'] = data['Date'].map(mdates.date2num)

    #print("date",data)
    j = 0

    for pp in pairs:
        #print(pp, "uy4rwegk")

        if yy[pp[0]] < yy[pp[1]]:
            min_chosen = yy[pp[0]]
            max_chosen = yy[pp[1]]
            p = (pp[0], pp[1])
        else:
            max_chosen = yy[pp[0]]
            min_chosen = yy[pp[1]]
            p = (pp[1], pp[0])

        min_ = min(yy[max(0, p[0] - 7):p[0] + 7])
        max_ = max(yy[max(0, p[1] - 7):p[1] + 7])

        max_index = np.argmax(np.asarray(yy[max(0, p[0] - 7):p[0] + 7]))
        min_index = np.argmin(np.asarray(yy[max(0, p[0] - 7):p[0] + 7]))

        actualMinINdex = min_index - 7 + max(0, p[0] - 7)
        actualMaxINdex = max_index - 7 + max(0, p[1] - 7)

        firstDate = data['Date1'][0]
        f1 = data['Date'][0]
        l1 = data['Date'][len(data)-1]

        if intervalSET == INTERVAL_WEEKLY:
            plottedPoint = [
                (xScaler.getUnscaledValue(min(xx[p[0]], xx[p[0]])) * 7) + firstDate,
                (xScaler.getUnscaledValue(xx[p[1]]) * 7) + firstDate
            ], [
                10 ** yScaler.getUnscaledValue(min_),
                10 ** yScaler.getUnscaledValue(min_)
            ]
        if intervalSET == INTERVAL_MONTHLY:
            plottedPoint = [
                (xScaler.getUnscaledValue(min(xx[p[0]], xx[p[0]])) * 30.5) + firstDate,
                (xScaler.getUnscaledValue(min(xx[p[1]], xx[p[1]])) * 30.5) + firstDate
            ], [
                10 ** yScaler.getUnscaledValue(min_),
                10 ** yScaler.getUnscaledValue(min_)
            ]
        
        #print(min_chosen <= max_chosen, "jhbjhbn")

        if not plottedPoint in plotted:
            plotted.append(plottedPoint)

            candlestick = go.Candlestick(
                x = data['Date'],
                open = data['open'],
                high = data['high'],
                low = data['low'],
                close = data['close'],
                increasing_line_color = 'green',
                decreasing_line_color = 'red'
            )
            layout = go.Layout(
                title='Candlestick Chart',
                yaxis=dict(type='log', autorange=True),
                xaxis=dict(
                    type='date',
                    tickformat='%Y-%m-%d',  
                    rangeslider=dict(visible=False)
                )
            )

            fig = go.Figure(data=[candlestick], layout=layout)

            fig.update_xaxes(
                rangeslider=dict(visible=False)
            )
            fig.update_layout(
            autosize=False,
            width=1200,
            height=800,)

            fig.add_trace(go.Scatter(
                x=[f1,
                l1],
                y=[10 ** yScaler.getUnscaledValue(min_), 10 ** yScaler.getUnscaledValue(min_)],
                mode='lines'
            ))

            fig.add_trace(go.Scatter(
                        x=[f1,
                l1],
                y=[10 ** yScaler.getUnscaledValue(max_), 10 ** yScaler.getUnscaledValue(max_)],
                mode='lines'
            ))

            nums = [0.2366, 0.382, 0.5, 0.618, 0.764, 0.786, 0.886]
            for num in nums:
                if intervalSET == INTERVAL_WEEKLY:
                    y = 10 ** yScaler.getUnscaledValue(max_) - (
                            (10 ** yScaler.getUnscaledValue(max_) - 10 ** yScaler.getUnscaledValue(min_)) * num)
                    fig.add_trace(go.Scatter(
                        x=[f1,l1],
                        y=[y, y],
                        mode='lines',
                        name=str(num)
                    ))
                elif intervalSET == INTERVAL_MONTHLY:
                    y = 10 ** yScaler.getUnscaledValue(max_) - (
                            (10 ** yScaler.getUnscaledValue(max_) - 10 ** yScaler.getUnscaledValue(min_)) * num)
                    fig.add_trace(go.Scatter(
                        x=[f1,l1],
                        y=[y, y],
                        mode='lines',
                        name=str(num)
                    ))
                

            figures.append(fig)

        j += 1

    return figures

def get_divergence_data(stock_symbol, stdate, endate, oldData):
        year, month, day = map(int, stdate.split('-'))
        sdate = date(year, month, day)
        
        year1, month1, day1 = map(int, endate.split('-'))
        edate = date(year1, month1, day1 )
    
        COMMON_START_DATE = sdate
        STOCK = stock_symbol

        days = pd.date_range(sdate, edate, freq = 'd').strftime('%Y-%m-%d').tolist()
        TT1s, TT2s = [], []

        for dd in tqdm(days):
            # Calculate divergence and get transactions
            type1, type2, df = runStochDivergance(STOCK, COMMON_START_DATE, dd, True, old_data = oldData[:dd])
            t1s = []
            
            for t in type1:
                stockPart = t[1]
                indicatorPart = t[0]
                startDate = stockPart['x0']
                endDate = stockPart['x1']
                DvalueStart = indicatorPart['y0']
                DvalueEnd = indicatorPart['y1']
                stockValueStart = stockPart['y0']
                stockValueEnd = stockPart['y1']
                t1s.append((startDate, endDate, DvalueStart, DvalueEnd, stockValueStart, stockValueEnd, dd))
            
            t2s = []
            
            for t in type2:
                stockPart = t[1]
                indicatorPart = t[0]
                startDate = stockPart['x0']
                endDate = stockPart['x1']
                DvalueStart = indicatorPart['y0']
                DvalueEnd = indicatorPart['y1']
                stockValueStart = stockPart['y0']
                stockValueEnd = stockPart['y1']
                t2s.append((startDate, endDate, DvalueStart, DvalueEnd, stockValueStart, stockValueEnd, dd))
            
            TT1s.append(t1s)
            TT2s.append(t2s)

        def find_unique_smallest_date(arrays_of_tuples):
            unique_tuples = defaultdict(list)

            for arr in arrays_of_tuples:
                for tup in arr:
                    key = tuple(tup[:-1])
                    date_str = tup[-1]

                    if key not in unique_tuples or date_str < unique_tuples[key][-1][-1]:
                        unique_tuples[key] = [(tup, date_str)]

            result = [min(tups, key = lambda x: x[-1]) for tups in unique_tuples.values()]
            return result

        def rearrange(od):
            od = [list(t[0]) for t in od]
            od.sort(key = lambda x : x[0])
            
            recs = []

            for i in range(len(od)):
                tr = od[i]
                if np.sign(tr[2] - tr[3]) == np.sign(tr[4] - tr[5]): continue
                
                found_in = False
                
                for j in range(len(od)):
                    if i == j: continue
                    otr = od[j]
                    
                    if otr[0] <= tr[0] and tr[1] <= otr[1]:
                        found_in = True
                        break
                
                if found_in: continue
                recs.append(tr)
            
            # while True:
            #     found_adj = False
                
            #     for i in range(len(recs) - 1):
            #         tr, otr = recs[i], recs[i + 1]
                    
            #         if tr[1] == otr[0]:
            #             tr[1] = otr[1]
            #             tr[3] = otr[3]
            #             tr[5] = otr[5]
            #             tr[6] = otr[6]
                        
            #             found_adj = True
            #             recs.pop(i + 1)
            #             break
                
            #     if not found_adj: break
            
            new_recs = []
            
            for startDate, endDate, DvalueStart, DvalueEnd, stockValueStart, stockValueEnd, dd in recs:
                if (datetime.strptime(dd, YMD_FORMAT) - endDate).days <= 14:
                    new_recs.append((startDate, endDate, DvalueStart, DvalueEnd, stockValueStart, stockValueEnd, dd))
            
            return new_recs

        out1 = find_unique_smallest_date(TT1s)
        out2 = find_unique_smallest_date(TT2s)
        
        out1 = rearrange(out1)
        out2 = rearrange(out2)
        
        return out1, out2


######### causing some identation issue ##########
# Get the closes date point that is included in data
def getClosestPrevIndex(start, data, type):
    min_ = 10000000000
    selected = None
    
    if type == 'start':
        for d in data:
            if start - d > 0:
                if start - d < min_:
                    min_ = start - d
                    selected = d
                    
        return selected
    else:
        for d in data:
            if start - d < 0:
                if -start + d < min_:
                    min_ = -start + d
                    selected = d
                    
        return selected