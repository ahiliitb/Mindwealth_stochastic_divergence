
from dash import dcc, callback, Output, Input, State
from datetime import date, datetime, timedelta
from plotly.subplots import make_subplots
from collections import defaultdict
from constant import *
from tqdm import tqdm
from compute import *
from config import *
from yahoo import *
from plot import *
from util import *
from data import *
from ui import *
from pages.general_divergence import general_div_for_crossover_buys
import plotly.graph_objects as go
import numpy as np
import dash
from scipy.signal import find_peaks
from stoploss import calculate_stoploss_profit
import itertools
import json
import csv

dash.register_page(__name__, path = '/divergence', name = 'Stochastic Divergence', order = '04')

# Page Layout
scenario_div = get_scenario_div([
	get_symbol_input(),
	get_date_range(from_date = get_offset_date_str(get_today_str(), -365)),
    get_run_button('diver'),
])
# parameter_div = get_parameter_div([
#     get_run_button('diver'),
# 	#get_cur_date_picker(),
# 	# get_analyze_button('diver'),
# 	# get_backtest_button('diver')
# ])
out_tab = get_out_tab({
	'Plot': get_plot_div(),
	'Report': get_report_div()
})
layout = get_page_layout('Stochastic|Divergence', scenario_div, None, out_tab)

# Triggered when Analyze button clicked
@callback(
	[
		Output('alert-dlg', 'is_open', allow_duplicate = True),
		Output('alert-msg', 'children', allow_duplicate = True),
		Output('alert-dlg', 'style', allow_duplicate = True),
        Output('out_tab', 'value', allow_duplicate = True),
		Output('out-plot', 'children', allow_duplicate = True),
        Output('out-report', 'children', allow_duplicate = True)
	],
	Input('diver-run-button', 'n_clicks'),
	[
		State('symbol-input', 'value'),
		State('from-date-input', 'date'),
		State('to-date-input', 'date'),
        #State('cur-date-input', 'date')
	],
	prevent_initial_call = True
)
def on_run_clicked(n_clicks, symbol, from_date, to_date):
    #print("divergence runs from ", from_date," to ", to_date)
    none_ret = ['Plot', None, None] # Padding return values

    if n_clicks == 0: return alert_hide(none_ret)

    if symbol is None: return alert_error('Invalid symbol. Please select one and retry.', none_ret)
    if from_date is None: return alert_error('Invalid starting date. Please select one and retry.', none_ret)
    if to_date is None: return alert_error('Invalid ending date. Please select one and retry.', none_ret)
    if from_date > to_date: return alert_error('Invalid duration. Please check and retry.', none_ret)

    # if cur_date is None: return alert_error('Invalid current date. Please select one and retry.', none_ret)
    
    # if cur_date < from_date: cur_date = from_date
    # if cur_date > to_date: cur_date = to_date
    
    # cur_date = get_timestamp(cur_date)
    #fig, df = runStochDivergance(symbol, from_date, to_date, cur_date = cur_date)

    # with open('json_file/tolerance.json', 'r') as f:
    #     tolerance_dict = json.load(f)
    
    # month_list = tolerance_dict[symbol][0]
    # i = 0
    # today_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    # to_test = 0
    # while len(month_list) != 0 and month_list[0] <= today_date:
    #     to_test = 1
    #     month_list = month_list[1:]

    # if to_test:
    #     input_date = datetime.strptime(to_date, "%Y-%m-%d")
    #     one_year_ago = input_date - timedelta(days=365)
    #     one_year_ago_str = one_year_ago.strftime("%Y-%m-%d")

    #     bull_entry, bear_entry, bull_exit, bear_exit = get_best_tolerance(symbol, one_year_ago_str, today_date)
    #     tolerance_dict[symbol][0] = month_list
    #     tolerance_dict[symbol][1] = (bull_entry, bear_entry, bull_exit, bear_exit)
    #     with open('json_file/tolerance.json', 'w') as f:
    #         json.dump(tolerance_dict, f)
    # else:
    #     bull_entry, bear_entry, bull_exit, bear_exit = tolerance_dict[symbol][1]


    
    df, _, _ = getPointsGivenR(symbol, 1.02, startDate = from_date, endDate = to_date)

    if(len(df) == 0):
        return alert_success('Input Dates are before IPO!') + ['Plot', None, None]

    # tempdf = yf.download(symbol, from_date, to_date)
    # tempdf.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

    # if symbol == "EURUSD=X":
    #     tempdf = 100*tempdf
    #     df = tempdf


    K = TA.STOCH(df,14)
    D = TA.STOCHD(df)

    buy_crossover_dates, buy_intersection_levels = get_buy_crossover_dates(K,D)
    buy_crossover_dates = [date.strftime('%Y-%m-%d') for date in buy_crossover_dates]
    
    from_date = get_nearest_forward_date(df, get_timestamp(from_date)).strftime(YMD_FORMAT)
    out1, out2 = get_divergence_data(symbol, from_date, to_date, oldData = df)

    # input_date = datetime.strptime(from_date, "%Y-%m-%d")
    # one_year_ago = input_date - timedelta(days=365)
    # one_year_ago_str = one_year_ago.strftime("%Y-%m-%d")
    # input_date_str = input_date.strftime("%Y-%m-%d")

    # bull_entry, bear_entry, bull_exit, bear_exit = get_best_tolerance(symbol, one_year_ago_str, input_date_str)
    bull_entry, bear_entry, bull_exit, bear_exit = 1, 2, 2, 1
    
    # correct_buy_entry_dates,correct_buy_exit_dates,buy_results_df = buy_strategy(symbol,from_date,to_date,buy_crossover_dates,buy_intersection_levels,D,out1)
    
    sell_crossover_dates, sell_intersection_levels = get_sell_crossover_dates(K,D)
    sell_crossover_dates = [date.strftime('%Y-%m-%d') for date in sell_crossover_dates]
    # correct_sell_entry_dates,correct_sell_exit_dates,sell_results_df = sell_strategy(symbol,from_date,to_date,sell_crossover_dates,sell_intersection_levels,D,out2)
    #empty_row = pd.Series([None] * len(buy_results_df.columns), name='empty_row')
    # empty_row = pd.DataFrame([[''] * len(buy_results_df.columns)], columns=buy_results_df.columns)

    # Concatenating dataframes with an empty row in between
    #merged_df = pd.concat([buy_results_df, empty_row, sell_results_df], ignore_index=True)
    # merged_df = pd.concat([buy_results_df, sell_results_df], ignore_index=True)
    csv_path = 'out/CROSSOVERS_{}_{}.csv'.format(symbol, to_date)
    
    # merged_df.to_csv(csv_path, index = False)
    fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True, vertical_spacing = 0.1, subplot_titles = ('Stock prices '+ symbol, 'Stochastic Indicator '+ symbol), row_width = [0.29,0.7])
    fig.update_yaxes(type = 'log', row = 1, col = 1)
    fig.add_trace(go.Candlestick(x = df.index, open = df['open'], high = df['high'], low = df['low'], close = df['close'], showlegend = False), row = 1, col = 1)
    fig.update_layout(xaxis_rangeslider_visible = False, yaxis_tickformat = '0')
    
    fig.add_trace(go.Scatter(x = D.index, y = D, showlegend = False), row = 2, col = 1)
    fig.add_trace(go.Scatter(x = df.index, y = df['close'].rolling(10).mean(), name = 'MA-10W'))
    fig.add_trace(go.Scatter(x = df.index, y = df['close'].rolling(40).mean(), name = 'MA-40W'))

    res_df, acc, cum, full_list = backtest_stock_divergence(symbol, from_date, to_date, out1, out2, bull_entry, bear_entry, bull_exit, bear_exit)

    lines_to_draw = []
    smarker_x, smarker_y = [], []
    emarker_x, emarker_y = [], []
    Dsmarker_x, Dsmarker_y = [], []
    Demarker_x, Demarker_y = [], []
    # nmarker_x, nmarker_y = [], []
    # nemarker_x, nemarker_y = [], []
    # newmarker_x, newmarker_y = [], []

    # for entry_date in correct_buy_entry_dates:
    #     #print("correct_buy_entry_dates are ", correct_buy_entry_dates)
    #     nmarker_x.append(entry_date) 
    #     nmarker_y.append(df.loc[entry_date].close)
        

    # for exit_date in correct_buy_exit_dates:
    #     nemarker_x.append(exit_date) 
    #     nemarker_y.append(df.loc[exit_date].close)

    # for entry_date in correct_sell_entry_dates:
    #     #print("correct_sell_entry_dates are ",correct_sell_entry_dates)
    #     newmarker_x.append(entry_date) 
    #     newmarker_y.append(df.loc[entry_date].close)
    

    for dStart, dEnd, dVStart, dVEnd, sv, se, dd, edd in full_list:

        lines_to_draw.extend(get_lines(dStart, dEnd, df, D, dd, edd, dVEnd > dVStart))
        if dd is not None:
            smarker_x.append(dd) 
            smarker_y.append(df.loc[dd].low if dVEnd > dVStart else df.loc[dd].high)
            Dsmarker_x.append(dd)
            Dsmarker_y.append(D.loc[dd])

        if edd is not None:
            emarker_x.append(edd)
            emarker_y.append(df.loc[edd].low if dVEnd > dVStart else df.loc[edd].high)
            Demarker_x.append(edd)
            Demarker_y.append(D.loc[edd])
    
    fig.update_layout(shapes = lines_to_draw)

#     fig.add_trace(
#     go.Scatter(
#         x=newmarker_x,
#         y=newmarker_y,
#         mode="markers",
#         marker=dict(
#             size=15,
#             color='purple'
#         ),
#         marker_symbol="triangle-down",  # Change this to the desired symbol
#         name='Entry_Date'
#     ),
#     row=1,
#     col=1
# )
#     fig.add_trace(
#     go.Scatter(
#         x=nmarker_x,
#         y=nmarker_y,
#         mode="markers",
#         marker=dict(
#             size=15,
#             color='black'
#         ),
#         marker_symbol="triangle-up",  # Change this to the desired symbol
#         name='EntryDate'
#     ),
#     row=1,
#     col=1
# )

#     fig.add_trace(
#     go.Scatter(
#         x=nemarker_x,
#         y=nemarker_y,
#         mode="markers",
#         marker=dict(
#             size=15,
#             color='black'
#         ),
#         marker_symbol="triangle-up",  # Change this to the desired symbol
#         name='EntryDate'
#     ),
#     row=1,
#     col=1
# )

    # for entry_x, entry_y, exit_x, exit_y in zip(nmarker_x, nmarker_y, nemarker_x, nemarker_y):
    #     fig.add_trace(
    #         go.Scatter(
    #             x=[entry_x, exit_x],
    #             y=[entry_y, exit_y],
    #             mode="lines",
    #             line=dict(
    #                 color='black',
    #                 width=2,
    #                 dash='dot'
    #             ),
    #             showlegend=False
    #         ),
    #         row=1,
    #         col=1
    #     )
    
    fig.add_trace(
        go.Scatter(
            x = smarker_x,
            y = smarker_y,
            mode = "markers",
            marker = dict(
                size = 7,
                color = 'purple'
                # color='rgba(0, 0, 0, 0)'
            ),
            marker_symbol = "circle",
            name = 'EntryDate'
        ),
        row = 1,
        col = 1
    )
    # remove all of these from comments
    fig.add_trace(
        go.Scatter(
            x = emarker_x,
            y = emarker_y,
            mode = "markers",
            marker = dict(
                size = 7,
                color = 'orange'
                # color='rgba(0, 0, 0, 0)'
            ),
            marker_symbol = "circle",
            name = 'ExitDate'
        ),
        row = 1,
        col = 1
    )
    fig.add_trace(
        go.Scatter(
            x = Dsmarker_x,
            y = Dsmarker_y,
            mode = "markers",
            marker = dict(
                size = 7,
                #color = 'purple'
                color='rgba(0, 0, 0, 0)'
            ),
            marker_symbol = "circle",
            showlegend= False
        ),
        row = 2,
        col = 1
    )
    fig.add_trace(
        go.Scatter(
            x = Demarker_x,
            y = Demarker_y,
            mode = "markers",
            marker = dict(
                size = 7,
                #color = 'orange'
                color='rgba(0, 0, 0, 0)'
            ),
            marker_symbol = "circle",
            showlegend=False
        ),
        row = 2,
        col = 1
    )
    fig.update_xaxes(
        rangeslider_visible = False,
        range = [df.index[0], df.index[-1]],
        showticklabels=True,
        row = 1, col = 1
    )
    fig.update_xaxes(
        rangeslider_visible = False,
        range = [df.index[0], df.index[-1]],
        showticklabels=True,
        row = 2, col = 1
    )
    
    csv_path = 'out/DIVERGENCE-REPORT_{}_{}_{}_sr={:.1f}%_cp={:.1f}%.csv'.format(
		symbol, from_date, to_date, acc, cum
    )
    res_df.to_csv(csv_path, index = False)    
    return alert_success('Analysis Completed') + ['Plot', dcc.Graph(figure = fig, className = 'diver_graph'), get_report_content(res_df, csv_path)]

# def get_earlier_date(symbol, start_date_str):
#     # Convert start_date_str to a datetime object
#     today = datetime.strptime(start_date_str, '%Y-%m-%d')

#     # Calculate the end date (2 months later)
#     earlier = today - timedelta(days=120)

#     # Fetch historical stock data using yfinance
#     #print("get_earlier_date ",earlier, " to ", today )
#     stock_data = yf.download(symbol, start=earlier, end=today)

#     # Extract dates from the data
#     dates = stock_data.index.strftime('%Y-%m-%d').tolist()

#     return dates[0]

# def get_stock_dates(start_date_str):
#     symbol='AAPL'
#     # Convert start_date_str to a datetime object
#     start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

#     # Calculate the end date (2 months later)
#     end_date = start_date + timedelta(days=120)

#     # Fetch historical stock data using yfinance
#     #print("get_stock_dates ", start_date, " to ", end_date)
#     stock_data = yf.download(symbol, start=start_date, end=end_date)

#     # Extract dates from the data
#     dates = stock_data.index.strftime('%Y-%m-%d').tolist()

#     return dates

# def get_bullishness(symbol,start_date,end_date,window=15):
#     try:
#         #("get_bullishness ",start_date, " to ",end_date  )
#         stock_data = yf.download(symbol, start=start_date, end=end_date)

#         if stock_data.empty:
#             raise ValueError(f"No data available for {symbol} between {start_date} and {end_date}")

#         # Calculate the moving average
#         moving_average = stock_data['Close'].rolling(window=window).mean()

#         # Check if the most recent closing price is above the moving average
#         last_close = stock_data['Close'].iloc[-1]
#         last_ma = moving_average.iloc[-1]
        
#         return last_close - last_ma

#     except Exception as e:
#         print(f"Error: {e}")
#         return False
# def get_most_recent_date(out1, entry_date):
#     # Filter out tuples where the 7th element is less than the entry_date
#     filtered_dates = [date for date in out1 if date[6] < entry_date]

#     if not filtered_dates:
#         return None  # No matching dates found

#     # Find the tuple with the most recent date
#     most_recent_date_tuple = max(filtered_dates, key=lambda x: datetime.strptime(x[6], '%Y-%m-%d'))
#     ss = most_recent_date_tuple[0]
#     se = most_recent_date_tuple[1]
#     #ss = datetime.strptime(ss, "%Y-%m-%d")
#     #se = datetime.strptime(se, "%Y-%m-%d")
#     time_difference = se - ss
#     days_difference = time_difference.days
#     i_s = most_recent_date_tuple [2]
#     i_e = most_recent_date_tuple [3]
#     p_s = most_recent_date_tuple [4]
#     p_e = most_recent_date_tuple [5]
#     indicator_slope = (i_e - i_s)/days_difference
#     price_slope = (p_e - p_s)/days_difference
    
#     if most_recent_date_tuple[6] is None:
#         print("None")
#         return None,None,None

#     # Return the most recent date
#     #print("returning")
#     #return most_recent_date_tuple[6], indicator_slope, price_slope
#     return most_recent_date_tuple[6]

# def buy_strategy(stock_symbol,from_date,end_date,buy_crossover_dates,intersection_levels,d_levels,out1):
#     results_list = []
#     correct_entry_dates = []
#     correct_exit_dates = []
#     #d_levels.to_csv("D levels.csv", index = True)
#     for index, crossover_date in enumerate(buy_crossover_dates):
#         datess = get_stock_dates(crossover_date)
#         start_date = datess[1]
#         #end_put_date, divergence_indicator_slope, divergence_price_slope = get_most_recent_date(out1,start_date)
#         end_put_date = get_most_recent_date(out1,start_date)
#         if end_put_date is None:
#             continue
#         end_put_date = datetime.strptime(end_put_date, '%Y-%m-%d')
#         crossover_date_dt = datetime.strptime(crossover_date, '%Y-%m-%d')
#         from_for_inter = crossover_date_dt - timedelta(days = 100)
#         from_for_inter = from_for_inter.strftime("%Y-%m-%d")
#         date1 = "2023-01-31"
#         date2 = "2023-01-01"
#         date1 = datetime.strptime(date1, '%Y-%m-%d')
#         date2 = datetime.strptime(date2, '%Y-%m-%d')
#         difference_in_dates = crossover_date_dt - end_put_date
#         if not difference_in_dates < date1-date2:
#             continue
#         inter_div_date = general_div_for_crossover_buys(stock_symbol, from_for_inter, crossover_date)
#         #print("inter_div_date is ",inter_div_date," for crossover_date = ",crossover_date)
#         earlier_date = get_earlier_date(stock_symbol,crossover_date)
#         bullish_level = get_bullishness(stock_symbol,earlier_date,crossover_date)
#         #correct_entry_dates.append(end_date)
#         intersection_level = intersection_levels[index]
#         temporary_d_levels = d_levels.loc[crossover_date:end_date]
#         d_crossover_date = temporary_d_levels[0]
#         d_entry_date = temporary_d_levels[0]
#         #print("new_strategy_backtest ", start_date, " to ",end_date)
#         if start_date >= end_date:
#             #print("problem case ", start_date)
#             none = None
#             results_list.append({
#             "Crossover Date": crossover_date,
#             "Intersection Level": intersection_level,
#             "Crossover D": d_crossover_date,
#             "Entry D": d_entry_date,
#             "Exit after D": none,
#             "Bullish level": bullish_level,
#             "Entry Date": end_date,
#             "Exit Date": none,
#             "Entry Price": none,
#             "Exit Price": none,
#             "Profit Percentage" : none,
#             "Signal Type": "Long",
#             "Symbol": stock_symbol
#         })
#             continue
#         data = yf.download(stock_symbol, start=crossover_date, end=end_date)
#         a = data['Close'].values
#         #print("the cross_date is ", crossover_date)
#         cross_date = datetime.strptime(crossover_date, '%Y-%m-%d')
#         data_for_d_levels = yf.download(stock_symbol, start=cross_date - timedelta(days = 120), end=cross_date)

#         d_levels_for_exit = TA.STOCHD(data_for_d_levels)
#         d_levels_for_exit = d_levels_for_exit.dropna()
#         d_for_exit = get_d_for_exit_long(d_levels_for_exit)
#         #print("exit d for crossover date ", crossover_date, " should be ", d_for_exit)
#         start_price = a[0]
#         sl,tp=calculate_stoploss_profit(stock_symbol,'1d')
#         threshold_loss = sl
#         changed = False
#         stoploss = False
#         for i in range(2, len(a)):
#             current_price = a[i]
#             if temporary_d_levels[i]>d_for_exit and current_price > start_price:
#                 #print("profit case")
#                 exit_after_d = temporary_d_levels[i]
#                 result  = i
#                 changed = True
#                 break # Take-Profit reached

#             if current_price <= start_price * (1 - threshold_loss): #and current_price < a[i - 1]:
#                 #print(current_price, " and ",start_price * (1 - threshold_loss))
#                 #print("stoploss case")
#                 result  = i
#                 exit_after_d = temporary_d_levels[i]
#                 changed = True
#                 stoploss = True
#                 break # Stop-Loss reached

#         if not changed:
#             #print("No maximas/minimas to exit")
#             none = None
#             entry_date = data.index[1]
#             entry_price = a[1]
#             results_list.append({
#             "Crossover Date": crossover_date,
#             "Intersection Level": intersection_level,
#             "Crossover D": d_crossover_date,
#             "Entry D": d_entry_date,
#             "Exit after D": none,
#             "Bullish level": bullish_level,
#             "End put date": end_put_date,
#             "Interdivergence Date": inter_div_date,
#             #"Divergence Indicator Slope": divergence_indicator_slope,
#             #"Divergence Price Slope": divergence_price_slope,
#             "Entry Date": entry_date,
#             "Exit Date": none,
#             "Entry Price": entry_price,
#             "Exit Price": none,
#             "Profit Percentage" : none,
#             "Signal Type": "Long",
#             "Symbol": stock_symbol
#         })
#             continue
    
#         exit_date = data.index[result]
#         entry_date = data.index[0]
#         entry_price = a[0]
#         exit_price = a[result]
#         correct_entry_dates.append(entry_date)
#         correct_exit_dates.append(exit_date)
#         profit_percentage = (exit_price-entry_price)*100/entry_price
#         if stoploss:
#             profit_percentage = -threshold_loss*100

#         results_list.append({
#             "Crossover Date": crossover_date,
#             "Intersection Level": intersection_level,
#             "Crossover D": d_crossover_date,
#             "Entry D": d_entry_date,
#             "Exit after D": d_for_exit,
#             "Bullish level": bullish_level,
#             "End put date": end_put_date,
#             "Interdivergence Date": inter_div_date,
#             #"Divergence Indicator Slope": divergence_indicator_slope,
#             #"Divergence Price Slope": divergence_price_slope,
#             "Entry Date": entry_date,
#             "Exit Date": exit_date,
#             "Entry Price": entry_price,
#             "Exit Price": exit_price,
#             "Profit Percentage" : profit_percentage,
#             "Signal Type": "Long",
#             "Symbol": stock_symbol
#         })

#     results_df = pd.DataFrame(results_list)
#     #exit_dates = results_df["Exit Date"].values
#     csv_path = 'out/CROSSOVERS_{}_{}.csv'.format(
# 		stock_symbol, end_date)
    
#     #results_df.to_csv(csv_path, index = False)
#     #return exit_dates
#     return correct_entry_dates,correct_exit_dates,results_df

# def sell_strategy(stock_symbol,from_date,end_date,sell_crossover_dates,intersection_levels,d_levels,out2):
#     results_list = []
#     correct_entry_dates = []
#     correct_exit_dates = []
#     #d_levels.to_csv("D levels.csv", index = True)
#     #print("these are d_levels ",d_levels)
#     for index, crossover_date in enumerate(sell_crossover_dates):
#         datess = get_stock_dates(crossover_date)
#         start_date = datess[1]
#         #end_put_date, divergence_indicator_slope, divergence_price_slope = get_most_recent_date(out1,start_date)
#         end_put_date = get_most_recent_date(out2,start_date)
#         if end_put_date is None:
#             continue
#         end_put_date = datetime.strptime(end_put_date, '%Y-%m-%d')
#         crossover_date_dt = datetime.strptime(crossover_date, '%Y-%m-%d')
#         from_for_inter = crossover_date_dt - timedelta(days = 100)
#         from_for_inter = from_for_inter.strftime("%Y-%m-%d")
#         date1 = "2023-01-31"
#         date2 = "2023-01-01"
#         date1 = datetime.strptime(date1, '%Y-%m-%d')
#         date2 = datetime.strptime(date2, '%Y-%m-%d')
#         difference_in_dates = crossover_date_dt - end_put_date
#         if not difference_in_dates < date1-date2:
#             continue
#         inter_div_date = general_div_for_crossover_buys(stock_symbol, from_for_inter, crossover_date)
#         #print("inter_div_date is ",inter_div_date," for crossover_date = ",crossover_date)
#         earlier_date = get_earlier_date(stock_symbol,crossover_date)
#         bullish_level = get_bullishness(stock_symbol,earlier_date,crossover_date)
#         #correct_entry_dates.append(end_date)
#         intersection_level = intersection_levels[index]
#         temporary_d_levels = d_levels.loc[crossover_date:end_date]
#         d_crossover_date = temporary_d_levels[0]
#         d_entry_date = temporary_d_levels[0]
#         #print("new_strategy_backtest ", start_date, " to ",end_date)
#         if start_date >= end_date:
#             #print("problem case ", start_date)
#             none = None
#             results_list.append({
#             "Crossover Date": crossover_date,
#             "Intersection Level": intersection_level,
#             "Crossover D": d_crossover_date,
#             "Entry D": d_entry_date,
#             "Exit after D": none,
#             "Bullish level": bullish_level,
#             "Entry Date": end_date,
#             "Exit Date": none,
#             "Entry Price": none,
#             "Exit Price": none,
#             "Profit Percentage" : none,
#             "Signal Type": "Short",
#             "Symbol": stock_symbol
#         })
#             continue
#         data = yf.download(stock_symbol, start=crossover_date, end=end_date)
#         a = data['Close'].values
#         #print("the cross_date is ", crossover_date)
#         cross_date = datetime.strptime(crossover_date, '%Y-%m-%d')
#         data_for_d_levels = yf.download(stock_symbol, start=cross_date - timedelta(days = 120), end=cross_date)

#         d_levels_for_exit = TA.STOCHD(data_for_d_levels)
#         d_levels_for_exit = d_levels_for_exit.dropna()
#         d_for_exit = get_d_for_exit_short(d_levels_for_exit)
#         #print("d_levels_for_exit for date = ",crossover_date, " are ", d_levels_for_exit)
#         #print("exit d for crossover date ", crossover_date, " should be ", d_for_exit)
#         start_price = a[0]
#         sl,tp=calculate_stoploss_profit(stock_symbol,'1d')
#         threshold_loss = sl
#         changed = False
#         stoploss_for_short = False
#         for i in range(2, len(a)):
#             current_price = a[i]
#             if temporary_d_levels[i]<d_for_exit+10 and current_price < 0.97*start_price:
#                 #print("profit case for short")
#                 exit_after_d = temporary_d_levels[i]
#                 result  = i
#                 changed = True
#                 break # Take-Profit reached

#             if current_price >= start_price * (1 + threshold_loss): #and current_price < a[i - 1]:
#                 #print("stoploss case for short")
#                 result  = i
#                 exit_after_d = temporary_d_levels[i]
#                 changed = True
#                 stoploss_for_short = True
#                 break # Stop-Loss reached

#         if not changed:
#             #print("No maximas/minimas to exit")
#             none = None
#             entry_date = data.index[1]
#             entry_price = a[1]
#             results_list.append({
#             "Crossover Date": crossover_date,
#             "Intersection Level": intersection_level,
#             "Crossover D": d_crossover_date,
#             "Entry D": d_entry_date,
#             "Exit after D": none,
#             "Bullish level": bullish_level,
#             "End put date": end_put_date,
#             "Interdivergence Date": inter_div_date,
#             #"Divergence Indicator Slope": divergence_indicator_slope,
#             #"Divergence Price Slope": divergence_price_slope,
#             "Entry Date": entry_date,
#             "Exit Date": none,
#             "Entry Price": entry_price,
#             "Exit Price": none,
#             "Profit Percentage" : none,
#             "Signal Type": "Short",
#             "Symbol": stock_symbol
#         })
#             continue
    
#         exit_date = data.index[result]
#         entry_date = data.index[0]
#         entry_price = a[0]
#         exit_price = a[result]
#         correct_entry_dates.append(entry_date)
#         correct_exit_dates.append(exit_date)
#         profit_percentage = (entry_price-exit_price)*100/entry_price
#         if stoploss_for_short:
#             profit_percentage = -threshold_loss*100
#         results_list.append({
#             "Crossover Date": crossover_date,
#             "Intersection Level": intersection_level,
#             "Crossover D": d_crossover_date,
#             "Entry D": d_entry_date,
#             "Exit after D": d_for_exit+10,
#             "Bullish level": bullish_level,
#             "End put date": end_put_date,
#             "Interdivergence Date": inter_div_date,
#             #"Divergence Indicator Slope": divergence_indicator_slope,
#             #"Divergence Price Slope": divergence_price_slope,
#             "Entry Date": entry_date,
#             "Exit Date": exit_date,
#             "Entry Price": entry_price,
#             "Exit Price": exit_price,
#             "Profit Percentage" : profit_percentage,
#             "Signal Type": "Short",
#             "Symbol": stock_symbol
#         })

#     results_df = pd.DataFrame(results_list)
#     #exit_dates = results_df["Exit Date"].values
#     csv_path = 'out/SELL_CROSSOVERS_{}_{}.csv'.format(
# 		stock_symbol, end_date)
    
#     #results_df.to_csv(csv_path, index = False)
#     #return exit_dates
#     return correct_entry_dates,correct_exit_dates,results_df

def get_lines(dStart, dEnd, df, D, dd, edd, is_bullish):
    if dd is not None:
        lines = [dict(
                x0 = df.loc[dStart].name,
                y0 = D.loc[dStart],
                x1 = df.loc[dEnd].name,
                y1 = D.loc[dEnd],
                type = 'line',
                xref = 'x2',
                yref = 'y2',
                line_width = 4,
                line_color = 'blue' if is_bullish else 'black'
            ),
            dict(
                x0 = df.loc[dEnd].name,
                y0 = D.loc[dEnd],
                x1 = df.loc[dd].name,
                y1 = D.loc[dd],
                type = 'line',
                xref = 'x2',
                yref = 'y2',
                line_dash = 'dot',
                line_color = 'blue' if is_bullish else 'black'
            ),        
            dict(
                x0 = df.loc[dStart].name,
                y0 = df.loc[dStart].low if is_bullish else df.loc[dStart].high,
                x1 = df.loc[dEnd].name,
                y1 = df.loc[dEnd].low if is_bullish else df.loc[dEnd].high,
                type = 'line',
                xref = 'x',
                yref = 'y',
                line_width = 4,
                line_color = 'blue' if is_bullish else 'black'
            ),
            dict(
                x0 = df.loc[dEnd].name,
                y0 = df.loc[dEnd].low if is_bullish else df.loc[dEnd].high,
                x1 = df.loc[dd].name,
                y1 = df.loc[dd].low if is_bullish else df.loc[dd].high,
                type = 'line',
                xref = 'x',
                yref = 'y',
                line_dash = 'dot',
                line_color = 'blue' if is_bullish else 'black'
            )
        ]
    else:
        lines = [dict(
                x0 = df.loc[dStart].name,
                y0 = D.loc[dStart],
                x1 = df.loc[dEnd].name,
                y1 = D.loc[dEnd],
                type = 'line',
                xref = 'x2',
                yref = 'y2',
                line_width = 4,
                line_color = 'blue' if is_bullish else 'black'
            ),     
            dict(
                x0 = df.loc[dStart].name,
                y0 = df.loc[dStart].low if is_bullish else df.loc[dStart].high,
                x1 = df.loc[dEnd].name,
                y1 = df.loc[dEnd].low if is_bullish else df.loc[dEnd].high,
                type = 'line',
                xref = 'x',
                yref = 'y',
                line_width = 4,
                line_color = 'blue' if is_bullish else 'black'
            )
        ]
    if edd is not None:
        lines.extend([
            dict(
                x0 = df.loc[dd].name,
                y0 = D.loc[dd],
                x1 = df.loc[edd].name,
                y1 = D.loc[edd],
                type = 'line',
                xref = 'x2',
                yref = 'y2',
                line_dash = 'dot',
                #line_color = 'purple'
                line_color = 'rgba(0, 0, 0, 0)'
            ),
            dict(
                x0 = df.loc[dd].name,
                y0 = df.loc[dd].low if is_bullish else df.loc[dd].high,
                x1 = df.loc[edd].name,
                y1 = df.loc[edd].low if is_bullish else df.loc[edd].high,
                type = 'line',
                xref = 'x',
                yref = 'y',
                line_dash = 'dot',
                line_color = 'purple'
            )
        ])
    return lines

# Triggered when Symbol combo box changed
@callback(
	[
		Output('from-date-input', 'date', allow_duplicate = True),
		Output('cur-date-input', 'date', allow_duplicate = True)
	],
	Input('symbol-input', 'value'),
	[
		State('from-date-input', 'date'), State('cur-date-input', 'date')
	],
	prevent_initial_call = True
)
def on_symbol_changed(symbol, from_date, cur_date):
	if symbol is None: return [from_date, cur_date]

	# Adjust start date considering IPO date of the symbol chosen
	ipo_date = load_stake().loc[symbol]['ipo']

	if from_date is None:
		from_date = ipo_date
	elif from_date < ipo_date:
		from_date = ipo_date

	# If pivot date is not selected yet, automatically sets it as the 2/3 point of [start-date, end-date] range.
	if cur_date is None:
		from_date = get_timestamp(from_date)
		days = (datetime.now() - from_date).days

		cur_date = (from_date + timedelta(days = days * 2 // 3)).strftime(YMD_FORMAT)
		from_date = from_date.strftime(YMD_FORMAT)

	return [from_date, cur_date]

def find_intersection_point(y1_1, y1_2, y2_1, y2_2):
    # Calculate slopes and y-intercepts for both lines
    x1 = 50
    x2 = 100
    
    m1 = (y1_2 - y1_1) / (x2 - x1)
    b1 = y1_1 - m1 * x1

    m2 = (y2_2 - y2_1) / (x2 - x1)
    b2 = y2_1 - m2 * x1

    # Solve for the intersection point
    x_intersection = (b2 - b1) / (m1 - m2)
    y_intersection = m1 * x_intersection + b1

    return y_intersection

def get_d_for_exit_long(data_series):
    df = pd.DataFrame(data_series, columns=[data_series.name if data_series.name is not None else 'value'])
    df.columns = ['indicator']
    values = df['indicator']
    # Find indices of local maxima
    peaks, _ = find_peaks(values)

    # Extract the corresponding values from 'indicator'
    maximas_list = values.iloc[peaks].tolist()
    maximas_list = [value for value in maximas_list if value >= 70]
    maximas_list = sorted(maximas_list)
    #print("this is maximas ",maximas_list)

    # Calculate the median of the maximas
    median_maximas = np.median(maximas_list)

    return median_maximas

def get_d_for_exit_short(data_series):
    df = pd.DataFrame(data_series, columns=[data_series.name if data_series.name is not None else 'value'])
    df.columns = ['indicator']
    values = df['indicator']
    
    # Find indices of local minima
    valleys, _ = find_peaks(-values)

    # Extract the corresponding values from 'indicator'
    minimas_list = values.iloc[valleys].tolist()
    minimas_list = [value for value in minimas_list if value <= 50]
    minimas_list = sorted(minimas_list)
    #print("minimas list is ", minimas_list)
    # Calculate the median of the minimas
    median_minimas = np.median(minimas_list)

    return median_minimas

def get_buy_crossover_dates(series1, series2):
    df = pd.DataFrame({'Indicator1': series1, 'Indicator2': series2})
    df['Date'] = df.index

    # List to store all intersection dates and intersection_levels
    intersection_dates = []
    intersection_levels = []

    # Check for intersection points
    for i in range(1, len(df)):
        y1_1, y1_2 = df['Indicator1'].iloc[i-1], df['Indicator1'].iloc[i]
        y2_1, y2_2 = df['Indicator2'].iloc[i-1], df['Indicator2'].iloc[i]

        # Check if the two lines intersect
        if (y1_1 <= y2_1 and y1_2 >= y2_2) and y1_1 != y2_1 and y1_2 != y1_1: # K crosses above D and Intersection below 20
            # Interpolate to find the exact date of intersection
            date1, date2 = df['Date'].iloc[i-1], df['Date'].iloc[i]
            #fraction = (y2_1 - y1_1) / (y1_2 - y1_1)
            intersection_date = date2 #+ pd.to_timedelta(fraction * (date2 - date1))
            intersection_level = find_intersection_point(y1_1, y1_2, y2_1, y2_2)
            if(intersection_level<10):
                #print("buy on ", intersection_date, " , indicators at ", intersection_level)
                intersection_dates.append(intersection_date)
                intersection_levels.append(intersection_level)

    return intersection_dates, intersection_levels

def get_sell_crossover_dates(series1, series2):
    df = pd.DataFrame({'Indicator1': series1, 'Indicator2': series2})
    df['Date'] = df.index

    # List to store all intersection dates and intersection_levels
    intersection_dates = []
    intersection_levels = []

    # Check for intersection points
    for i in range(1, len(df)):
        y1_1, y1_2 = df['Indicator1'].iloc[i-1], df['Indicator1'].iloc[i]
        y2_1, y2_2 = df['Indicator2'].iloc[i-1], df['Indicator2'].iloc[i]

        # Check if the two lines intersect
        if (y1_1 >= y2_1 and y1_2 <= y2_2) and y1_1 != y2_1 and y1_2 != y1_1: # K crosses above D and Intersection below 20
            # Interpolate to find the exact date of intersection
            date1, date2 = df['Date'].iloc[i-1], df['Date'].iloc[i]
            intersection_date = date2 #+ pd.to_timedelta(fraction * (date2 - date1))
            intersection_level = find_intersection_point(y1_1, y1_2, y2_1, y2_2)
            if(intersection_level>90):
                #print("intersection level for short = ", intersection_level)
                #print("buy on ", intersection_date, " , indicators at ", intersection_level)
                intersection_dates.append(intersection_date)
                intersection_levels.append(intersection_level)

    return intersection_dates, intersection_levels

def backtest_stock_divergence(symbol, from_date, to_date, out1, out2, bull_entry, bear_entry, bull_exit, bear_exit):
    
    df, _, _ = getPointsGivenR(symbol, 1.02, startDate = from_date, endDate = to_date)
    D = TA.STOCHD(df)
    
    out = sorted(out1 + out2, key = lambda x : (x[0], x[1]))
        
    columns = ['Position', 'Diver-Dur', 'EntryDate', 'EntryPrice', 'ExitDate', 'ExitPrice', 'Return', 'Cum-Profit']
    records, full_list, bull_cumprof, bear_cumprof, bull_matches, bear_matches, stay_still = [], [], 0, 0, 0, 0, 0
    
    d_val_cutoff = 40
    date_diff_cutoff = 15

    signal_end = {}
    signal_end[0] = []
    signal_end[1] = []
    for dStart, dEnd, dVStart, dVEnd, sv, se, dd in out:
        
        d_val_diff = abs(dVStart-dVEnd)
        date_diff = dEnd-dStart
        date_diff = date_diff.days

        if(d_val_diff > d_val_cutoff and date_diff < date_diff_cutoff):
            continue
        
        is_bullish = dVEnd > dVStart
        signal_end[is_bullish].append(dEnd)

    bull_no, bear_no = 0, 0
    for dStart, dEnd, dVStart, dVEnd, sv, se, dd in out:
        prev_dd = 1
        d_val_diff = abs(dVStart-dVEnd)
        date_diff = dEnd-dStart
        date_diff = date_diff.days

        if(d_val_diff > d_val_cutoff and date_diff < date_diff_cutoff):
            continue
        
        is_bullish = dVEnd > dVStart
        sg = 1 if is_bullish else -1

        sig_idx = 0
        templst = signal_end[is_bullish]
        for i in range(len(templst)):
            if dEnd == templst[i]:
                sig_idx = i+1
                break;

        
        idx = df.index.get_loc(dEnd.strftime("%Y-%m-%d"))
        if sg == 1:
            bull_no += 1
            pv = df.iloc[idx+1].low
        else:
            bear_no += 1
            pv = df.iloc[idx+1].high
        count = 0
        for i in range(max(1, idx + 2), len(df)):
            try:
                if sg == 1:
                    
                    if np.sign(df.iloc[i].low - pv) == sg:
                        count += 1
                        if count == bull_entry:
                            dd = df.index[i].strftime("%Y-%m-%d")
                            prev_dd = 0
                            #if i == idx + 1: edd = None
                            break
                    else:
                        count = 0
                    
                    if df.index[i] == templst[sig_idx]:
                        dd = None
                        bull_no -= 1
                        break
                    
                    pv = df.iloc[i].low
                else:
                    
                    if np.sign(df.iloc[i].high - pv) == sg:
                        count += 1
                        if count == bear_entry:
                            dd = df.index[i].strftime("%Y-%m-%d")
                            prev_dd = 0
                            #if i == idx + 1: edd = None
                            break
                    else:
                        count = 0
                    
                    if df.index[i] == templst[sig_idx]:
                        dd = None
                        bear_no -= 1
                        break
                    pv = df.iloc[i].high
            except:
                pass

        if prev_dd == 1:
            dd = None

        if dd is None:
            edd = None
        else:
            r, bull_cumprof, bear_cumprof, succ, edd = get_trans_record(dStart, dEnd, dd, df, bull_cumprof, bear_cumprof, dVEnd > dVStart, bull_exit, bear_exit)
            records.append(r)        

        full_list.append((dStart, dEnd, dVStart, dVEnd, sv, se, dd, edd))
        
        if dd is None: pass
        elif succ or (edd is None): 
            if sg == 1:
                bull_matches += 1
            else:
                bear_matches += 1
    
    bull_acc = bull_matches / bull_no if len(signal_end[1]) > 0 else 0
    bear_acc = bear_matches / bear_no if len(signal_end[0]) > 0 else 0
    acc = (bear_matches + bull_matches) / len(records) if len(records) > 0 else 0
    
    records.append(('', '', '', '', '', '', '', ''))
    records.append((
        '',
        f"From {change_date_format(from_date, YMD_FORMAT, DBY_FORMAT)} To {change_date_format(to_date, YMD_FORMAT, DBY_FORMAT)}",
        f"Symbol: {symbol}",
        '',
        'Success Rate:',
        '{:.1f}%'.format(100 * acc),
        '', ''
    ))
    records.append((
        '', '', '', '',
        'Cumulative Profit:',
        '{:.1f}%'.format(100 * (bull_cumprof+bear_cumprof)),
        '', ''
    ))
    records.append((
        '', '', '', '',
        'Bullish Success Rate:',
        '{:.1f}%'.format(100 * (bull_acc)),
        '', ''
    ))
    records.append((
        '', '', '', '',
        'Bullish Cumulative Profit:',
        '{:.1f}%'.format(100 * bull_cumprof),
        '', ''
    ))
    records.append((
        '', '', '', '',
        'Bearish Success Rate:',
        '{:.1f}%'.format(100 * (bear_acc)),
        '', ''
    ))
    records.append((
        '', '', '', '',
        'Bearish Cumulative Profit:',
        '{:.1f}%'.format(100 * bear_cumprof),
        '', ''
    ))
    records.append(('', '', '', '', '', '', '', ''))
    
    res_df = pd.DataFrame(records, columns = columns)
    return res_df, 100 * acc, 100 * (bull_cumprof+bear_cumprof), full_list



# def get_trans_record(dStart, dEnd, dd, df, D, cumprof, is_bullish):
#     dd = datetime.strptime(dd, YMD_FORMAT)
#     di = list(D.index)
#     idx = 0
    
#     while idx < len(di):
#         if di[idx] >= dd: break
#         idx += 1
    
#     idx = min(idx, len(di) - 1)
    
#     pv = D.iloc[idx]
#     edd = None
#     sg = 1 if is_bullish else -1
    
#     for i in range(max(1, idx + 1), len(D)):
#         if np.sign(D.iloc[i] - pv) != sg:
#             edd = di[i]
#             #if i == idx + 1: edd = None
#             break
#         else:
#             pv = D.iloc[i]

#     #if edd is None: return None, cumprof, None, None
#     ret = (sg * (df.loc[edd].close - df.iloc[idx].close) / df.iloc[idx].close) if edd is not None else 0
#     cumprof += ret
    
#     return (
#         'Long' if is_bullish else 'Short',
#         '{} - {}'.format(dStart.strftime(DBY_FORMAT), dEnd.strftime(DBY_FORMAT)),
#         di[idx].strftime(DBY_FORMAT),
#         df.iloc[idx].close,
#         edd.strftime(DBY_FORMAT) if edd is not None else 'Stay Still',
#         df.loc[edd].close if edd is not None else '',
#         ('{:.1f}%'.format(ret * 100)) if edd is not None else '',
#         '{:.1f}%'.format(cumprof * 100)
#     ), cumprof, ret > 0, edd

# def backtest_helper_function(df, out1, out2, bull_entry, bear_entry, bull_exit, bear_exit):
   
#     out = sorted(out1 + out2, key = lambda x : (x[0], x[1]))
        
#     records, bull_cumprof, bear_cumprof, matches = [], 0, 0, 0
    
#     d_val_cutoff = 40
#     date_diff_cutoff = 15

#     signal_end = {}
#     signal_end[0] = []
#     signal_end[1] = []
#     for dStart, dEnd, dVStart, dVEnd, sv, se, dd in out:
        
#         d_val_diff = abs(dVStart-dVEnd)
#         date_diff = dEnd-dStart
#         date_diff = date_diff.days

#         if(d_val_diff > d_val_cutoff and date_diff < date_diff_cutoff):
#             continue
        
#         is_bullish = dVEnd > dVStart
#         signal_end[is_bullish].append(dEnd)

#     for dStart, dEnd, dVStart, dVEnd, sv, se, dd in out:

#         d_val_diff = abs(dVStart-dVEnd)
#         date_diff = dEnd-dStart
#         date_diff = date_diff.days

#         if(d_val_diff > d_val_cutoff and date_diff < date_diff_cutoff):
#             continue

#         is_bullish = dVEnd > dVStart
#         sg = 1 if is_bullish else -1

#         sig_idx = 0
#         templst = signal_end[is_bullish]
#         for i in range(len(templst)):
#             if dEnd == templst[i]:
#                 sig_idx = i+1
#                 break;

#         idx = df.index.get_loc(dEnd.strftime("%Y-%m-%d"))
#         if sg == 1:
#             pv = df.iloc[idx+1].low
#         else:
#             pv = df.iloc[idx+1].high
#         count = 0
#         for i in range(max(1, idx + 2), len(df)):
#             try:
#                 if sg == 1:
                    
#                     if np.sign(df.iloc[i].low - pv) == sg:
#                         count += 1
#                         if count == bull_entry:
#                             dd = df.index[i].strftime("%Y-%m-%d")
#                             #if i == idx + 1: edd = None
#                             break
#                     else:
#                         count = 0
                    
#                     if df.index[i] == templst[sig_idx]:
#                         dd = None
#                         break
#                     pv = df.iloc[i].low
#                 else:
#                     if np.sign(df.iloc[i].high - pv) == sg:
#                         count += 1
#                         if count == bear_entry:
#                             dd = df.index[i].strftime("%Y-%m-%d")
#                             #if i == idx + 1: edd = None
#                             break
#                     else:
#                         count = 0
                    
#                     if df.index[i] == templst[sig_idx]:
#                         dd = None
#                         break
#                     pv = df.iloc[i].high
#             except:
#                 pass


#         if dd is None:
#             edd = None
#         else:
#             r, bull_cumprof, bear_cumprof, succ, edd = get_trans_record(dStart, dEnd, dd, df, bull_cumprof, bear_cumprof, dVEnd > dVStart, bull_exit, bear_exit)
#             records.append(r)        
        
#         if dd is None: pass
#         elif succ or (edd is None): 
#             matches += 1
    
#     acc = matches / len(records) if len(records) > 0 else 0    
#     return 100 * acc, 100 * (bull_cumprof + bear_cumprof)

# def get_best_tolerance(symbol, one_year_ago_str, input_date_str):

#     df, _, _ = getPointsGivenR(symbol, 1.02, startDate = one_year_ago_str, endDate = input_date_str)

#     one_year_ago_str = get_nearest_forward_date(df, get_timestamp(one_year_ago_str)).strftime(YMD_FORMAT)
#     out1, out2 = get_divergence_data(symbol, one_year_ago_str, input_date_str, oldData = df)

#     # Define the iterables for the four loops
#     iterable1 = range(1, 5)
#     iterable2 = range(1, 5)
#     iterable3 = range(1, 5)
#     iterable4 = range(1, 5)

#     val = 0
#     final_bull_entry = 1
#     final_bear_entry = 1
#     final_bull_exit = 3
#     final_bear_exit = 2
#     # Use itertools.product to generate the Cartesian product
#     total_iterations = len(iterable1) * len(iterable2) * len(iterable3) * len(iterable4)

#     # Use tqdm to wrap itertools.product for the progress bar
#     for bull_entry, bear_entry, bull_exit, bear_exit in tqdm(itertools.product(iterable1, iterable2, iterable3, iterable4), total=total_iterations):
#         acc, cumprof = backtest_helper_function(df, out1, out2, bull_entry, bear_entry, bull_exit, bear_exit)
#         tempval = 0.8*acc + 0.2*cumprof
#         if tempval > val:
#             val = tempval
#             final_bull_entry = bull_entry
#             final_bear_entry = bear_entry
#             final_bull_exit = bull_exit
#             final_bear_exit = bear_exit

#     return final_bull_entry, final_bear_entry, final_bull_exit, final_bear_exit

def get_trans_record(dStart, dEnd, dd, df, bull_cumprof, bear_cumprof, is_bullish, bull_exit, bear_exit):
    dd = datetime.strptime(dd, YMD_FORMAT)
    di = list(df.index)
    idx = 0
    
    while idx < len(di):
        if di[idx] >= dd: break
        idx += 1


    idx = min(idx, len(di) - 1)
    
    edd = None
    sg = 1 if is_bullish else -1

    count = 0
    if sg == 1:
        pv = df.iloc[idx].high
    else:
        pv = df.iloc[idx].low
    entry_price = df.loc[dd].close
    for i in range(max(1, idx + 1), len(di)):
        try:
        
            if sg == 1:
                if np.sign(df.iloc[i].high - pv) != sg:
                    count += 1
                    if count == bull_exit:
                        if np.sign(df.iloc[i].close-df.iloc[idx].close) == sg:
                            edd = df.index[i]
                            #if i == idx + 1: edd = None
                            break
                        else: count = 0

                else:
                    count = 0
                pv = df.iloc[i].high
            else:
                if np.sign(df.iloc[i].low - pv) != sg:
                    count += 1
                    if count == bear_exit:
                        if np.sign(df.iloc[i].close-df.iloc[idx].close) == sg:
                            edd = df.index[i]
                            #if i == idx + 1: edd = None
                            break
                        else: count = 0

                else:
                    count = 0
                pv = df.iloc[i].low
            curr_price = df.iloc[i].close
            # try:
            #     atr = ta.atr(high = df['high'], low = df['low'], close = df['close'], length = 14);
            #     stop_percentage = 2 * atr.iloc[-1] / df['close'].iloc[-1]
            # except:
            #     atr = ta.atr(high = df['high'], low = df['low'], close = df['close'], length = len(df))
            #     stop_percentage = 2 * atr.iloc[-1] / df['close'].iloc[-1]

            if sg == 1:
                gain_perc = ((curr_price-entry_price)*100)/entry_price
                # if gain_perc <= -5 or gain_perc >= 15:
                # if gain_perc <= -(stop_percentage*100):
                if gain_perc <= -5:
                    edd = df.index[i]
                    break
            else:
                gain_perc = (-(curr_price-entry_price)*100)/entry_price
                # if gain_perc <= -2.5 or gain_perc >= 15:
                # if gain_perc <= -stop_percentage*(100/2):
                if gain_perc <= -2.5:
                    edd = df.index[i]
                    break


            
        except:
            print("Hey! error occur")

    #if edd is None: return None, cumprof, None, None
    if sg == 1:
        ret = (sg * (df.loc[edd].close - df.iloc[idx].close) / df.iloc[idx].close) if edd is not None else 0
        bull_cumprof += ret
    else:
        ret = (sg * (df.loc[edd].close - df.iloc[idx].close) / df.iloc[idx].close) if edd is not None else 0
        bear_cumprof += ret
    
    return (
        'Long' if is_bullish else 'Short',
        '{} - {}'.format(dStart.strftime(DBY_FORMAT), dEnd.strftime(DBY_FORMAT)),
        di[idx].strftime(DBY_FORMAT),
        df.iloc[idx].close,
        edd.strftime(DBY_FORMAT) if edd is not None else 'Stay Still',
        df.loc[edd].close if edd is not None else '',
        ('{:.1f}%'.format(ret * 100)) if edd is not None else '',
        '{:.1f}%'.format((bull_cumprof + bear_cumprof) * 100)
    ), bull_cumprof, bear_cumprof , ret > 0, edd
