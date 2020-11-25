import pandas as pd , numpy as np, requests, time
from statistics import mean
from matplotlib import pyplot as plt

Compare_Stocks = pd.DataFrame(columns=["Company", "Days_Observed", "Crosses", "True_Positive", "False_Positive", "True_Negative", "False_Negative", "Sensitivity", 
"Specificity", "Accuracy", "TPR", "FPR"])

hist_data = pd.read_csv("MARUTI.NS.csv")
# Create the title
title = 'Close Price History '

#Create and plot the graph
plt.figure(figsize=(12.2,4.5)) #width = 12.2in, height = 4.5
plt.plot( hist_data['Close'],  label='Close')#plt.plot( X-Axis , Y-Axis, line_width, alpha_for_blending,  label)
plt.xticks(rotation=45) 
plt.title(title)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Price USD ($)',fontsize=18)
plt.show()
rsi_low = 35
rsi_high = 70
mac_short = 5
mac_long = 10
max_days = 5
rsi_overhead = 15
# This list holds the closing prices of a stock
prices = []
dates = []
c = 0
while c < len(hist_data):
    prices.append(hist_data.iloc[c,5])
    dates.append(hist_data.iloc[c,0])
    # print(dates[c])
    # print("c is ",c)
    c +=1
i = 0
upPrices=[]
downPrices=[]
while i < len(prices):
    if i == 0:
        upPrices.append(0)
        downPrices.append(0)
    else:
        if (prices[i]-prices[i-1])>0:
            upPrices.append(prices[i]-prices[i-1])
            downPrices.append(0)
        else:
            downPrices.append(prices[i]-prices[i-1])
            upPrices.append(0)
    i += 1
x = 0
avg_gain = []
avg_loss = []
shorter_ma = []
longer_ma = []
    #  Loop to calculate the average gain and loss
while x < len(upPrices):
    if x <rsi_overhead:
        avg_gain.append(0)
        avg_loss.append(0)
    else:
        sumGain = 0
        sumLoss = 0
        y = x - 14
        while y<=x:
            sumGain += upPrices[y]
            sumLoss += downPrices[y]
            y += 1
        sumGain = sumGain/14
        sumLoss = sumLoss/14
        avg_gain.append(sumGain)
        avg_loss.append(abs(sumLoss))
    #else :
        # sumGain = (sumGain*13 + upPrices[x])/14
        # sumLoss = (sumLoss*13 + downPrices[x])/14
        # avg_gain.append(sumGain)
        # avg_loss.append(abs(sumLoss))
    if x < mac_short:
        shorter_ma.append(0)
    elif x == mac_short:
        avg = 0
        y = 0
        while y < mac_short:
            avg += prices[y]
            y+=1
        shorter_ma.append(avg/mac_short)
    else:
        avg = avg + prices[x-1] - prices[x-mac_short-1]
        shorter_ma.append(avg/mac_short)
    if x < mac_long:
        longer_ma.append(0)
    elif x == mac_long:
        avgx = 0
        y = 0
        while y < mac_long:
            avgx += prices[y]
            y +=1
        longer_ma.append(avgx/mac_long)
    else:
        avgx = avgx + prices[x-1] - prices[x-mac_long-1]
        longer_ma.append(avgx/mac_long)
    # print("shorter ma is ",shorter_ma[x])
    # print("longer ma is ",longer_ma[x])

    x += 1
p = 0
RS = []
RSI = []
#  Loop to calculate RSI and RS
while p < len(prices):
    if p <rsi_overhead:
        RS.append(0)
        RSI.append(0)
    else:
        try:
            RSvalue = (avg_gain[p]/avg_loss[p])
        except ZeroDivisionError:
            RSalue = 100000000
        RS.append(RSvalue)
        RSI.append(100 - (100/(1+RSvalue)))
    p+=1
#  Creates the csv for each stock's RSI and price movements
df_dict = {
        'Date'  : dates,
        'Prices' : prices,
        'upPrices' : upPrices,
        'downPrices' : downPrices,
        'AvgGain' : avg_gain,
        'AvgLoss' : avg_loss,
        'RS' : RS,
        'RSI' : RSI
    }
df = pd.DataFrame(df_dict, columns = ['Date','Prices', 'upPrices', 'downPrices', 'AvgGain','AvgLoss', 'RS', "RSI"])
df.to_csv("MARUTI.NS_RSI.csv", index = False)
#Plot the chart###################################################
plt.figure(figsize=(12.2,4.5)) #width = 12.2in, height = 4.5
plt.plot(df.index, RSI, label='RSI', color = 'red')
plt.axhline(y= rsi_low, color='green', label = 'Oversold')
plt.axhline(y= rsi_high, color='blue', label = 'Overbought')
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.show()
############################################################
i = max(mac_long,rsi_overhead)
rsi_buy_flag = 0
rsi_sell_flag = 0
profit_movement = []
sell_date = []
days_limit = 0
profit = 0
cost_price = 0
sell_price = 0

stocks  = 0
while(i < len(prices)):
    if(RSI[i] < rsi_low and not stocks ):
        rsi_buy_flag = 1
        limit_days = 0
        print("kchange day rsi was " + str(RSI[i]) + " at day " + str(i))
    elif(RSI[i] > rsi_high and stocks):
        rsi_sell_flag = 1
        limit_days = 0
        print("bchange day rsi was " + str(RSI[i]) + " at day " + str(i))
    if (rsi_buy_flag == 1 and shorter_ma[i] >= longer_ma[i]):
        #buy()
        rsi_buy_flag = 0
        limit_days = 0
        stocks = 1
        cost_price = prices[i]
        print("khareeedaaaa " + str(cost_price) + " on " + str(i) + " day with rsi " + str(RSI[i])  )
    elif(rsi_sell_flag == 1 and longer_ma[i] >= shorter_ma[i]):
        #sell()
        rsi_sell_flag =0
        stocks = 0
        limit_days = 0
        sell_price = prices[i]
        profit+= sell_price - cost_price
        profit_movement.append(profit)
        sell_date.append(hist_data.iloc[i,0])
        print("bikkkaaaa " + str(sell_price) + " on " + str(i) + " day with rsi " + str(RSI[i]) )

    if (RSI[i] > rsi_low and rsi_buy_flag == 1):
        limit_days += 1
        if limit_days == max_days:
            rsi_buy_flag = 0
            limit_days = 0
    if (RSI[i] < rsi_high and rsi_sell_flag == 1):
        limit_days += 1
        if limit_days == max_days:
            rsi_sell_flag = 0
            limit_days = 0
    i+=1
if(stocks):
    i = len(prices) - 1
    sell_price = prices[i]
    print("bikkkaaaa " + str(sell_price) + " on " + str(i) + " day with rsi " + str(RSI[i]) )
    profit+= sell_price - cost_price
    profit_movement.append(profit)
    sell_date.append(hist_data.iloc[i,0])
print("profit is ",profit)
# fig = plt.figure()
# fig.set_size_inches((25, 18))
# ax_rsi = fig.add_axes((0, 0.24, 1, 0.2))
# #ax_rsi.plot(data.index, [70] * len(data.index), label="overbought")
# #ax_rsi.plot(data.index, [30] * len(data.index), label="oversold")
# plt.axhline(y = 70,color = 'r',label = "overbought")
# ax_rsi.plot(data.index, RSI, label="RSI")
# ax_rsi.plot(data["Close"])
# ax_rsi.legend()
# plt.show()
# plt.plot(sell_date,profit_movement,marker='o', markerfacecolor='blue', markersize=12)
# # setting x and y axis range 
# plt.ylim(-500,500) 
# #plt.xlim(2019-09-16,2020-11-15) 
# # naming the x axis 
# plt.xlabel('Date') 
# # naming the y axis 
# plt.ylabel('Profit') 

# plt.show()
  