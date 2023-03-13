#!/usr/bin/env python
# coding: utf-8

# # تمرین 1: بدست آوردن روند در یک سری زمانی
# 

# In[30]:


import pandas_datareader as pdr 
import matplotlib.pyplot as plt
import statsmodels.api as sm
df=pdr.DataReader("MSFT","yahoo",start="2021-04-04")
cycle , trend=sm.tsa.filters.hpfilter(df["Adj Close"])


# In[31]:


df[["Adj Close","Open"]]


# In[32]:


df["cycle"]=cycle
df["trend"]=trend

df["trend"].plot()
df["cycle"].plot()
df["Adj Close"].plot()
df["Open"].plot()

plt.legend(["trend","cycle", "Adj Close"])
plt.show()


# # exercise2: Statmodels

# In[33]:


import numpy as np 
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
df=pdr.DataReader("MSFT","yahoo",start="2021-04-04")
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
series = [i**2.0 for i in range(1,50)]
result = seasonal_decompose(series, model='multiplicative', period=1)
result.plot()
pyplot.show()


# # تمرین 3: به عنوان مثال، AR(1) یک مدل خودرگرسیون مرتبه اول است.

# In[34]:


import tsemodule5 as tm5
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import statsmodels as sm
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
#url=r"C:\Users\laptop\Downloads\G.Barekat.Pharm.csv"
#url=r"C:\Users\laptop\Downloads\Shazand.Petr..csv"
#url=r"C:\Users\laptop\Downloads\NORI.Petrochemical.csv"
#url=r"C:\Users\laptop\Downloads\Fanavaran.Petr..csv"
#url=r"C:\Users\laptop\Downloads\S_Metals.&.Min..csv"
url="C:/Users/Ateeq/Downloads/S_Isf..Oil.Ref.Co..csv"
#url=r"C:\Users\laptop\Downloads\Melli.Ind..Grp..csv"
df=pd.read_csv(url,index_col="<DTYYYYMMDD>",parse_dates=True)["2020-08-01":"2021-12-01"][::-1]
#print(df)

model = ARIMA(df["<CLOSE>"].asfreq("D"),order=(1, 1, 0),seasonal_order=(0, 1, 0, 40))

result=model.fit()

#print(result.summary())

#df["forcast"]=result.predict(start=-50,end=-1)
#df[["Close","forcast"]].plot()

from pandas.tseries.offsets import DateOffset
future_dates =[df.index[-1]+DateOffset(day=x)  for x in range(1,30)]
future_df=pd.DataFrame(index=future_dates , columns=df.columns)
#print(future_df)

final_df=pd.concat([df,future_df])

#....for backtest
#final_df["forcast"]=result.predict(start=400,end=450)

#....for predic
final_df["forcast"]=result.predict(start=480,end=530)

final_df[["<CLOSE>","forcast"]].plot()

print(final_df["forcast"])
plt.show()

result.summary()


# # Example 4:AR example

# In[35]:


from statsmodels.tsa.ar_model import AutoReg
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = AutoReg(data, lags=1)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)


# # Example 5:

# In[36]:


from statsmodels.tsa.arima.model import ARIMA
from random import random
# contrived datasetدیتاست ساختگی 
data = [x + random() for x in range(1, 100)]
# fit model
model = ARIMA(data, order=(0, 0, 1))
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)


# # Exercise6: Import the required libraries

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## To use statsmodels for linear regression
import statsmodels.formula.api as smf

## To use sklearn for linear regression
from sklearn.linear_model import LinearRegression


# In[38]:


femeli="C:/Users/Ateeq/Downloads/S_I..N..C..Ind..csv"
shapna="C:/Users/Ateeq/Downloads/S_Isf..Oil.Ref.Co..csv"


# In[39]:


fml=pd.read_csv(femeli,index_col="<DTYYYYMMDD>",parse_dates=True)
#fml


# In[40]:


shp=pd.read_csv(shapna,index_col="<DTYYYYMMDD>",parse_dates=True)
#shp


# In[41]:


## Calculate log returns for the period based on  Close prices'
fml['femeli'] = np.log(fml['<CLOSE>'] / fml['<CLOSE>'].shift(1))
shp['shapna'] = np.log(shp['<CLOSE>'] / shp['<CLOSE>'].shift(1))
## Create a dataframe
df = pd.concat([fml['femeli'], shp['shapna']], axis = 1).dropna()
#df


# In[42]:


### Create an instance of the class LinearRegression()
slr_skl_model = LinearRegression()

### Fit the model (sklearn calculates beta_0 and beta_1 here)

X = df['femeli'].values.reshape(-1, 1)
slr_skl_model_shapna = slr_skl_model.fit(X, df['shapna'])

print("The intercept in the sklearn regression result is", \
      np.round(slr_skl_model_shapna.intercept_, 4))
print("The slope in the sklearn regression model is", \
      np.round(slr_skl_model_shapna.coef_[0], 4))


# In[43]:


## Linear regression plot of X (fml) and Y (shp)

plt.figure(figsize = (10, 6))
plt.rcParams.update({'font.size': 14})
plt.xlabel("femeli returns")
plt.ylabel("shapna returns")
plt.title("Simple linear regression model")
plt.scatter(df['femeli'], df['shapna'])
plt.plot(X, slr_skl_model.predict(X), 
         label='Y={:.4f}+{:.4f}X'.format(slr_skl_model_shapna.intercept_, \
                                         slr_skl_model_shapna.coef_[0]), 
             color='red')
plt.legend()
plt.show()


# In[44]:


## Print the parameter estimates of the simple linear regression model

print("\n")
print("====================================================================")
print("The intercept in the sklearn regression result is", \
      np.round(slr_skl_model_shapna.intercept_, 4))
print("The slope in the sklearn regression model is", \
      np.round(slr_skl_model_shapna.coef_[0], 4))
print("====================================================================")
print("\n")


# هیچ اتفاق نظری در مورد اندازه مجموعه داده ما وجود ندارد. بیایید به بررسی آن ادامه دهیم و نگاهی به آمار توصیفی این داده های جدید بیندازیم. این بار، مقایسه آمار را با گرد کردن مقادیر به دو اعشار با متد round() و جابه‌جایی جدول با ویژگی T تسهیل می‌کنیم:

# In[45]:


print(df.describe().round(2).T)


# In[46]:


import seaborn as sns # Convention alias for Seaborn


# In[47]:


variables = ['femeli','shapna']


# In[48]:


for var in variables:
    plt.figure() # Creating a rectangle (figure) for each plot
    # Regression Plot also by default includes
    # best-fitting regression line
    # which can be turned off via `fit_reg=False`
    sns.regplot(x=var, y='shapna', data=df).set(title=f'Regression plot of {var}');


# In[49]:


correlations = df.corr()


# In[50]:


# annot=True displays the correlation values
sns.heatmap(correlations, annot=True).set(title='Heatmap of stock Data - Pearson Correlations');
#has a strong positive linear relationship


# # example: curve fit .....>  az ein رگرسیون غیر خطی

# In[51]:


from scipy.optimize import curve_fit
from numpy import arange
import matplotlib.pyplot as plt


# In[52]:


# define the true objective function
def objective(x, a, b, c):
    return a * x + b * x**2 + c


# In[53]:


data = df.values
x, y = data[:, 1], data[:, -1]


# In[54]:


# curve fit
popt, _ = curve_fit(objective, x, y)
# summarize the parameter values
a, b, c = popt
print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
# plot input vs output
plt.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x), 1)
# calculate the output for the range
y_line = objective(x_line, a, b, c)
# create a line plot for the mapping function
plt.plot(x_line, y_line, '--', color='red')
plt.show()


# # SHARP & Sortino

# In[55]:


import numpy as np 
import pandas as pd
import tsemodule5 as tm5


# In[59]:


df=pd.DataFrame()
Shares=["فملي","خودرو"]
for Share in Shares:
    df[Share]=tm5.stock(Share,standard=True)["Close"]
    
df[::-1].dropna()


# In[60]:


df.plot()


# In[61]:


df.pct_change().sum()


# In[62]:


khodro=tm5.stock("خودرو")[::-1]


# In[63]:


khodro['lastday']=khodro["<CLOSE>"].shift(1)
khodro


# In[64]:


khodro["Pnl"]=khodro["<CLOSE>"]-khodro["lastday"]
khodro


# In[65]:


khodro["Pnl"].plot()


# In[66]:


khodro["Pnl"].sum()


# #
# Stop-loss
# 
# اولین محدودیت ریسک که به آن نگاه خواهیم کرد کاملاً شهودی است و توقف ضرر یا حداکثر ضرر نامیده می شود. این محدودیت حداکثر مقدار پولی است که یک استراتژی مجاز است از دست بدهد، یعنی حداقل PnL مجاز. این اغلب مفهومی از یک چارچوب زمانی برای آن ضرر دارد، به این معنی که توقف ضرر می تواند برای یک روز، برای یک هفته، یک ماه یا برای کل طول عمر استراتژی باشد. توقف ضرر با بازه زمانی یک روز به این معنی است که اگر استراتژی یک مقدار استاپ ضرر را در یک روز از دست داد، دیگر مجاز به معامله در آن روز نیست، اما می تواند روز بعد از سر گرفته شود. به همین ترتیب، برای مبلغ توقف ضرر در یک هفته، دیگر مجاز به معامله برای آن هفته نیست، اما می تواند هفته آینده از سر گرفته شود.
# 

# In[67]:


num_days = len(khodro.index)
pnl = khodro["Pnl"]
weekly_losses = []
monthly_losses = []


# In[68]:


for i in range(0, num_days):
    if i >= 5 and pnl[i - 5] > pnl[i]:
        weekly_losses.append(pnl[i] - pnl[i - 5])
    if i >= 20 and pnl[i - 20] > pnl[i]:
        monthly_losses.append(pnl[i] - pnl[i - 20])
        
plt.hist(weekly_losses, 50)
plt.gca().set(title='Weekly Loss Distribution', xlabel='Rial',ylabel='Frequency')
plt.show()


# In[69]:


max_pnl = 0
max_drawdown = 0
drawdown_max_pnl = 0
drawdown_min_pnl = 0
for i in range(0, num_days):
    max_pnl = max(max_pnl, pnl[i])
    drawdown = max_pnl - pnl[i]
if drawdown > max_drawdown:
    max_drawdown = drawdown
    drawdown_max_pnl = max_pnl
    drawdown_min_pnl = pnl[i]
print('Max Drawdown:', max_drawdown)
khodro["Pnl"].plot(x='Date', legend=True)
plt.axhline(y=drawdown_max_pnl, color='g')
plt.axhline(y=drawdown_min_pnl, color='r')
plt.show()


# سبت شارپ یک معیار عملکرد و ریسک بسیار رایج است که در صنعت برای اندازه‌گیری و مقایسه عملکرد استراتژی‌های معاملاتی الگوریتمی استفاده می‌شود. نسبت شارپ به عنوان نسبت میانگین PnL در یک دوره زمانی و انحراف استاندارد PnL در همان دوره تعریف می شود. مزیت نسبت شارپ این است که سودآوری یک استراتژی معاملاتی را در بر می گیرد و در عین حال ریسک را با استفاده از نوسانات بازده محاسبه می کند. بیایید به نمایش ریاضی نگاهی بیندازیم: نسبت های شارپ و سورتینو را برای استراتژی معاملاتی خود محاسبه کنیم. ما از یک هفته به عنوان افق زمانی برای استراتژی معاملاتی خود استفاده خواهیم کرد:

# In[70]:


last_week = 0
weekly_pnls = []
weekly_losses = []


# In[71]:


for i in range(0, num_days):
    if i - last_week >= 5:
        pnl_change = pnl[i] - pnl[last_week]
        weekly_pnls.append(pnl_change)
        if pnl_change < 0:
            weekly_losses.append(pnl_change)
        last_week = i


# In[72]:


sharpe_ratio_week=pd.Series(weekly_pnls).mean()/pd.Series(weekly_pnls).std()
sortino_ratio_week=pd.Series(weekly_losses).mean()/pd.Series(weekly_losses).std()


# In[73]:


print(sharpe_ratio_week,sortino_ratio_week)


# In[ ]:





# In[ ]:





# # example: REGRESSION BETWEEN aPPLE , mSFT , GOOGLE

# In[74]:


### Import the required libraries

import numpy as np
import pandas as pd

import yfinance as yf
import datetime
import matplotlib.pyplot as plt

## To use statsmodels for linear regression
import statsmodels.formula.api as smf

## To use sklearn for linear regression
from sklearn.linear_model import LinearRegression


# In[75]:


APPLE= yf.download("AAPL", start = "2021-01-01", end ="2022-11-29" , progress = False).tail(100)
#APPLE


# In[76]:


MIC=yf.download("MSFT", start ="2021-01-01", end = "2022-11-29", progress = False).tail(100)


# In[77]:


GOOGLE=yf.download("GOOG", start ="2021-01-01", end = "2022-11-29", progress = False).tail(100)


# In[78]:


## Calculate log returns for the period based on Adj Close prices
APPLE_df= np.log(APPLE['Adj Close'] / APPLE['Adj Close'].shift(1))
MIC_df=np.log(MIC['Adj Close'] /MIC['Adj Close'].shift(1))
GOOGLE_df=np.log(GOOGLE['Adj Close'] / GOOGLE['Adj Close'].shift(1))


# In[79]:


df = pd.concat([APPLE_df,MIC_df], axis = 1).dropna()


# In[80]:


plt.figure(figsize = (10, 6))
plt.rcParams.update({'font.size': 14})
plt.xlabel("APPLe returns")
plt.ylabel("microsoft returns")
plt.title("Scatter plot of daily returns")
plt.scatter(APPLE_df, MIC_df)
plt.show()


# In[81]:


df.corr()


# # Multiple Regression Predictions
# 

# # example of training a final regression model

# In[82]:


from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
# generate regression dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
# fit final model
model = LinearRegression()
model.fit(X, y)
# new instances where we do not know the answer
Xnew, _ = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1)
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))


# In[ ]:





# In[88]:


df=pd.read_csv("C:/Users/Ateeq/Downloads/S_Isf..Oil.Ref.Co..csv",index_col="<DTYYYYMMDD>", parse_dates=True)


# In[89]:


# Plot
plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)

# Decorations
plt.title('Correlogram of mtcars', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[91]:


# Plot
plt.figure(figsize=(10,8), dpi= 80)
sns.pairplot(df, kind="scatter", hue="<CLOSE>", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
plt.show()


# # exercise :Linear Regression in One Variable

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


data = pd.read_csv('C:/Users/Ateeq/Downloads/S Isf..Oil.Ref.Co..csv', index_col="<DTYYYYMMDD>",parse_dates=True)[::-1]


# In[9]:


data.tail(10)


# In[10]:


X = data['<CLOSE>'].values              #Assign 'close price' to X
y = data['<HIGH>'].values                  #Assign 'high price' to y
m = len(y)                   #This is the length of the training set
plt.scatter(X,y, c='red', marker='x')         #Plot scatter plot
plt.ylabel('close price')         #Label on the y axis
plt.xlabel('high price')     #Label on the X axis
plt.title('Scatter Plot of Training Data')      #Title for the plot

#The plot is shown below:


# # Exercise: Multiple Linear Regression

# In[23]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("C:/Users/Ateeq/Downloads/S Isf..Oil.Ref.Co..csv",index_col="<DTYYYYMMDD>",parse_dates=True)[::-1]
print(data.info())
print(data.tail())
print(data.describe())


# In[27]:


close=data["<CLOSE>"]
x = data.iloc[:,[1,2]].values
y = close.values.reshape(-1,1)

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0: ",multiple_linear_regression.intercept_)
print("b1: ", multiple_linear_regression.coef_)

#predict
x_ = np.array([[10,35],[5,35]])
multiple_linear_regression.predict(x_)

y_head = multiple_linear_regression.predict(x) 
from sklearn.metrics import r2_score
print("r_square score: ", r2_score(y,y_head))


# In[28]:


x


# # exercise Polynomial Linear Regression

# In[29]:


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/Ateeq/Downloads/S Isf..Oil.Ref.Co..csv")
print(data.info())
print(data.head())
#print(data.describe()


# In[31]:


close=data["<CLOSE>"]
high=data["<HIGH>"]
x = close.values.reshape(-1,1)
y = high.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("CLOSE")
plt.ylabel("HIGH")
plt.show()


# In[33]:


# polynomial regression =  y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

polynominal_regression = PolynomialFeatures(degree=4)
x_polynomial = polynominal_regression.fit_transform(x,y)

# %% fit
linear_regression = LinearRegression()
linear_regression.fit(x_polynomial,y)
# %%
y_head2 = linear_regression.predict(x_polynomial)

plt.plot(x,y_head2,color= "green",label = "poly")
plt.legend()
plt.scatter(x,y)
plt.xlabel("CLOSE")
plt.ylabel("High")
plt.show()

from sklearn.metrics import r2_score
print("r_square score: ", r2_score(y,y_head2))


# # European Call Option Inner Value Plot

# In[2]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'serif'
# Option Strike
K = 8000
# Graphical Output
S = np.linspace(7000, 9000, 100) # index level values
h = np.maximum(S - K, 0) # inner values of call option
plt.figure()
plt.plot(S, h, lw=2.5) # plot inner values at maturity
plt.xlabel('index level $S_t$ at maturity')
plt.ylabel('inner value of European call option')
plt.grid(True)


# In[25]:


import pandas_datareader as pdr
import datetime


# In[26]:


tickers = ['msft', 'aapl', 'twtr', 'goog', 'amzn']
df1 = pdr.DataReader(tickers, data_source='yahoo', start='2020-01-01', end='2021-09-28')["Adj Close"]


# In[27]:


df1.head()


# In[28]:


plt.figure(figsize=(15, 6))
for i in range(df1.shape[1]):
    plt.plot(df1.iloc[:,i], label=df1.columns.values[i])
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('Price in $')
plt.show()


# In[29]:


df3 = df1.divide(df1.iloc[0] / 100)

plt.figure(figsize=(15, 6))
for i in range(df3.shape[1]):
    plt.plot(df3.iloc[:,i], label=df3.columns.values[i])
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('Normalized prices')
plt.show()


# محاسبه مرز کارآمد
# 
# 2 ورودی وجود دارد که باید قبل از یافتن مرز کارآمد برای سهام خود محاسبه کنیم: نرخ بازده سالانه و ماتریس کوواریانس.
# 
# نرخ بازده سالانه با ضرب درصد تغییر روزانه همه سهام در تعداد روزهای کاری هر سال (252) محاسبه می‌شود.

# In[32]:


#Calculate daily changes in the stocks' value
df2 = df1.pct_change()
#Remove nan values at the first row of df2. Create a new dataframe df
df=df2.iloc[1:len(df2.index),:]
# Calculate annualized average return for each stock. Annualized average return = Daily average return * 252 business days.
r = np.mean(df,axis=0)*252

# Create a covariance matrix
covar = df.cov()


# در مرحله بعد، باید چند توابع را تعریف کنیم که بعداً در محاسبات از آنها استفاده خواهیم کرد.
# 
# نرخ بازده، نرخ بازده سالانه کل پرتفوی است.
# 
# نوسان سطح ریسک است که به عنوان تقسیم استاندارد بازده تعریف می شود.
# 
# نسبت شارپ کارایی ریسک است. بازده سرمایه گذاری را در مقایسه با ریسک آن ارزیابی می کند.

# In[33]:


#Define frequently used functions.
# r is each stock's return, w is the portion of each stock in our portfolio, and covar is the covariance matrix
# Rate of return
def ret(r,w):
    return r.dot(w)
# Risk level - or volatility
def vol(w,covar):
    return np.sqrt(np.dot(w,np.dot(w,covar)))
def sharpe (ret,vol):
    return ret/vol


# مشکل اکنون این است که چگونه نمونه کارها را بهینه کنیم. فرض کنید کمترین سطح ریسک ممکن را می خواهیم. بنابراین، ما باید پرتفویی با حداقل نوسان پیدا کنیم. این یک مشکل کمینه سازی ساده است. با این حال، مرزهای خاصی وجود دارد که باید آنها را رعایت کنیم:
# 
# همه اوزان باید بین 0 و 1 باشد (زیرا نمی توانیم مقدار منفی سهام بخریم و نمی توانیم با بیش از 100٪ از 1 سهام، پرتفوی تشکیل دهیم).
# مجموع وزن کل سهام باید 1 باشد.
# اکنون، باید الگوریتمی را برای بهینه سازی نمونه کارها انتخاب کنیم. روش گرادیان کاهش یافته تعمیم یافته یک انتخاب عملی است، اما هیچ کتابخانه ای وجود ندارد که آن را داشته باشد. اگر می خواهید، می توانید در مورد این الگوریتم اینجا بخوانید، یا این کدی که در Github پیدا کردم را امتحان کنید (من این کار را نکرده ام). در غیر این صورت، می‌توانیم برخی از گزینه‌های موجود در scipy.optimize را بررسی کنیم. چیزی که من انتخاب کردم الگوریتم محدود منطقه اعتماد (method='trust-constr') است زیرا برای توابع اسکالر چند متغیره مناسب است.
# 
# من برای پیاده سازی این کد در کد توضیح خواهم داد.

# In[34]:


# All weights, of course, must be between 0 and 1. Thus we set 0 and 1 as the boundaries.
from scipy.optimize import Bounds
bounds = Bounds(0, 1)

# The second boundary is the sum of weights.
from scipy.optimize import LinearConstraint
linear_constraint = LinearConstraint(np.ones((df2.shape[1],), dtype=int),1,1)

# Find a portfolio with the minimum risk.
from scipy.optimize import minimize
#Create x0, the first guess at the values of each stock's weight.
weights = np.ones(df2.shape[1])
x0 = weights/np.sum(weights)
#Define a function to calculate volatility
fun1 = lambda w: np.sqrt(np.dot(w,np.dot(w,covar)))
res = minimize(fun1,x0,method='trust-constr',constraints = linear_constraint,bounds = bounds)

#These are the weights of the stocks in the portfolio with the lowest level of risk possible.
w_min = res.x

np.set_printoptions(suppress = True, precision=2)
print(w_min)
print('return: % .2f'% (ret(r,w_min)*100), 'risk: % .3f'% vol(w_min,covar))


# اگر بخواهیم پرتفویی با بالاترین سطح کارایی ریسک پیدا کنیم - یعنی پرتفویی که بالاترین نسبت بازده/ریسک (نسبت شارپ) را داشته باشد؟
# 
# ما فقط می‌توانیم از همان الگوریتم، با محدودیت‌های یکسان استفاده کنیم، اما این بار، بیایید بالاترین نسبت شارپ را بهینه کنیم، که یک مشکل حداکثرسازی است.
# 
# اما همانطور که یکی از مدرس های مورد علاقه من گفت
# 
# حداکثر سازی برای بازنده هاست! میخوایم کم کنیم!!
# 
# بنابراین، به جای آن، حداقل 1/Sharpe_ratio را خواهیم یافت.

# In[35]:


#Define 1/Sharpe_ratio
fun2 = lambda w: np.sqrt(np.dot(w,np.dot(w,covar)))/r.dot(w)
res_sharpe = minimize(fun2,x0,method='trust-constr',constraints = linear_constraint,bounds = bounds)

#These are the weights of the stocks in the portfolio with the highest Sharpe ratio.
w_sharpe = res_sharpe.x
print(w_sharpe)
print('return: % .2f'% (ret(r,w_sharpe)*100), 'risk: % .3f'% vol(w_sharpe,covar))


# In[36]:


w = w_min
num_ports = 100
gap = (np.amax(r) - ret(r,w_min))/num_ports


all_weights = np.zeros((num_ports, len(df.columns)))
all_weights[0],all_weights[1]=w_min,w_sharpe
ret_arr = np.zeros(num_ports)
ret_arr[0],ret_arr[1]=ret(r,w_min),ret(r,w_sharpe)
vol_arr = np.zeros(num_ports)
vol_arr[0],vol_arr[1]=vol(w_min,covar),vol(w_sharpe,covar)

for i in range(num_ports):
    port_ret = ret(r,w) + i*gap
    double_constraint = LinearConstraint([np.ones(df2.shape[1]),r],[1,port_ret],[1,port_ret])
    
    #Create x0: initial guesses for weights.
    x0 = w_min
    #Define a function for portfolio volatility.
    fun = lambda w: np.sqrt(np.dot(w,np.dot(w,covar)))
    a = minimize(fun,x0,method='trust-constr',constraints = double_constraint,bounds = bounds)
    
    all_weights[i,:]=a.x
    ret_arr[i]=port_ret
    vol_arr[i]=vol(a.x,covar)
    sharpe_arr = ret_arr/vol_arr  

plt.figure(figsize=(20,10))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.show()


# In[ ]:




