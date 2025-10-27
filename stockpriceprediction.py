#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sb
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
from datetime import datetime, timedelta


# In[2]:


ticker_symbol= 'HDFCBANK.NS'
today= datetime.today()
one_year_ego= today-timedelta(days=365)
start_date= "2015-12-30"
end_date= "2025-10-22"


# In[3]:


data= yf.download(ticker_symbol , start= start_date, end= end_date)


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.isnull()


# In[7]:


data.info()


# In[8]:


data.columns


# In[9]:


data.isnull().sum()


# In[10]:


data['50d_MA']= data['Close'].rolling(window=50).mean()
data['200d_MA']=data['Close'].rolling(window=200).mean()


# In[11]:


data.head()


# In[12]:


data.isnull().sum()


# In[13]:


data.fillna(method='ffill', inplace=True)


# In[14]:


data.isnull().sum()


# In[15]:


data=data.dropna()


# In[16]:


data.isnull().sum()


# In[17]:


data.info()


# In[18]:


data.describe()


# In[19]:


data['Close'].plot()


# In[20]:


data['Open'].plot()


# In[21]:


data['Volume'].plot()


# In[22]:


data['Returns']=data['Close'].pct_change()


# In[23]:


data.head()


# In[24]:


data['Returns'].plot()


# In[25]:


import statsmodels.api as sm
import scipy.stats as stats


# In[26]:


stats.probplot(data['Returns'], dist='norm')


# In[27]:


data.dropna(inplace=True)


# In[28]:


stats.probplot(data['Returns'], dist='norm', plot=plt)


# In[29]:


plt.hist(data['Returns'], bins=1000)


# In[30]:


data['50d_MA'].plot()


# In[31]:


data['200d_MA'].plot()


# In[32]:


sb.kdeplot(data['Returns'])


# In[24]:


features= ['Open', 'High', 'Low', 'Close', 'Volume', '50d_MA', '200d_MA']
target='Close'


# In[25]:


X= data[features].dropna()
y= X.pop(target)


# In[26]:


y


# In[27]:


X_train, X_test, y_train, y_test= train_test_split(X,y , test_size= 0.2 , shuffle=False)


# In[28]:


model=LinearRegression()
model.fit(X_train, y_train)


# In[29]:


model.coef_


# In[30]:


model.intercept_


# In[31]:


predictions= model.predict(X_test)


# In[32]:


predictions


# In[33]:


predicted_next_day_close= predictions[-1]
#print(f"Predicted Next Days Closing Price: {predicted_next_day_close:.2f}")


# In[34]:


predicted_next_day_close


# In[35]:


mean_squared_error(y_test, predictions)


# In[36]:


predictions.size


# In[37]:


y_test.size


# In[38]:


plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, predictions, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()
plt.show()


# In[39]:


from sklearn.metrics import mean_squared_error, r2_score


# In[40]:


# = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# In[41]:


RMSE = 9.938712940 ** 0.5
print(RMSE)


# In[42]:


plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(data[col])
plt.show()


# In[43]:


plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.boxplot(data[col])
plt.show()


# In[ ]:





# In[44]:


from sklearn.metrics import  confusion_matrix,accuracy_score


# In[45]:


model.score(X_test,y_test)


# In[46]:


rmse = mean_squared_error(y_test, predictions) ** 0.5

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual Price', color='blue')
plt.plot(predictions, label='Predicted Price', color='red', linestyle='--')
plt.title(f'Actual vs Predicted Stock Prices (RMSE = {rmse:.2f})')
plt.xlabel('Time')
plt.ylabel('Stock Price (₹)')
plt.legend()
plt.grid(True)
plt.show()


# In[47]:


import klib


# In[48]:


klib.corr_plot(data, target='Close')


# In[49]:


klib.dist_plot(data['Returns'])


# In[50]:


klib.dist_plot(data['Close'])


# In[51]:


klib.cat_plot(data)


# In[52]:


klib.corr_plot(data)


# In[54]:


import dtale
d= dtale.show(data)
d.open_browser()


# In[88]:


data=data.dropna()


# In[89]:


features2= data[['Low', 'High', 'Volume', 'Returns', 'Open', '50d_MA', '200d_MA']]
target= data['Close']


# In[90]:


features2


# In[91]:


X=features2.dropna()


# In[92]:


X


# In[93]:


X.isnull().sum()


# In[94]:


y= target


# In[95]:


y


# In[96]:


y.isnull().sum()


# In[97]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size= 0.2, shuffle=False)


# In[98]:


X.value_counts()


# In[99]:


y.value_counts()


# In[100]:





# In[101]:


model= LinearRegression()
model.fit(X_train, y_train)


# In[102]:


model.coef_


# In[103]:


model.intercept_


# In[104]:


model.predict(X_test)


# In[107]:


predictions2 = model.predict(X_test)


# In[105]:


model.score(X_test, y_test)


# In[108]:


plt.figure(figsize=(14,7))
plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
plt.plot(y_test.index, predictions, label='Linear Regression', color='red', linestyle='--')
plt.title('Actual vs Predicted Stock Prices (HDFCBANK.NS)')
plt.xlabel('Date')
plt.ylabel('Stock Price (₹)')
plt.legend()
plt.grid(True)
plt.show()


# In[109]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[118]:


rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train.squeeze())
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)


# In[121]:


print(f"Random Forest MSE:    {mse_rf:.4f}")


# In[123]:


plt.figure(figsize=(14,7))
plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
plt.plot(y_test.index, predictions2, label='Linear Regression', color='red', linestyle='--')
plt.plot(y_test.index, y_pred_rf, label='Random Forest', color='green', linestyle=':')
plt.title('Actual vs Predicted Stock Prices (HDFCBANK.NS)')
plt.xlabel('Date')
plt.ylabel('Stock Price (₹)')
plt.legend()
plt.grid(True)
plt.show()


# In[113]:


y.shape


# In[114]:


y= data['Close']


# In[120]:


y.shape


# In[119]:


plt.figure(figsize=(10,6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.title('Feature Importance - Random Forest (HDFCBANK.NS)')
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# In[124]:


importances = rf.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance)


# In[130]:


plt.figure(figsize=(10,6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.title('Feature Importance - Random Forest (HDFCBANK.NS)')
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# In[126]:


feature_importance['Feature'].head()


# In[128]:


feature_importance['Feature'] = feature_importance['Feature'].apply(
    lambda x: x[0] if isinstance(x, tuple) else x )


# In[129]:


plt.figure(figsize=(10,6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title('Feature Importance - Random Forest (HDFCBANK.NS)')
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# In[ ]:




