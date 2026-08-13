# Application of Statistics in Time Series Analysis

# 1. Load libraries and packages


```python
from statsmodels.tsa.ar_model import AutoRegResults
```


```python
import pandas as pd 
import numpy as np 

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import plotly.figure_factory as ff
from statsmodels.graphics.tsaplots import plot_predict

# Machine Learning
import sklearn
## Pipeline
from sklearn.pipeline import Pipeline
## Column Transformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error

## Feature Union for Customized Class 
from sklearn.pipeline import FeatureUnion

# Statistics
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Cross-sectional models and methods
import statsmodels.api as sm

# Augmented Dicker Fuller Test for Stationary vs Random Walk
from statsmodels.tsa.stattools import adfuller

# Random array
from numpy.random import normal, seed

from plotly.offline import iplot

# Modeling 
## Autoregressive Model
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
## Moving Average Model
# from statsmodels.tsa.arima.model import ARMA
from statsmodels.tsa.arima.model import ARIMA


# Mathematic operations
import math

```

- `FeatureUnion` is a class from scikit-learn used to combine the outputs of multiple transformer objects into a single feature space.
    - It allows parallel application of different data preprocessing steps (e.g. scaling, encoding, feature extraction) on the same input dataset, then concatenates their results column-wise—similar to `ColumnTransformer`, but typically applied to the same columns rather than different ones.
- `MinMaxScaler` is a preprocessing class from sklearn.preprocessing used to scale numerical features to a fixed range, typically [0, 1].
    - It transforms data by subtracting the minimum value and dividing by the range (max – min), ensuring all features contribute equally to model training, especially important in algorithms sensitive to input scale like neural networks or distance-based models.
- `BaseEstimator` is a base class in scikit-learn used to define estimator interfaces.
    - It provides a foundational structure for creating custom machine learning estimators in Python, ensuring compatibility with scikit-learn's tools like pipelines and model selection utilities. Though not used directly, it enforces consistent method signatures (e.g., fit, get_params, set_params) across all estimators.
- `TransformerMixin` is a class mixin from scikit-learn that enables custom transformer classes to integrate seamlessly with scikit-learn's pipeline system.
    - It provides a standard fit_transform method so that any class inheriting from it can be used in a Pipeline or FeatureUnion without requiring boilerplate code. This is essential for building reusable, composable data preprocessing workflows.
- `rcParams` is a configuration object used to customize default settings in Matplotlib visualizations.
    - It allows users to globally adjust figure size, font, style, and other plot parameters without repeating settings across plots. This improves consistency and reduces boilerplate code in data visualization workflows.

# Dataset Information

1. [DJIA 30 Stock Time Series](https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231)
2. [Historical Hourly Weather Data 2012-2017](https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data)

- Brief description of datasets:
    - Data being used:
        - Google Stock Data
        - Humidity in different world cities
        - Microsoft Stock Data
        - Pressure in different world cities


```python
# pip install kagglehub
```


```python
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("szrlee/stock-time-series-20050101-to-20171231")

# print("Path to dataset files:", path)
```


```python
# # Download latest version
# path = kagglehub.dataset_download("selfishgene/historical-hourly-weather-data")

# print("Path to dataset files:", path)
```


```python
google_stock = pd.read_csv("stock-time-series-20050101-to-20171231\GOOGL_2006-01-01_to_2018-01-01.csv", index_col="Date", parse_dates=["Date"])
google_stock.head()
```

    <>:1: SyntaxWarning:
    
    invalid escape sequence '\G'
    
    <>:1: SyntaxWarning:
    
    invalid escape sequence '\G'
    
    C:\Users\yenpth8\AppData\Local\Temp\ipykernel_14780\1069917469.py:1: SyntaxWarning:
    
    invalid escape sequence '\G'
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Name</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2006-01-03</th>
      <td>211.47</td>
      <td>218.05</td>
      <td>209.32</td>
      <td>217.83</td>
      <td>13137450</td>
      <td>GOOGL</td>
    </tr>
    <tr>
      <th>2006-01-04</th>
      <td>222.17</td>
      <td>224.70</td>
      <td>220.09</td>
      <td>222.84</td>
      <td>15292353</td>
      <td>GOOGL</td>
    </tr>
    <tr>
      <th>2006-01-05</th>
      <td>223.22</td>
      <td>226.00</td>
      <td>220.97</td>
      <td>225.85</td>
      <td>10815661</td>
      <td>GOOGL</td>
    </tr>
    <tr>
      <th>2006-01-06</th>
      <td>228.66</td>
      <td>235.49</td>
      <td>226.85</td>
      <td>233.06</td>
      <td>17759521</td>
      <td>GOOGL</td>
    </tr>
    <tr>
      <th>2006-01-09</th>
      <td>233.44</td>
      <td>236.94</td>
      <td>230.70</td>
      <td>233.68</td>
      <td>12795837</td>
      <td>GOOGL</td>
    </tr>
  </tbody>
</table>
</div>




```python
humidity = pd.read_csv("historical-hourly-weather-data\humidity.csv", index_col="datetime", parse_dates=["datetime"])
humidity.tail()
```

    <>:1: SyntaxWarning:
    
    invalid escape sequence '\h'
    
    <>:1: SyntaxWarning:
    
    invalid escape sequence '\h'
    
    C:\Users\yenpth8\AppData\Local\Temp\ipykernel_14780\2666078636.py:1: SyntaxWarning:
    
    invalid escape sequence '\h'
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Vancouver</th>
      <th>Portland</th>
      <th>San Francisco</th>
      <th>Seattle</th>
      <th>Los Angeles</th>
      <th>San Diego</th>
      <th>Las Vegas</th>
      <th>Phoenix</th>
      <th>Albuquerque</th>
      <th>Denver</th>
      <th>...</th>
      <th>Philadelphia</th>
      <th>New York</th>
      <th>Montreal</th>
      <th>Boston</th>
      <th>Beersheba</th>
      <th>Tel Aviv District</th>
      <th>Eilat</th>
      <th>Haifa</th>
      <th>Nahariyya</th>
      <th>Jerusalem</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-11-29 20:00:00</th>
      <td>NaN</td>
      <td>81.0</td>
      <td>NaN</td>
      <td>93.0</td>
      <td>24.0</td>
      <td>72.0</td>
      <td>18.0</td>
      <td>68.0</td>
      <td>37.0</td>
      <td>18.0</td>
      <td>...</td>
      <td>27.0</td>
      <td>NaN</td>
      <td>64.0</td>
      <td>37.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-29 21:00:00</th>
      <td>NaN</td>
      <td>71.0</td>
      <td>NaN</td>
      <td>87.0</td>
      <td>21.0</td>
      <td>72.0</td>
      <td>18.0</td>
      <td>73.0</td>
      <td>34.0</td>
      <td>12.0</td>
      <td>...</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>59.0</td>
      <td>74.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-29 22:00:00</th>
      <td>NaN</td>
      <td>71.0</td>
      <td>NaN</td>
      <td>93.0</td>
      <td>23.0</td>
      <td>68.0</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>32.0</td>
      <td>15.0</td>
      <td>...</td>
      <td>31.0</td>
      <td>NaN</td>
      <td>66.0</td>
      <td>74.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-29 23:00:00</th>
      <td>NaN</td>
      <td>71.0</td>
      <td>NaN</td>
      <td>87.0</td>
      <td>14.0</td>
      <td>63.0</td>
      <td>17.0</td>
      <td>33.0</td>
      <td>30.0</td>
      <td>28.0</td>
      <td>...</td>
      <td>26.0</td>
      <td>NaN</td>
      <td>58.0</td>
      <td>56.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-30 00:00:00</th>
      <td>NaN</td>
      <td>76.0</td>
      <td>NaN</td>
      <td>75.0</td>
      <td>56.0</td>
      <td>72.0</td>
      <td>17.0</td>
      <td>23.0</td>
      <td>34.0</td>
      <td>31.0</td>
      <td>...</td>
      <td>32.0</td>
      <td>NaN</td>
      <td>58.0</td>
      <td>56.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
# Set color palette
palette = sns.color_palette("Set2") 
```

# 2. Exploration Data Analysis

# 2.1 Stock Dataset


```python
google_stock.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 3019 entries, 2006-01-03 to 2017-12-29
    Data columns (total 6 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   Open    3019 non-null   float64
     1   High    3019 non-null   float64
     2   Low     3019 non-null   float64
     3   Close   3019 non-null   float64
     4   Volume  3019 non-null   int64  
     5   Name    3019 non-null   object 
    dtypes: float64(4), int64(1), object(1)
    memory usage: 165.1+ KB
    

- Dataset of Google Stock includes:
    - Feature list:
        - 6 columns: Open, High, Low, Close, Volume
        - DatatimeIndex
            - Data range: 2006-01-03 to 2017-12-29
            - Frequency: Daily
        - Detail:
            - Date - in format: yy-mm-dd
            - Open - price of the stock at market open (this is NYSE data so all in USD)
            - High - Highest price reached in the day
            - Low - Lowest price reached in the day
            - Close - price of the stock at market close
            - Volume - Number of shares traded
            - Name - the stock's ticker name
    - Number of record: 3019
- Business Understanding:
    - Stock price for each day has 6 characteristics should be care:
        - Open price
        - Close price
        - Highest price
        - Lowest price
        - Volume: The number of stock transactions
        - Adjust close price: Have been existed in this dataset


```python
pd.date_range(start='2006-01-03', end='2017-12-29')
```




    DatetimeIndex(['2006-01-03', '2006-01-04', '2006-01-05', '2006-01-06',
                   '2006-01-07', '2006-01-08', '2006-01-09', '2006-01-10',
                   '2006-01-11', '2006-01-12',
                   ...
                   '2017-12-20', '2017-12-21', '2017-12-22', '2017-12-23',
                   '2017-12-24', '2017-12-25', '2017-12-26', '2017-12-27',
                   '2017-12-28', '2017-12-29'],
                  dtype='datetime64[ns]', length=4379, freq='D')



- About time range:
    - Existing the missing data in some days (4379 > 3019). 


```python
google_stock.isnull().sum()
```




    Open      0
    High      0
    Low       0
    Close     0
    Volume    0
    Name      0
    dtype: int64




```python
google_stock.duplicated().sum()
```




    np.int64(0)




```python
google_stock.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3019.000000</td>
      <td>3019.000000</td>
      <td>3019.000000</td>
      <td>3019.000000</td>
      <td>3.019000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>428.200802</td>
      <td>431.835618</td>
      <td>424.130275</td>
      <td>428.044001</td>
      <td>3.551504e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>236.320026</td>
      <td>237.514087</td>
      <td>234.923747</td>
      <td>236.343238</td>
      <td>3.038599e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>131.390000</td>
      <td>134.820000</td>
      <td>123.770000</td>
      <td>128.850000</td>
      <td>5.211410e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>247.775000</td>
      <td>250.190000</td>
      <td>244.035000</td>
      <td>247.605000</td>
      <td>1.760854e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>310.480000</td>
      <td>312.810000</td>
      <td>307.790000</td>
      <td>310.080000</td>
      <td>2.517630e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>572.140000</td>
      <td>575.975000</td>
      <td>565.900000</td>
      <td>570.770000</td>
      <td>4.242182e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1083.020000</td>
      <td>1086.490000</td>
      <td>1072.270000</td>
      <td>1085.090000</td>
      <td>4.118289e+07</td>
    </tr>
  </tbody>
</table>
</div>



- About the variability:
    - Haven't found any great range between Open, High, Low, Close in almost all of the statistical metrics.


```python
google_stock.shape
```




    (3019, 6)



### 2.1.1. Growth rate daily of Google stock

- Growth rate is the percentage change in the value of a variable over a period of time.
    - The important metric in stock market.
    - Formula: Current Value / Previous Value


```python
# Percent change in Highest price of Google stock
## Option 1: pct_change() method
google_stock['High'].pct_change(freq='D').add(1).plot(figsize=(20,8), color = palette[2])
## Option 2: div() 
# google_stock['High'].div(google_stock['High'].shift(1, freq='D')).plot(figsize=(20,8))
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_25_1.png)
    



```python
google_stock['Change'] = google_stock['High'].div(google_stock['High'].shift(1, freq='D'))

```


```python
# Percent change in Lowest price of Google stock
google_stock['Low'].pct_change(periods=1, freq='D').add(1).plot(figsize=(20,8), color = palette[2])
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_27_1.png)
    



```python
# Percent change in Open price of Google stock
google_stock['Open'].pct_change(periods=1, freq='D').add(1).plot(figsize=(20,8), color = palette[2])
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_28_1.png)
    



```python
# Percent change in Close price of Google stock
google_stock['Close'].pct_change(periods=1, freq='D').add(1).plot(figsize=(20,8), color = palette[2])
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_29_1.png)
    



```python
# Percent change in Volume of Google stock
google_stock['Volume'].pct_change(periods=1, freq='D').add(1).plot(figsize=(20,8), color = palette[2])
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_30_1.png)
    


### 2.1.2. Stock returns

-  Stock return is the percentage of change in the stock price from one period to the next.
    - Formula: (Current Value - Previous Value) / Previous Value


```python
google_stock['Change'].sub(1).mul(100)
```




    Date
    2006-01-03         NaN
    2006-01-04    3.049759
    2006-01-05    0.578549
    2006-01-06    4.199115
    2006-01-09         NaN
                    ...   
    2017-12-22   -0.538273
    2017-12-26         NaN
    2017-12-27   -0.055199
    2017-12-28   -0.321080
    2017-12-29   -0.637654
    Name: Change, Length: 3019, dtype: float64




```python
google_stock['Change'].sub(1).mul(100).plot(figsize=(20,8), color = palette[2])
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_34_1.png)
    


- `sub` is a method used to perform element-wise subtraction on a pandas Series or DataFrame.
    - It is typically called on a data column to subtract a scalar value or another series from each element, often used in financial or time-series calculations to compute changes or differences.
    - This line computes the percentage change by:
        - Subtracting 1 from each value in the 'Change' column (assumed to represent multiplicative factors, e.g., 1.05 for a 5% increase)
        - Multiplying the result by 100 to convert to a percentage

- `mul` is a method used to perform element-wise multiplication on data structures.
    - It is commonly used in data analysis workflows to scale or transform numerical values, particularly in pandas DataFrame or Series objects, though its exact implementation depends on the context in which it is called.
    - Purpose: Scales numerical data by a specified factor (here, 100) element-wise.
    - Object context: Likely a pandas Series or similar vectorized numeric type.
    - Returns: A new Series with each element multiplied by the argument.
    - Side effects: None — it is a pure transformation function.


```python
google_stock['High'].pct_change().add(1)
```




    Date
    2006-01-03         NaN
    2006-01-04    1.030498
    2006-01-05    1.005785
    2006-01-06    1.041991
    2006-01-09    1.006157
                    ...   
    2017-12-22    0.994617
    2017-12-26    0.997331
    2017-12-27    0.999448
    2017-12-28    0.996789
    2017-12-29    0.993623
    Name: High, Length: 3019, dtype: float64




```python
google_stock['High'].pct_change()
```




    Date
    2006-01-03         NaN
    2006-01-04    0.030498
    2006-01-05    0.005785
    2006-01-06    0.041991
    2006-01-09    0.006157
                    ...   
    2017-12-22   -0.005383
    2017-12-26   -0.002669
    2017-12-27   -0.000552
    2017-12-28   -0.003211
    2017-12-29   -0.006377
    Name: High, Length: 3019, dtype: float64




```python
google_stock['High'].div(google_stock['High'].shift(1, freq='D')).sub(1)
```




    Date
    2006-01-03         NaN
    2006-01-04    0.030498
    2006-01-05    0.005785
    2006-01-06    0.041991
    2006-01-07         NaN
                    ...   
    2017-12-26         NaN
    2017-12-27   -0.000552
    2017-12-28   -0.003211
    2017-12-29   -0.006377
    2017-12-30         NaN
    Name: High, Length: 3672, dtype: float64




```python
google_stock['High'].pct_change().mul(100) # Another way to calculate return
```




    Date
    2006-01-03         NaN
    2006-01-04    3.049759
    2006-01-05    0.578549
    2006-01-06    4.199115
    2006-01-09    0.615737
                    ...   
    2017-12-22   -0.538273
    2017-12-26   -0.266861
    2017-12-27   -0.055199
    2017-12-28   -0.321080
    2017-12-29   -0.637654
    Name: High, Length: 3019, dtype: float64



### 2.1.3. Specific value of change


```python
google_stock['High'].diff(periods = 1)
```




    Date
    2006-01-03     NaN
    2006-01-04    6.65
    2006-01-05    1.30
    2006-01-06    9.49
    2006-01-09    1.45
                  ... 
    2017-12-22   -5.80
    2017-12-26   -2.86
    2017-12-27   -0.59
    2017-12-28   -3.43
    2017-12-29   -6.79
    Name: High, Length: 3019, dtype: float64




```python
google_stock['High'].diff(periods = 1).plot(figsize = (20, 8), color = palette[2])
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_42_1.png)
    


- `diff` is a method used to compute the difference between consecutive elements in a time series or sequence.
    - It is commonly applied in data analysis and time series modeling to transform non-stationary data into stationary data by removing trends, which helps in preparing data for statistical modeling such as ARIMA or other forecasting techniques.

## 2.1.4. Compare two or more timeseries


- Compare 2 time series by normalizing them. 
    - Normalization: Dividing by each time series element of all time series by the first element.
        - Dividing by the start date of each time series.


```python
# Import dataset of microsoft stock
microsoft_stock = pd.read_csv('stock-time-series-20050101-to-20171231\MSFT_2006-01-01_to_2018-01-01.csv', 
                        index_col='Date', 
                        parse_dates=['Date'])
```

    <>:2: SyntaxWarning:
    
    invalid escape sequence '\M'
    
    <>:2: SyntaxWarning:
    
    invalid escape sequence '\M'
    
    C:\Users\yenpth8\AppData\Local\Temp\ipykernel_14780\2705525054.py:2: SyntaxWarning:
    
    invalid escape sequence '\M'
    
    


```python
# Ploting the absolute value of the Highest price in each time series
google_stock['High'].plot(color = palette[2])
microsoft_stock['High'].plot(color = palette[3])
plt.legend(['Google', 'Microsoft'])
plt.show()

```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_47_0.png)
    



```python
# Normalizing and Comparison both stocks 
normalized_google_stock = google_stock['High'].div(google_stock['High'].iloc[0]).mul(100)
normalized_microsoft_stock = microsoft_stock['High'].div(microsoft_stock['High'].iloc[0]).mul(100)

# Visualization
normalized_google_stock.plot(label='Google', color = palette[2])
normalized_microsoft_stock.plot(label='Microsoft', color = palette[3])
plt.legend()
plt.show()
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_48_0.png)
    


- Conclusion:
    - See clearly how google outperforms microsoft over time.


## 2.1.5. Window functions

- **What are Window functions and how are they useful?**
    - Window functions are used to identify sub periods, calculates sub-metrics of sub-periods.
    - **Rolling - Same size and sliding**
        - Each window has
            - End point: Current timestamp
            - Start point: End point - window size
            - Window size
        - If the current timestamp is the start date of timeseries, then the window is not full.
            - Size of window is less than window size.
            - Window size at that point = 1.
    - **Expanding - Contains all prior values**
        - Each window has
            - End point: Current timestamp
            - Start point: Start date of timeseries
            - Window size: End point - start point


```python
google_stock['High']
```




    Date
    2006-01-03     218.05
    2006-01-04     224.70
    2006-01-05     226.00
    2006-01-06     235.49
    2006-01-09     236.94
                   ...   
    2017-12-22    1071.72
    2017-12-26    1068.86
    2017-12-27    1068.27
    2017-12-28    1064.84
    2017-12-29    1058.05
    Name: High, Length: 3019, dtype: float64



### Rolling Window


```python
google_stock['High'].rolling(window='90D').mean()
```




    Date
    2006-01-03     218.050000
    2006-01-04     221.375000
    2006-01-05     222.916667
    2006-01-06     226.060000
    2006-01-09     228.236000
                     ...     
    2017-12-22    1031.255000
    2017-12-26    1035.805161
    2017-12-27    1037.451774
    2017-12-28    1038.887742
    2017-12-29    1039.191905
    Name: High, Length: 3019, dtype: float64



- [`pandas.DataFrame.rolling`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html#pandas.DataFrame.rolling) is a method in pandas used to define a rolling window for time series or sequential data analysis.
    - It enables the calculation of rolling (moving) statistics such as mean, standard deviation, sum, etc., over a specified window size.
    - Params: Takes a required `window` parameter (int, timedelta, str, offset, or BaseIndexer subclass) specifying the number of periods to include in the rolling calculation. Optional parameters include `min_periods`, `center`, `win_type`, etc.
    - Side effects: None — returns a new Rolling object; does not modify original data.
    - Returns: A pandas.core.window.rolling.Rolling object, which can then be used to compute aggregate functions like `.mean()`, `.std()`, `.sum()`, etc.


```python
google_stock['High'].plot(label = 'High', color = palette[2])
google_stock['High'].rolling(window='90D').mean().plot(label = 'Rolling Mean 90D', color = palette[5])
plt.legend()
plt.show()
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_55_0.png)
    


- Usecase:
    - Rolling mean plot is a smoother version of the original plot.

### Expand Window


```python
microsoft_stock['High'].plot(label = 'High', color = palette[3])
microsoft_stock['High'].expanding().mean().plot(label = 'Expanding Mean', color = palette[6])
microsoft_stock['High'].expanding().std().plot(label = 'Expanding Standard Deviation', color = palette[7])
plt.legend()
plt.show()
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_58_0.png)
    


## 2.1.6. OHLC charts [Soon!]

## 2.1.7. Candlestick charts [Soon!]

## 2.1.8. Trends, Seasonality, Noise

- These are the components of a time series
    - Trend - Consistent upwards or downwards slope of a time series
    - Seasonality - Clear periodic pattern of a time series(like `sine` funtion)
    - Noise - Outliers or missing values


```python
rcParams['figure.figsize'] = 11, 9
decomposed_google_volume = sm.tsa.seasonal_decompose(google_stock["High"], model = 'additive', period = 360) # The frequency is annual
figure = decomposed_google_volume.plot()
plt.show()
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_62_0.png)
    


| Bước                                           | Mô tả kỹ thuật                                                                                | Giải thích dễ hiểu                                                                       |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **Bước 1: Ước lượng Trend (xu hướng)**         | Dùng **moving average (trung bình trượt)** với chu kỳ bằng `period` hoặc `freq` bạn cung cấp. | Làm mượt dữ liệu để thấy đường “nền” dài hạn — ví dụ: giá cổ phiếu đang đi lên theo năm. |
| **Bước 2: Loại bỏ Trend khỏi chuỗi gốc**       | Tính phần dư tạm thời:  `Yt - Tt` (với additive) hoặc `Yt / Tt` (với multiplicative).         | Sau khi bỏ xu hướng dài hạn, phần còn lại là biến động ngắn hạn (seasonality + noise).   |
| **Bước 3: Ước lượng Seasonal (mùa vụ)**        | Gom các giá trị theo chu kỳ (ví dụ: trung bình của tất cả tháng 1, tháng 2, …).               | Xác định mẫu lặp lại đều đặn trong năm / tháng / tuần.                                   |
| **Bước 4: Tính Residual (phần dư ngẫu nhiên)** | Lấy phần còn lại: `Rt = Yt - Tt - St` (hoặc `Yt / (Tt × St)`)                                 | Phần không thể giải thích – thể hiện yếu tố bất thường hoặc sai số đo lường.             |

- Trend (Manual caculation)


```python
google_stock["High"].rolling(window='360D').mean().plot(figsize = (20, 4), label = 'Trend', color = palette[2])
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_65_1.png)
    



```python
# google_stock.drop(['Seasonal_noise_D'], axis = 1)
```

- Seasonal by Quartern (Manual caculation)


```python
google_stock['Seasonal_noise_90D'] = google_stock["High"] - google_stock["High"].rolling(window='90D').mean()
google_stock['Seasonal_noise_90D'].plot(figsize = (20, 4), label = 'Seasonal + Noise', color = palette[2])
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_68_1.png)
    



```python
seasonal_means = google_stock['Seasonal_noise_90D'].groupby(google_stock.index.quarter).mean()
google_stock['seasonal_90D'] = (google_stock.index.quarter).map(seasonal_means)
google_stock['seasonal_90D'].plot(figsize = (20, 8), label = 'Seasonal', color = palette[2])

```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_69_1.png)
    


- Seasonal by Week (Manual caculation)


```python
google_stock['Seasonal_noise_7D'] = google_stock["High"] - google_stock["High"].rolling(window='7D').mean()
google_stock['Seasonal_noise_7D'].plot(figsize = (20, 4), label = 'Seasonal + Noise', color = palette[2])
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_71_1.png)
    



```python
seasonal_means = google_stock['Seasonal_noise_7D'].groupby(google_stock.index.isocalendar().week).mean()

google_stock['seasonal_7D'] = (google_stock.index.isocalendar().week).map(seasonal_means)

google_stock['seasonal_7D'].plot(figsize = (20, 8), label = 'Seasonal', color = palette[2])

```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_72_1.png)
    


- Seasonal by 2 Weeks (Manual caculation)


```python
google_stock['Seasonal_noise_14D'] = google_stock["High"] - google_stock["High"].rolling(window='14D').mean()
google_stock['Seasonal_noise_14D'].plot(figsize = (20, 4), label = 'Seasonal + Noise', color = palette[2])
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_74_1.png)
    



```python
seasonal_means = google_stock['Seasonal_noise_14D'].groupby((google_stock.index.isocalendar().week - 1)//2 + 1).mean()

google_stock['seasonal_14D'] = ((google_stock.index.isocalendar().week - 1)//2 + 1).map(seasonal_means)

google_stock['seasonal_14D'].plot(figsize = (20, 8), label = 'Seasonal', color = palette[2])

```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_75_1.png)
    


- Seasonal by 4 Days (Manual caculation)


```python
google_stock['Seasonal_noise_4D'] = google_stock["High"] - google_stock["High"].rolling(window='4D').mean()
google_stock['Seasonal_noise_4D'].plot(figsize = (20, 4), label = 'Seasonal + Noise', color = palette[2])
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_77_1.png)
    



```python
seasonal_means = google_stock['Seasonal_noise_4D'].groupby((google_stock.index.dayofyear - 1)//4 + 1).mean()

google_stock['seasonal_4D'] = ((google_stock.index.dayofyear - 1)//4 + 1).map(seasonal_means)

google_stock['seasonal_4D'].plot(figsize = (20, 8), label = 'Seasonal', color = palette[2])

```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_78_1.png)
    


- Seasonal by Month (Manual caculation)


```python
google_stock['Seasonal_noise_Month'] = google_stock["High"] - google_stock["High"].rolling(window='30D').mean()
google_stock['Seasonal_noise_Month'].plot(figsize = (20, 4), label = 'Seasonal + Noise', color = palette[2])
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_80_1.png)
    



```python
seasonal_means = google_stock['Seasonal_noise_Month'].groupby(google_stock.index.month).mean()

google_stock['seasonal_Month'] = (google_stock.index.month).map(seasonal_means)

google_stock['seasonal_Month'].plot(figsize = (20, 8), label = 'Seasonal', color = palette[2])

```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_81_1.png)
    


- Residual (Manual caculation)


```python
google_stock['residual'] = google_stock['High'] - google_stock["High"].rolling(window='90D').mean() - google_stock['seasonal_90D']

google_stock['residual'].plot(figsize = (20, 8), label = 'Trend', color = palette[2])
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_83_1.png)
    


| Component    | What it means                                                    | Business translation                                                                                    |
| ------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Observed** | The original stock price data (daily or periodic highs).         | The actual market performance you see.                                                                  |
| **Trend**    | The long-term upward or downward direction in data.              | The general growth or decline in Google’s market value over years.                                      |
| **Seasonal** | The repeating short-term pattern (e.g., monthly or yearly).      | Predictable fluctuations in stock price due to investor behavior, earnings seasons, or macro events.    |
| **Residual** | What’s left after removing trend and seasonality (random noise). | Irregular changes caused by unexpected events like news shocks, product launches, or regulatory issues. |

| Key finding | Business interpretation | Implication |
| --- | --- | --- | 
| **Upward Trend** | Over the years (2006–2018), Google’s stock consistently increased. This shows **sustained business growth** and rising investor confidence. | The company’s long-term fundamentals are positive; suitable for long-term investors. |
| **Uniform Seasonality** | The stock exhibits **repetitive fluctuations** — possibly due to annual cycles like **quarterly earnings reports, advertising seasons, or tech launches**. This means certain times of the year are predictably stronger. | Trading strategies or marketing campaigns can be aligned with recurring cycles. |
| **Residual / Noise** | Irregular spikes or drops that don’t follow the pattern — could represent **market reactions to news, product events, or global shocks**. These are **non-systematic** and hard to forecast. | Helps analysts separate real performance from short-term volatility or data errors. |

- [`sm.tsa.seasonal_decompose`](https://www.statsmodels.org/stable/release/version0.6.html#seasonal-decomposition)is a function from the statsmodels library used to decompose time series data into trend, seasonal, and residual components.
    - It applies classical seasonal decomposition via moving averages, enabling analysis of recurring patterns and underlying trends in time-based datasets.
    - Params:
        - x: array_like – Time series data (1D or 2D with series in columns)
        - `model`: {"additive", "multiplicative"} – Type of decomposition; default is "additive"
        - filt: array_like, optional – Filter coefficients for trend extraction
        - `period`: int, optional – The number of observations per seasonal cycle (replaces freq)
        - `freq`: int, optional – Deprecated; use period instead (e.g., 12 for monthly, 360 for daily annual cycles)
        - two_sided: bool – Whether the moving average is centered (default True)
        - extrapolate_trend: int or "freq" – Number of observations to extrapolate trend for at boundaries
    - Side effects: None – the function is purely analytical
    - Returns: A DecomposeResult object with attributes .trend, .seasonal, .resid, and .observed
    - Plotting: The result has a .plot() method that visualizes all components
        - Note: The freq parameter is deprecated in newer versions of statsmodels; period is preferred.

## 2.1.9. White noise

- Note: Don't know the reason why "White noise" existing in this report.

- White noise includes:
    - Constant mean
    - Constant variance
    - Zero auto-correlation at all lags


```python
# Plotting white noise
rcParams['figure.figsize'] = 16, 6
white_noise = np.random.normal(loc=0, scale=1, size=1000)
# loc = mean of the distribution
# scale = standard deviation (not variance, despite comment)
# size = number of random samples to generate
plt.plot(white_noise)
```




    [<matplotlib.lines.Line2D at 0x2415574ae90>]




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_88_1.png)
    


- See random ups and downs without trend or pattern.


```python
# Plotting autocorrelation of white noise
plot_acf(white_noise,lags=20)
plt.show()
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_90_0.png)
    


- If compute ACF (autocorrelation function), all lags ≈ 0 → confirms it’s white noise.

- Business term:
    - White noise = "natural randomness".
    - If the forecast errors look like white noise, the model is strong.
    - If not — there’s still a pattern left to learn.

| Aspect | White Noise |
| --- | --- |
| Pattern | None (completely random) |
| Predictability | None |
| Mean & variance | Constant |
| Ideal residual | Yes, for good forecasting models |
| Business meaning | What’s left after removing all explainable structure (trend + seasonality) |


| Use case | How white noise helps | Business interpretation |
| --- | --- | --- |
| **Model checking (ARIMA, forecasting)** | After fitting a model, if residuals are white noise → your model has captured all real patterns | Good: model explains everything that’s explainable |
| **Anomaly detection** | White noise defines “normal randomness” → any large deviation from it might be an outlier | Detect sales spikes, fraud, system errors |
| **Risk modeling (finance)** | Returns with white noise imply efficient market — no predictable profit opportunities | Used to test market efficiency |
| **Process control** | In manufacturing or operations, output should behave like white noise if process is stable | Any pattern → problem in production |


## 2.1.10. Random Walk

A random walk is a mathematical object, known as a stochastic or random process, that describes a path that consists of a succession of random steps on some mathematical space such as the integers.


**Basic Random Walk**

$P_t = P_{t-1} + \varepsilon_t$

* $(P_t)$: price (or value) at time *t*
* $(\varepsilon_t)$: random noise (unexpected shock)

**Random Walk with Drift**

$P_t = \mu + P_{t-1} + \varepsilon_t$ 

* $\mu$: small constant drift (average upward or downward movement)
* Example: a stock with a long-term upward trend.

**Basic Random Walk**

$[
P_t = P_{t-1} + \varepsilon_t
]$

* $(P_t)$: price (or value) at time *t*
* $(\varepsilon_t)$: random noise (unexpected shock)

**Random Walk with Drift**

$[
P_t = \mu + P_{t-1} + \varepsilon_t
]$ 

* $(\mu)$: small constant drift (average upward or downward movement)
* Example: a stock with a long-term upward trend.

| Use Case | Explanation | Why Random Walk Important |
| --- | --- | --- |
| **Stock price forecasting** | Stock prices are often modeled as random walks. | Because if prices follow a random walk, **past prices can’t predict future ones** → markets are “efficient.” |
| **Stationarity testing (ADF Test)** | To decide whether a time series is stationary or not. | A **non-stationary** series usually behaves like a random walk. |
| **Simulation of future values** | Monte Carlo simulations for price paths. | Random walk is used to **generate many possible future scenarios**. |
| **Model testing** | Used to compare ARIMA or LSTM results. | You can test if your model performs better than a random walk baseline. |


Regression test for random walk

$P_t = \alpha + \beta P_{t-1} + \epsilon_t$

Equivalent to $P_t - P_{t-1} = \alpha + \beta P_{t-1} + \epsilon_t$

Test:

$H_0: \beta = 1$ 
(The series has a unit root → Random walk (non-stationary))<br/>
$H_1: \beta < 1$
(This is not a random walk)

Dickey-Fuller Test:

$H_0: \beta = 0$
(This is a random walk)<br/>
$H_1: \beta < 0$
(This is not a random walk)

**Augmented Dickey-Fuller test**

An augmented Dickey–Fuller test (ADF) tests the null hypothesis that a unit root is present in a time series sample. It is basically Dickey-Fuller test with more lagged changes on RHS.
- to augment /ɔːɡˈment/: to increase the size or value of something by adding something to it, to make something larger or fuller by adding something to it
    - VN: tăng cường
    - Ex: He augmented his income by taking a second job

- **Vấn đề mà Dickey-Fuller Test muốn giải quyết**
    - Khi bạn làm việc với dữ liệu chuỗi thời gian (time series), ví dụ:
        - Giá cổ phiếu
        - Tỷ giá
        - Doanh số bán hàng

    → Bạn cần biết: **Chuỗi đó có “tính dừng” hay không?**

- **Tính dừng (Stationarity) là gì?**

    | Đặc điểm | Chuỗi có tính dừng | Chuỗi không có tính dừng (như Random Walk) |
    | --- | --- | --- |
    | **Mean (trung bình)**     | Không thay đổi theo thời gian | Thay đổi dần theo thời gian                |
    | **Variance (phương sai)** | Ổn định                       | Tăng theo thời gian                        |
    | **Dễ dự đoán?**           | Có thể dự đoán phần nào       | Gần như không thể dự đoán                  |

    <br/>

    > Nếu chuỗi **không dừng**, các mô hình thống kê (như ARIMA, Regression, Machine Learning) sẽ **dự đoán sai lệch hoặc kém chính xác**.

- **Dickey-Fuller Test kiểm tra điều gì**

    Mục tiêu của **Dickey-Fuller Test**:

    > Kiểm tra xem chuỗi có phải là **Random Walk (không dừng)** hay **Stationary (dừng)**.

- **Công thức cơ bản**

    Giả sử chuỗi dữ liệu:
    $P_t = \alpha + \beta P_{t-1} + \varepsilon_t$

    Ta **trừ $P_{t-1}$** hai vế →

    $P_t - P_{t-1} = \alpha + (\beta - 1)P_{t-1} + \varepsilon_t$<br/>
    $\Delta P_t = \alpha + \gamma P_{t-1} + \varepsilon_t$<br/>
    trong đó $\gamma = (\beta - 1)$

- **Giả thuyết kiểm định**

    | Ký hiệu             | Giả thuyết         | Ý nghĩa                               |
    | ------------------- | ------------------ | ------------------------------------- |
    | $H_0$: $\gamma = 0$ | Có unit root       | Chuỗi là **random walk** (không dừng) |
    | $H_1$: $\gamma < 0$ | Không có unit root | Chuỗi **stationary** (dừng)           |

- **Cách đọc kết quả**

    | Kết quả kiểm định  | Kết luận           | Giải thích đơn giản                            |
    | ------------------ | ------------------ | ---------------------------------------------- |
    | **p-value < 0.05** | Bác bỏ $H_0$       | Chuỗi **có tính dừng**, KHÔNG phải random walk |
    | **p-value ≥ 0.05** | Không bác bỏ $H_0$ | Chuỗi **không dừng**, hành xử như random walk  |

- **Ví dụ trực quan**

    | Trường hợp                                                              | Diễn giải                                     |
    | ----------------------------------------------------------------------- | --------------------------------------------- |
    | Nếu bạn test giá cổ phiếu AAPL và p-value = 0.8                         | → Giá AAPL biến động ngẫu nhiên (random walk) |
    | Nếu bạn test phần **thay đổi giá hàng ngày (ΔPrice)** và p-value = 0.01 | → Chuỗi ΔPrice **đã dừng** (stationary)       |

- **Mở rộng: Augmented Dickey-Fuller (ADF) Test**

    Dickey-Fuller cơ bản giả định chỉ có **1 độ trễ (lag)**.
    Nhưng trong thực tế, dữ liệu có thể có nhiều độ trễ → **ADF test** (Augmented Dickey-Fuller) sẽ thêm các độ trễ này để kiểm tra chính xác hơn.

    $\Delta P_t = \alpha + \gamma P_{t-1} + \sum_{i=1}^{p}\delta_i \Delta P_{t-i} + \varepsilon_t$



```python
# Augmented Dicky-Fuller (ADF) Test om Volume of Google Stock
ggs_adf = adfuller(google_stock['Volume'])
print(f"p-value of Google: {float(ggs_adf[1]):.10f}")
print("p-value of Google: {}".format(float(ggs_adf[1])))

mcrs_adf = adfuller(microsoft_stock['Volume'])
print(f"p-value of Microsoft: {float(mcrs_adf[1]):.10f}")
print("p-value of Microsoft: {}".format(float(mcrs_adf[1])))
```

    p-value of Google: 0.0000006511
    p-value of Google: 6.510719605768891e-07
    p-value of Microsoft: 0.0003201525
    p-value of Microsoft: 0.0003201525277652051
    

- Conclusion:
    - Microsoft has p-value 0.0003201525 which is less than 0.05, null hypothesis is rejected and this is not a random walk.
    - Google has p-value 0.0000006510 which is more than 0.05, null hypothesis is rejected and this is not a random walk.

- `adfuller` is a Python function from the `statsmodels` library used to perform the Augmented Dickey-Fuller (ADF) test for stationarity in time series data.
    - It tests whether a unit root is present in a time series, helping determine if the series is stationary or requires differencing.
    - Params:
            - x: array-like, 1-dimensional – the time series to test
            - maxlag: int, optional – maximum number of lags to include in the test regression
            - autolag: {'AIC', 'BIC', 't-stat', None} – method to automatically select lag length
            - regression: {'c', 'ct', 'ctt', 'nc'} – type of regression: constant, trend, etc.
            - store: bool – if True, returns additional results
            - regresults: bool – if True, returns regression results
    - Returns:
        - tuple containing:
            - Test Statistic
            - p-value
            - Number of lags used
            - Number of observations used
            - Critical Values (at 1%, 5%, 10%)
            - Optional: icbest, resstore (if autolag or store is enabled)



```python
seed(42)
rcParams['figure.figsize'] = 16, 6
random_walk = normal(loc=0, scale=0.01, size=1000) # Random array
plt.plot(random_walk)
plt.show()
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_104_0.png)
    



```python
fig = ff.create_distplot([random_walk],['Random Walk'],bin_size=0.001)
iplot(fig, filename='Basic Distplot')
```



- `create_distplot` is a function from the Plotly library used to generate distribution plots.
    - It visualizes the distribution of a dataset by plotting histograms, kernel density estimates (KDE), and rug plots in a single figure, making it easier to analyze data spread and shape.
    - Params:
        - data: List of arrays, each representing a dataset (e.g., [random_walk]).
        - group_labels: List of strings labeling each dataset (e.g., ['Random Walk']).
        - bin_size: Optional float controlling histogram bin width (set to 0.001 here).
    - Side effects: Generates an interactive Plotly figure object.
    - Returns: A Plotly Figure object containing the distribution plot, which can be rendered using iplot.

## 2.1.11. Stationarity

- A stationary time series is one whose statistical properties such as mean, variance, autocorrelation, etc. are all constant over time.
    - Strong stationarity: is a stochastic process whose unconditional joint probability distribution does not change when shifted in time. Consequently, parameters such as mean and variance also do not change over time.
    - Weak stationarity: is a process where mean, variance, autocorrelation are constant throughout the time
- Stationarity is important as non-stationary series that depend on time have too many parameters to account for when modelling the time series. `diff()` method can easily convert a non-stationary series to a stationary series.

We will try to decompose seasonal component of the above decomposed time series.


```python
# The original non-stationary plot
decomposed_google_volume.trend.plot()
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_108_1.png)
    



```python
# Convert A non-stationary series to the new stationary plot <> The residual
decomposed_google_volume.trend.diff().plot()
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_109_1.png)
    


- `diff` is a method used to compute the first discrete difference of elements in a time series or sequence.
    - It is commonly used in time series analysis to transform non-stationary data into stationary data by subtracting the previous value from the current value, helping stabilize the mean and remove trends.
    - Basic information:
        - Type: Method of pandas Series/DataFrame
        - Purpose: Compute the first discrete difference (i.e., x[i] - x[i-1])
        - Returns: A new Series or DataFrame of the same size with NaN in the first position
        - Parameters: periods=1 by default (can be adjusted for higher-order differencing)
        - Side effects: None — it returns a transformed copy



```python
decomposed_google_volume.resid.plot()
```




    <Axes: xlabel='Date'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_111_1.png)
    


# 2.2 Humidity Dataset


```python
humidity.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 45253 entries, 2012-10-01 12:00:00 to 2017-11-30 00:00:00
    Data columns (total 36 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Vancouver          43427 non-null  float64
     1   Portland           44804 non-null  float64
     2   San Francisco      44311 non-null  float64
     3   Seattle            44964 non-null  float64
     4   Los Angeles        45101 non-null  float64
     5   San Diego          44909 non-null  float64
     6   Las Vegas          44411 non-null  float64
     7   Phoenix            43945 non-null  float64
     8   Albuquerque        44543 non-null  float64
     9   Denver             43445 non-null  float64
     10  San Antonio        44689 non-null  float64
     11  Dallas             44934 non-null  float64
     12  Houston            45132 non-null  float64
     13  Kansas City        44741 non-null  float64
     14  Minneapolis        44743 non-null  float64
     15  Saint Louis        43964 non-null  float64
     16  Chicago            44144 non-null  float64
     17  Nashville          44686 non-null  float64
     18  Indianapolis       44558 non-null  float64
     19  Atlanta            44831 non-null  float64
     20  Detroit            44391 non-null  float64
     21  Jacksonville       45044 non-null  float64
     22  Charlotte          44664 non-null  float64
     23  Miami              44166 non-null  float64
     24  Pittsburgh         44731 non-null  float64
     25  Toronto            44525 non-null  float64
     26  Philadelphia       44629 non-null  float64
     27  New York           43629 non-null  float64
     28  Montreal           43557 non-null  float64
     29  Boston             44804 non-null  float64
     30  Beersheba          44394 non-null  float64
     31  Tel Aviv District  44140 non-null  float64
     32  Eilat              44283 non-null  float64
     33  Haifa              44435 non-null  float64
     34  Nahariyya          44436 non-null  float64
     35  Jerusalem          44347 non-null  float64
    dtypes: float64(36)
    memory usage: 12.8 MB
    

- Dataset of Humidity includes:
    - Feature list:
        - 36 columns: 30 US and Canadian Cities, as well as 6 Israeli cities
        - DatatimeIndex
            - Data range: 2012-10-01 12:00:00 to 2017-11-30 00:00:00
            - Frequency: Hourly
    - Number of record: 45253
- About missing data in specific cities:
    - Found missing data in specific cities in some hours.


```python
pd.date_range(start='2012-10-01 12:00:00', end='2017-11-30 00:00:00', freq='h')
```




    DatetimeIndex(['2012-10-01 12:00:00', '2012-10-01 13:00:00',
                   '2012-10-01 14:00:00', '2012-10-01 15:00:00',
                   '2012-10-01 16:00:00', '2012-10-01 17:00:00',
                   '2012-10-01 18:00:00', '2012-10-01 19:00:00',
                   '2012-10-01 20:00:00', '2012-10-01 21:00:00',
                   ...
                   '2017-11-29 15:00:00', '2017-11-29 16:00:00',
                   '2017-11-29 17:00:00', '2017-11-29 18:00:00',
                   '2017-11-29 19:00:00', '2017-11-29 20:00:00',
                   '2017-11-29 21:00:00', '2017-11-29 22:00:00',
                   '2017-11-29 23:00:00', '2017-11-30 00:00:00'],
                  dtype='datetime64[ns]', length=45253, freq='h')



- About time range: 
    - Haven't found any missing datatimeindex.


```python
humidity.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Vancouver</th>
      <th>Portland</th>
      <th>San Francisco</th>
      <th>Seattle</th>
      <th>Los Angeles</th>
      <th>San Diego</th>
      <th>Las Vegas</th>
      <th>Phoenix</th>
      <th>Albuquerque</th>
      <th>Denver</th>
      <th>...</th>
      <th>Philadelphia</th>
      <th>New York</th>
      <th>Montreal</th>
      <th>Boston</th>
      <th>Beersheba</th>
      <th>Tel Aviv District</th>
      <th>Eilat</th>
      <th>Haifa</th>
      <th>Nahariyya</th>
      <th>Jerusalem</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>43427.000000</td>
      <td>44804.000000</td>
      <td>44311.000000</td>
      <td>44964.000000</td>
      <td>45101.000000</td>
      <td>44909.000000</td>
      <td>44411.000000</td>
      <td>43945.000000</td>
      <td>44543.000000</td>
      <td>43445.000000</td>
      <td>...</td>
      <td>44629.000000</td>
      <td>43629.000000</td>
      <td>43557.000000</td>
      <td>44804.000000</td>
      <td>44394.000000</td>
      <td>44140.000000</td>
      <td>44283.000000</td>
      <td>44435.000000</td>
      <td>44436.000000</td>
      <td>44347.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>81.895480</td>
      <td>74.697616</td>
      <td>76.875042</td>
      <td>77.159038</td>
      <td>62.773841</td>
      <td>67.784809</td>
      <td>31.937831</td>
      <td>37.484424</td>
      <td>45.186157</td>
      <td>53.022557</td>
      <td>...</td>
      <td>68.017769</td>
      <td>66.642417</td>
      <td>71.861538</td>
      <td>77.375301</td>
      <td>70.604857</td>
      <td>66.861509</td>
      <td>53.155184</td>
      <td>79.800383</td>
      <td>78.606760</td>
      <td>68.732293</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.522221</td>
      <td>19.042656</td>
      <td>17.396016</td>
      <td>18.147464</td>
      <td>21.818042</td>
      <td>19.419307</td>
      <td>20.041855</td>
      <td>21.662728</td>
      <td>23.336546</td>
      <td>23.905392</td>
      <td>...</td>
      <td>18.790524</td>
      <td>19.874727</td>
      <td>16.825027</td>
      <td>18.750190</td>
      <td>21.321606</td>
      <td>16.464177</td>
      <td>27.305008</td>
      <td>23.051692</td>
      <td>23.682244</td>
      <td>19.273881</td>
    </tr>
    <tr>
      <th>min</th>
      <td>12.000000</td>
      <td>10.000000</td>
      <td>6.000000</td>
      <td>13.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>...</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>7.000000</td>
      <td>11.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>73.000000</td>
      <td>63.000000</td>
      <td>68.000000</td>
      <td>66.000000</td>
      <td>48.000000</td>
      <td>56.000000</td>
      <td>16.000000</td>
      <td>21.000000</td>
      <td>26.000000</td>
      <td>33.000000</td>
      <td>...</td>
      <td>54.000000</td>
      <td>51.000000</td>
      <td>60.000000</td>
      <td>65.000000</td>
      <td>54.000000</td>
      <td>58.000000</td>
      <td>31.000000</td>
      <td>63.000000</td>
      <td>61.000000</td>
      <td>56.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>86.000000</td>
      <td>80.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>66.000000</td>
      <td>71.000000</td>
      <td>27.000000</td>
      <td>32.000000</td>
      <td>42.000000</td>
      <td>52.000000</td>
      <td>...</td>
      <td>68.000000</td>
      <td>68.000000</td>
      <td>74.000000</td>
      <td>81.000000</td>
      <td>77.000000</td>
      <td>69.000000</td>
      <td>48.000000</td>
      <td>89.000000</td>
      <td>87.000000</td>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>93.000000</td>
      <td>90.000000</td>
      <td>89.000000</td>
      <td>93.000000</td>
      <td>81.000000</td>
      <td>82.000000</td>
      <td>43.000000</td>
      <td>50.000000</td>
      <td>63.000000</td>
      <td>73.000000</td>
      <td>...</td>
      <td>84.000000</td>
      <td>83.000000</td>
      <td>86.000000</td>
      <td>93.000000</td>
      <td>88.000000</td>
      <td>78.000000</td>
      <td>75.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>83.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>...</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 36 columns</p>
</div>



- Max of humidity is 100.
- Range of humidity is 0 to 100.
- About the variability between cities:
    - Exist the difference between cities.


```python
humidity.isnull().sum()
```




    Vancouver            1826
    Portland              449
    San Francisco         942
    Seattle               289
    Los Angeles           152
    San Diego             344
    Las Vegas             842
    Phoenix              1308
    Albuquerque           710
    Denver               1808
    San Antonio           564
    Dallas                319
    Houston               121
    Kansas City           512
    Minneapolis           510
    Saint Louis          1289
    Chicago              1109
    Nashville             567
    Indianapolis          695
    Atlanta               422
    Detroit               862
    Jacksonville          209
    Charlotte             589
    Miami                1087
    Pittsburgh            522
    Toronto               728
    Philadelphia          624
    New York             1624
    Montreal             1696
    Boston                449
    Beersheba             859
    Tel Aviv District    1113
    Eilat                 970
    Haifa                 818
    Nahariyya             817
    Jerusalem             906
    dtype: int64




```python
humidity.duplicated().sum()
```




    np.int64(678)




```python
humidity.shape
```




    (45253, 36)




```python
humidity = humidity.drop_duplicates()
```


```python
humidity.shape
```




    (44575, 36)




```python
humidity = humidity.fillna(method='ffill')
humidity.tail()
```

    C:\Users\yenpth8\AppData\Local\Temp\ipykernel_14780\901494674.py:1: FutureWarning:
    
    DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Vancouver</th>
      <th>Portland</th>
      <th>San Francisco</th>
      <th>Seattle</th>
      <th>Los Angeles</th>
      <th>San Diego</th>
      <th>Las Vegas</th>
      <th>Phoenix</th>
      <th>Albuquerque</th>
      <th>Denver</th>
      <th>...</th>
      <th>Philadelphia</th>
      <th>New York</th>
      <th>Montreal</th>
      <th>Boston</th>
      <th>Beersheba</th>
      <th>Tel Aviv District</th>
      <th>Eilat</th>
      <th>Haifa</th>
      <th>Nahariyya</th>
      <th>Jerusalem</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-11-29 20:00:00</th>
      <td>87.0</td>
      <td>81.0</td>
      <td>22.0</td>
      <td>93.0</td>
      <td>24.0</td>
      <td>72.0</td>
      <td>18.0</td>
      <td>68.0</td>
      <td>37.0</td>
      <td>18.0</td>
      <td>...</td>
      <td>27.0</td>
      <td>58.0</td>
      <td>64.0</td>
      <td>37.0</td>
      <td>57.0</td>
      <td>60.0</td>
      <td>100.0</td>
      <td>96.0</td>
      <td>96.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>2017-11-29 21:00:00</th>
      <td>87.0</td>
      <td>71.0</td>
      <td>22.0</td>
      <td>87.0</td>
      <td>21.0</td>
      <td>72.0</td>
      <td>18.0</td>
      <td>73.0</td>
      <td>34.0</td>
      <td>12.0</td>
      <td>...</td>
      <td>29.0</td>
      <td>58.0</td>
      <td>59.0</td>
      <td>74.0</td>
      <td>57.0</td>
      <td>60.0</td>
      <td>100.0</td>
      <td>96.0</td>
      <td>96.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>2017-11-29 22:00:00</th>
      <td>87.0</td>
      <td>71.0</td>
      <td>22.0</td>
      <td>93.0</td>
      <td>23.0</td>
      <td>68.0</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>32.0</td>
      <td>15.0</td>
      <td>...</td>
      <td>31.0</td>
      <td>58.0</td>
      <td>66.0</td>
      <td>74.0</td>
      <td>57.0</td>
      <td>60.0</td>
      <td>100.0</td>
      <td>96.0</td>
      <td>96.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>2017-11-29 23:00:00</th>
      <td>87.0</td>
      <td>71.0</td>
      <td>22.0</td>
      <td>87.0</td>
      <td>14.0</td>
      <td>63.0</td>
      <td>17.0</td>
      <td>33.0</td>
      <td>30.0</td>
      <td>28.0</td>
      <td>...</td>
      <td>26.0</td>
      <td>58.0</td>
      <td>58.0</td>
      <td>56.0</td>
      <td>57.0</td>
      <td>60.0</td>
      <td>100.0</td>
      <td>96.0</td>
      <td>96.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>2017-11-30 00:00:00</th>
      <td>87.0</td>
      <td>76.0</td>
      <td>22.0</td>
      <td>75.0</td>
      <td>56.0</td>
      <td>72.0</td>
      <td>17.0</td>
      <td>23.0</td>
      <td>34.0</td>
      <td>31.0</td>
      <td>...</td>
      <td>32.0</td>
      <td>58.0</td>
      <td>58.0</td>
      <td>56.0</td>
      <td>57.0</td>
      <td>60.0</td>
      <td>100.0</td>
      <td>96.0</td>
      <td>96.0</td>
      <td>60.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
humidity = humidity.fillna(method='bfill')

```

    C:\Users\yenpth8\AppData\Local\Temp\ipykernel_14780\3131857391.py:1: FutureWarning:
    
    DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
    
    

- Google stocks data doesn't have any missing values but humidity data does have its fair share of missing values. It is cleaned using fillna() method with ffill parameter which propagates last valid observation to fill gaps.

## 2.2.1. The relationship of Observations at different time lags

- The relationship of Observations at different time lags, having 3 concepts:
    - AutoCorrelation
    - Partial Autocorrelation
    - Inverse Autocorrelation

| Thuộc tính           | ACF                    | PACF                                      | Inverse ACF                            |
| -------------------- | ---------------------- | ----------------------------------------- | -------------------------------------- |
| Khái niệm            | Autocorrelation đo lường **mức độ tương quan tuyến tính** giữa chuỗi ( Y_t ) và chính nó tại các độ trễ ( Y_{t-k} ).   | PACF đo **tương quan giữa ( Y_t ) và ( Y_{t-k} )** **sau khi loại bỏ ảnh hưởng trung gian** của các lag nhỏ hơn (1 đến k−1). | Inverse ACF là **hệ số tương quan còn lại sau khi đã lọc hết cấu trúc AR và MA** trong chuỗi. Nói cách khác, nó đo **độ “ngẫu nhiên” còn lại** trong phần dư (residual). |
| Mối quan hệ đo lường | Tương quan tổng        | Tương quan trực tiếp (loại bỏ trung gian) | Tương quan còn sót sau khi mô hình hóa |
| Dùng để xác định     | Bậc **MA(q)**          | Bậc **AR(p)**                             | Kiểm tra model có còn autocorrelation  |
| Giai đoạn sử dụng    | Trước khi modeling     | Trước khi modeling                        | Sau khi modeling                       |
| Biểu đồ kỳ vọng      | Giảm dần hoặc dao động | Cắt đột ngột                              | Gần như 0 (nếu model tốt)              |

### AutoCorrelation

- Careful note:
    - Autocorrelation can't present if having any null data/missing data in dataset.


```python
# Autocorrelation of humidity of San Diego
plot_acf(humidity['San Diego'], lags=25, title = 'San Diego', color = palette[2])
plt.show()
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_130_0.png)
    


- As all lags are either close to 1 or at least greater than the confidence interval, they are statistically significant.
    - It checks how much today’s humidity is related to the humidity of previous days (1 day ago, 2 days ago, etc.).
    - Every lag (past day) still has a strong connection with the current day’s humidity.
    - In other words, humidity levels don’t change randomly — they follow a persistent and predictable pattern over time.

- `plot_acf` is a function for visualizing autocorrelation in time series data.   It is part of the `statsmodels` library and generates a plot showing the correlation of a time series with its own lagged values.
    - `plot_acf` is not defined directly in the provided file, it is imported and used from the `statsmodels.graphics.tsaplots` module.
    - Params:
        - `x`: Array-like, the time series data to analyze.
        - `lags`: Integer, number of lags to display on the x-axis (e.g., 25).
        - `title`: String, optional title for the plot.
    - Side effects:
        - Generates a matplotlib figure displaying autocorrelation values at each lag.
        - Requires `plt.show()` to render the plot in interactive environments.
    - Returns:
        - A matplotlib `Figure` object (not explicitly returned but created implicitly).

### Partial AutoCorrelation


```python
# Partial Autocorrelation of humidity of San Diego
plot_pacf(humidity['San Diego'],lags=25, color = palette[2])
plt.show()
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_134_0.png)
    


- Though it is statistically signficant, partial autocorrelation after first 2 lags is very low.


```python
# Partial Autocorrelation of closing price of microsoft stocks
plot_pacf(microsoft_stock["Close"],lags=25, color = palette[3])
plt.show()
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_136_0.png)
    


- Here, only 0th, 1st and 20th lag are statistically significant.
    - It measures how much today’s price is related to previous prices, after removing the indirect effects from the other days in between.
    - 0th lag → today’s value itself (always = 1, no meaning).
    - 1st lag → yesterday’s closing price has a real, direct influence on today’s price.
    - 20th lag → the price from about 20 trading days ago (≈ one month) also has a smaller but still statistically meaningful effect.
    - All other lags (2–19, 21–25) show no significant relationship, meaning those past days don’t add extra predictive power once yesterday and ~1-month-ago prices are known.

# 3. Feature Engineering
## 3.1. Feature Selection

## 3.2. Feature Transformation
### 3.2.1.Time series Transformation [Framework]


#### Timestamp


```python
pd.Timestamp(2017, 1, 1, 12)
```




    Timestamp('2017-01-01 12:00:00')



- **What are timestamps and how are they useful?**
    - Timestamps are used to represent a point in time.

- The `Timestamp` symbol in this context refers to pd.Timestamp, a constructor from the pandas library used to create a timestamp object representing a specific date and time. The definition is observed in a Jupyter Notebook file where it is called with year, month, day, and hour components.
    - Params: Year (int), Month (int), Day (int), Hour (int) — positional arguments to define the datetime
    - Returns: A pandas.Timestamp object representing the specified moment in time
    - Side effects: None — it is an immutable datetime object
    - Usage context: Typically used to define anchor points in time for filtering, indexing, or generating time series data


```python
# Converting period to timestamp
pd.Period('2017-01-01').to_timestamp(freq='h', how='start')
```




    Timestamp('2017-01-01 00:00:00')



#### Period

- **What are periods and how are they useful?**
    - Periods represent an interval in time. Periods can used to check if a specific event in the given period.


```python
pd.Period('2017-01-01')
```




    Period('2017-01-01', 'D')




```python
# Checking if the given timestamp exists in the given period
pd.Period('2017-01-01').start_time < pd.Timestamp(2017, 1, 1, 12) < pd.Period('2017-01-01').end_time 
```




    True




```python
# Converting timestamp to period
pd.Timestamp(2017, 1, 1, 12).to_period(freq='h')
```




    Period('2017-01-01 12:00', 'h')



#### Data range

- **What is `date_range` and how is it useful?**
    - `date_range` is a method that returns a fixed frequency datetimeindex. It is quite useful when creating your own time series attribute for pre-existing data or arranging the whole data around the time series attribute created by you.


```python
# Creating a datetimeindex with daily frequency
pd.date_range(start='2006-01-03', end='2017-12-29')
```




    DatetimeIndex(['2006-01-03', '2006-01-04', '2006-01-05', '2006-01-06',
                   '2006-01-07', '2006-01-08', '2006-01-09', '2006-01-10',
                   '2006-01-11', '2006-01-12',
                   ...
                   '2017-12-20', '2017-12-21', '2017-12-22', '2017-12-23',
                   '2017-12-24', '2017-12-25', '2017-12-26', '2017-12-27',
                   '2017-12-28', '2017-12-29'],
                  dtype='datetime64[ns]', length=4379, freq='D')




```python
# Creating a datetimeindex with daily frequency
pd.date_range(start='1/1/18', end='1/9/18')
```




    DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
                   '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08',
                   '2018-01-09'],
                  dtype='datetime64[ns]', freq='D')




```python
# Creating a datetimeindex with monthly frequency
pd.date_range(start='1/1/18', end='1/1/19', freq='ME')
```




    DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31', '2018-04-30',
                   '2018-05-31', '2018-06-30', '2018-07-31', '2018-08-31',
                   '2018-09-30', '2018-10-31', '2018-11-30', '2018-12-31'],
                  dtype='datetime64[ns]', freq='ME')




```python
# Creating a datetimeindex without specifying start date and using periods
pd.date_range(end='1/4/2014', periods=8)
```




    DatetimeIndex(['2013-12-28', '2013-12-29', '2013-12-30', '2013-12-31',
                   '2014-01-01', '2014-01-02', '2014-01-03', '2014-01-04'],
                  dtype='datetime64[ns]', freq='D')




```python
# Creating a datetimeindex specifying start date , end date and periods
pd.date_range(start='2013-04-24', end='2014-11-27', periods=3)
```




    DatetimeIndex(['2013-04-24', '2014-02-09', '2014-11-27'], dtype='datetime64[ns]', freq=None)



#### Data time

- `pandas.to_datetime()` is used for converting arguments to datetime. 


```python
# DataFrame is converted to a datetime series.
df = pd.DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5]})
pd.to_datetime(df)
```




    0   2015-02-04
    1   2016-03-05
    dtype: datetime64[ns]




```python
# String to datetime
pd.to_datetime('01-01-2017')
```




    Timestamp('2017-01-01 00:00:00')




```python
from datetime import date, timedelta
date(2020, 10, 1) + timedelta(days=-1)
```




    datetime.date(2020, 9, 30)



#### Shifting & Lags

- We can shift index by desired number of periods with an optional time frequency. This is useful when comparing the time series with a past of itself



```python
humidity["Vancouver"].asfreq('ME').plot(legend=True)

```




    <Axes: xlabel='datetime'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_159_1.png)
    



```python
humidity["Vancouver"].asfreq('D').plot(legend=True)

```




    <Axes: xlabel='datetime'>




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_160_1.png)
    



```python
# Converts it to monthly-end frequency
# Shifts the data forward by 10 periods (months)
humidity["Vancouver"].asfreq('ME').plot(legend=True)
shifted = humidity["Vancouver"].asfreq('ME').shift(10).plot(legend=True)
shifted.legend(['Vancouver','Vancouver_lagged'])
plt.show()

```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_161_0.png)
    



```python
# Converts it to monthly-end frequency
# Shifts the data forward by 1 year
humidity["Vancouver"].asfreq('ME').plot(legend=True)
shifted = humidity["Vancouver"].asfreq('ME').shift(12).plot(legend=True)
shifted.legend(['Vancouver','Vancouver_lagged'])
plt.show()
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_162_0.png)
    


- `shift` is a method used to time-shift data in a time series, commonly for lagging or leading values. It is typically applied to pandas Series or DataFrame objects in time series analysis to align observations with past or future points in time.

- The `shift` method delays or advances the index of a time series by a specified number of periods. 
    - A positive integer, it shifts data forward in time (creating lags), 
    - A negative integer, it shifts data backward (creating leads). 
    - This operation is essential in feature engineering for time series forecasting, where past values are used to predict future ones.

#### Resampling

**Upsampling** - Time series is resampled from low frequency to high frequency(Monthly to daily frequency). It involves filling or interpolating missing data

**Downsampling** - Time series is resampled from high frequency to low frequency(Weekly to monthly frequency). It involves aggregation of existing data.


```python
# Let's use pressure data to demonstrate this
pressure = pd.read_csv('historical-hourly-weather-data\pressure.csv', index_col='datetime', parse_dates=['datetime'])
pressure.tail()
```

    <>:2: SyntaxWarning:
    
    invalid escape sequence '\p'
    
    <>:2: SyntaxWarning:
    
    invalid escape sequence '\p'
    
    C:\Users\yenpth8\AppData\Local\Temp\ipykernel_14780\3020657681.py:2: SyntaxWarning:
    
    invalid escape sequence '\p'
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Vancouver</th>
      <th>Portland</th>
      <th>San Francisco</th>
      <th>Seattle</th>
      <th>Los Angeles</th>
      <th>San Diego</th>
      <th>Las Vegas</th>
      <th>Phoenix</th>
      <th>Albuquerque</th>
      <th>Denver</th>
      <th>...</th>
      <th>Philadelphia</th>
      <th>New York</th>
      <th>Montreal</th>
      <th>Boston</th>
      <th>Beersheba</th>
      <th>Tel Aviv District</th>
      <th>Eilat</th>
      <th>Haifa</th>
      <th>Nahariyya</th>
      <th>Jerusalem</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-11-29 20:00:00</th>
      <td>NaN</td>
      <td>1031.0</td>
      <td>NaN</td>
      <td>1030.0</td>
      <td>1016.0</td>
      <td>1017.0</td>
      <td>1021.0</td>
      <td>1018.0</td>
      <td>1025.0</td>
      <td>1016.0</td>
      <td>...</td>
      <td>1021.0</td>
      <td>NaN</td>
      <td>1021.0</td>
      <td>1017.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-29 21:00:00</th>
      <td>NaN</td>
      <td>1030.0</td>
      <td>NaN</td>
      <td>1030.0</td>
      <td>1016.0</td>
      <td>1017.0</td>
      <td>1020.0</td>
      <td>1018.0</td>
      <td>1024.0</td>
      <td>1018.0</td>
      <td>...</td>
      <td>1021.0</td>
      <td>NaN</td>
      <td>1023.0</td>
      <td>1019.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-29 22:00:00</th>
      <td>NaN</td>
      <td>1030.0</td>
      <td>NaN</td>
      <td>1029.0</td>
      <td>1015.0</td>
      <td>1016.0</td>
      <td>1020.0</td>
      <td>1017.0</td>
      <td>1024.0</td>
      <td>1018.0</td>
      <td>...</td>
      <td>1022.0</td>
      <td>NaN</td>
      <td>1024.0</td>
      <td>1019.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-29 23:00:00</th>
      <td>NaN</td>
      <td>1029.0</td>
      <td>NaN</td>
      <td>1028.0</td>
      <td>1016.0</td>
      <td>1016.0</td>
      <td>1020.0</td>
      <td>1016.0</td>
      <td>1024.0</td>
      <td>1020.0</td>
      <td>...</td>
      <td>1023.0</td>
      <td>NaN</td>
      <td>1026.0</td>
      <td>1022.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-30 00:00:00</th>
      <td>NaN</td>
      <td>1029.0</td>
      <td>NaN</td>
      <td>1028.0</td>
      <td>1015.0</td>
      <td>1017.0</td>
      <td>1019.0</td>
      <td>1016.0</td>
      <td>1024.0</td>
      <td>1021.0</td>
      <td>...</td>
      <td>1024.0</td>
      <td>NaN</td>
      <td>1027.0</td>
      <td>1023.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>



- `ffill` parameter which propagates last valid observation to fill gaps. 
- `bfill` to propogate next valid observation to fill gaps.


```python
pressure = pressure.fillna(method='ffill')
```

    C:\Users\yenpth8\AppData\Local\Temp\ipykernel_14780\99596548.py:1: FutureWarning:
    
    DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
    
    


```python
pressure = pressure.ffill()
```


```python
pressure = pressure.bfill()
```


```python
# Shape before resampling(downsampling)
pressure.shape
```




    (45253, 36)




```python
# Downsample from hourly to 3 day frequency aggregated using mean
pressure = pressure.resample('3D').mean()
pressure.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Vancouver</th>
      <th>Portland</th>
      <th>San Francisco</th>
      <th>Seattle</th>
      <th>Los Angeles</th>
      <th>San Diego</th>
      <th>Las Vegas</th>
      <th>Phoenix</th>
      <th>Albuquerque</th>
      <th>Denver</th>
      <th>...</th>
      <th>Philadelphia</th>
      <th>New York</th>
      <th>Montreal</th>
      <th>Boston</th>
      <th>Beersheba</th>
      <th>Tel Aviv District</th>
      <th>Eilat</th>
      <th>Haifa</th>
      <th>Nahariyya</th>
      <th>Jerusalem</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-10-01</th>
      <td>929.550000</td>
      <td>1022.666667</td>
      <td>1010.850000</td>
      <td>1031.200000</td>
      <td>1011.650000</td>
      <td>1011.983333</td>
      <td>1016.350000</td>
      <td>1012.100000</td>
      <td>1022.566667</td>
      <td>1024.183333</td>
      <td>...</td>
      <td>1014.150000</td>
      <td>1013.400000</td>
      <td>938.683333</td>
      <td>1013.683333</td>
      <td>985.033333</td>
      <td>1012.933333</td>
      <td>1011.783333</td>
      <td>1013.000000</td>
      <td>1013.000000</td>
      <td>990.516667</td>
    </tr>
    <tr>
      <th>2012-10-04</th>
      <td>1019.083333</td>
      <td>1023.041667</td>
      <td>1014.694444</td>
      <td>1028.305556</td>
      <td>1015.555556</td>
      <td>1016.277778</td>
      <td>1013.194444</td>
      <td>1014.097222</td>
      <td>1019.972222</td>
      <td>1020.666667</td>
      <td>...</td>
      <td>1018.097222</td>
      <td>1017.680556</td>
      <td>1017.180556</td>
      <td>1019.805556</td>
      <td>984.930556</td>
      <td>1013.083333</td>
      <td>1012.611111</td>
      <td>1013.000000</td>
      <td>1013.000000</td>
      <td>990.083333</td>
    </tr>
    <tr>
      <th>2012-10-07</th>
      <td>1013.930556</td>
      <td>1017.444444</td>
      <td>1016.597222</td>
      <td>1018.736111</td>
      <td>1013.416667</td>
      <td>1014.222222</td>
      <td>1012.888889</td>
      <td>1011.861111</td>
      <td>1005.833333</td>
      <td>1020.458333</td>
      <td>...</td>
      <td>1017.958333</td>
      <td>1016.750000</td>
      <td>1014.152778</td>
      <td>1016.305556</td>
      <td>982.972222</td>
      <td>1013.027778</td>
      <td>1007.555556</td>
      <td>1013.000000</td>
      <td>1013.000000</td>
      <td>989.833333</td>
    </tr>
    <tr>
      <th>2012-10-10</th>
      <td>1015.000000</td>
      <td>1015.430556</td>
      <td>1014.833333</td>
      <td>1018.416667</td>
      <td>1010.694444</td>
      <td>1014.013889</td>
      <td>1000.166667</td>
      <td>1005.611111</td>
      <td>986.000000</td>
      <td>984.486111</td>
      <td>...</td>
      <td>1018.694444</td>
      <td>1017.916667</td>
      <td>1016.166667</td>
      <td>1017.319444</td>
      <td>979.763889</td>
      <td>1006.527778</td>
      <td>998.763889</td>
      <td>1012.333333</td>
      <td>1012.333333</td>
      <td>987.888889</td>
    </tr>
    <tr>
      <th>2012-10-13</th>
      <td>1008.152778</td>
      <td>1018.111111</td>
      <td>1021.069444</td>
      <td>1015.930556</td>
      <td>1017.277778</td>
      <td>1018.375000</td>
      <td>1015.666667</td>
      <td>1015.500000</td>
      <td>1013.625000</td>
      <td>1010.444444</td>
      <td>...</td>
      <td>1025.055556</td>
      <td>1024.388889</td>
      <td>1020.805556</td>
      <td>1023.736111</td>
      <td>984.527778</td>
      <td>1013.027778</td>
      <td>1007.194444</td>
      <td>1013.000000</td>
      <td>1013.000000</td>
      <td>990.430556</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
# Shape after resampling(downsampling)
pressure.shape
```




    (629, 36)



- Much less rows are left.


```python
# Upsample from 3 day frequency to daily frequency
pressure = pressure.resample('D').fillna(method='pad')
pressure.head()
```

    C:\Users\yenpth8\AppData\Local\Temp\ipykernel_14780\589251129.py:2: FutureWarning:
    
    DatetimeIndexResampler.fillna is deprecated and will be removed in a future version. Use obj.ffill(), obj.bfill(), or obj.nearest() instead.
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Vancouver</th>
      <th>Portland</th>
      <th>San Francisco</th>
      <th>Seattle</th>
      <th>Los Angeles</th>
      <th>San Diego</th>
      <th>Las Vegas</th>
      <th>Phoenix</th>
      <th>Albuquerque</th>
      <th>Denver</th>
      <th>...</th>
      <th>Philadelphia</th>
      <th>New York</th>
      <th>Montreal</th>
      <th>Boston</th>
      <th>Beersheba</th>
      <th>Tel Aviv District</th>
      <th>Eilat</th>
      <th>Haifa</th>
      <th>Nahariyya</th>
      <th>Jerusalem</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-10-01</th>
      <td>929.550000</td>
      <td>1022.666667</td>
      <td>1010.850000</td>
      <td>1031.200000</td>
      <td>1011.650000</td>
      <td>1011.983333</td>
      <td>1016.350000</td>
      <td>1012.100000</td>
      <td>1022.566667</td>
      <td>1024.183333</td>
      <td>...</td>
      <td>1014.150000</td>
      <td>1013.400000</td>
      <td>938.683333</td>
      <td>1013.683333</td>
      <td>985.033333</td>
      <td>1012.933333</td>
      <td>1011.783333</td>
      <td>1013.0</td>
      <td>1013.0</td>
      <td>990.516667</td>
    </tr>
    <tr>
      <th>2012-10-02</th>
      <td>929.550000</td>
      <td>1022.666667</td>
      <td>1010.850000</td>
      <td>1031.200000</td>
      <td>1011.650000</td>
      <td>1011.983333</td>
      <td>1016.350000</td>
      <td>1012.100000</td>
      <td>1022.566667</td>
      <td>1024.183333</td>
      <td>...</td>
      <td>1014.150000</td>
      <td>1013.400000</td>
      <td>938.683333</td>
      <td>1013.683333</td>
      <td>985.033333</td>
      <td>1012.933333</td>
      <td>1011.783333</td>
      <td>1013.0</td>
      <td>1013.0</td>
      <td>990.516667</td>
    </tr>
    <tr>
      <th>2012-10-03</th>
      <td>929.550000</td>
      <td>1022.666667</td>
      <td>1010.850000</td>
      <td>1031.200000</td>
      <td>1011.650000</td>
      <td>1011.983333</td>
      <td>1016.350000</td>
      <td>1012.100000</td>
      <td>1022.566667</td>
      <td>1024.183333</td>
      <td>...</td>
      <td>1014.150000</td>
      <td>1013.400000</td>
      <td>938.683333</td>
      <td>1013.683333</td>
      <td>985.033333</td>
      <td>1012.933333</td>
      <td>1011.783333</td>
      <td>1013.0</td>
      <td>1013.0</td>
      <td>990.516667</td>
    </tr>
    <tr>
      <th>2012-10-04</th>
      <td>1019.083333</td>
      <td>1023.041667</td>
      <td>1014.694444</td>
      <td>1028.305556</td>
      <td>1015.555556</td>
      <td>1016.277778</td>
      <td>1013.194444</td>
      <td>1014.097222</td>
      <td>1019.972222</td>
      <td>1020.666667</td>
      <td>...</td>
      <td>1018.097222</td>
      <td>1017.680556</td>
      <td>1017.180556</td>
      <td>1019.805556</td>
      <td>984.930556</td>
      <td>1013.083333</td>
      <td>1012.611111</td>
      <td>1013.0</td>
      <td>1013.0</td>
      <td>990.083333</td>
    </tr>
    <tr>
      <th>2012-10-05</th>
      <td>1019.083333</td>
      <td>1023.041667</td>
      <td>1014.694444</td>
      <td>1028.305556</td>
      <td>1015.555556</td>
      <td>1016.277778</td>
      <td>1013.194444</td>
      <td>1014.097222</td>
      <td>1019.972222</td>
      <td>1020.666667</td>
      <td>...</td>
      <td>1018.097222</td>
      <td>1017.680556</td>
      <td>1017.180556</td>
      <td>1019.805556</td>
      <td>984.930556</td>
      <td>1013.083333</td>
      <td>1012.611111</td>
      <td>1013.0</td>
      <td>1013.0</td>
      <td>990.083333</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
# Upsample from 3 day frequency to daily frequency
pressure = pressure.resample('D').nearest()
pressure.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Vancouver</th>
      <th>Portland</th>
      <th>San Francisco</th>
      <th>Seattle</th>
      <th>Los Angeles</th>
      <th>San Diego</th>
      <th>Las Vegas</th>
      <th>Phoenix</th>
      <th>Albuquerque</th>
      <th>Denver</th>
      <th>...</th>
      <th>Philadelphia</th>
      <th>New York</th>
      <th>Montreal</th>
      <th>Boston</th>
      <th>Beersheba</th>
      <th>Tel Aviv District</th>
      <th>Eilat</th>
      <th>Haifa</th>
      <th>Nahariyya</th>
      <th>Jerusalem</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-10-01</th>
      <td>929.550000</td>
      <td>1022.666667</td>
      <td>1010.850000</td>
      <td>1031.200000</td>
      <td>1011.650000</td>
      <td>1011.983333</td>
      <td>1016.350000</td>
      <td>1012.100000</td>
      <td>1022.566667</td>
      <td>1024.183333</td>
      <td>...</td>
      <td>1014.150000</td>
      <td>1013.400000</td>
      <td>938.683333</td>
      <td>1013.683333</td>
      <td>985.033333</td>
      <td>1012.933333</td>
      <td>1011.783333</td>
      <td>1013.0</td>
      <td>1013.0</td>
      <td>990.516667</td>
    </tr>
    <tr>
      <th>2012-10-02</th>
      <td>929.550000</td>
      <td>1022.666667</td>
      <td>1010.850000</td>
      <td>1031.200000</td>
      <td>1011.650000</td>
      <td>1011.983333</td>
      <td>1016.350000</td>
      <td>1012.100000</td>
      <td>1022.566667</td>
      <td>1024.183333</td>
      <td>...</td>
      <td>1014.150000</td>
      <td>1013.400000</td>
      <td>938.683333</td>
      <td>1013.683333</td>
      <td>985.033333</td>
      <td>1012.933333</td>
      <td>1011.783333</td>
      <td>1013.0</td>
      <td>1013.0</td>
      <td>990.516667</td>
    </tr>
    <tr>
      <th>2012-10-03</th>
      <td>929.550000</td>
      <td>1022.666667</td>
      <td>1010.850000</td>
      <td>1031.200000</td>
      <td>1011.650000</td>
      <td>1011.983333</td>
      <td>1016.350000</td>
      <td>1012.100000</td>
      <td>1022.566667</td>
      <td>1024.183333</td>
      <td>...</td>
      <td>1014.150000</td>
      <td>1013.400000</td>
      <td>938.683333</td>
      <td>1013.683333</td>
      <td>985.033333</td>
      <td>1012.933333</td>
      <td>1011.783333</td>
      <td>1013.0</td>
      <td>1013.0</td>
      <td>990.516667</td>
    </tr>
    <tr>
      <th>2012-10-04</th>
      <td>1019.083333</td>
      <td>1023.041667</td>
      <td>1014.694444</td>
      <td>1028.305556</td>
      <td>1015.555556</td>
      <td>1016.277778</td>
      <td>1013.194444</td>
      <td>1014.097222</td>
      <td>1019.972222</td>
      <td>1020.666667</td>
      <td>...</td>
      <td>1018.097222</td>
      <td>1017.680556</td>
      <td>1017.180556</td>
      <td>1019.805556</td>
      <td>984.930556</td>
      <td>1013.083333</td>
      <td>1012.611111</td>
      <td>1013.0</td>
      <td>1013.0</td>
      <td>990.083333</td>
    </tr>
    <tr>
      <th>2012-10-05</th>
      <td>1019.083333</td>
      <td>1023.041667</td>
      <td>1014.694444</td>
      <td>1028.305556</td>
      <td>1015.555556</td>
      <td>1016.277778</td>
      <td>1013.194444</td>
      <td>1014.097222</td>
      <td>1019.972222</td>
      <td>1020.666667</td>
      <td>...</td>
      <td>1018.097222</td>
      <td>1017.680556</td>
      <td>1017.180556</td>
      <td>1019.805556</td>
      <td>984.930556</td>
      <td>1013.083333</td>
      <td>1012.611111</td>
      <td>1013.0</td>
      <td>1013.0</td>
      <td>990.083333</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
google_stock.asfreq('D')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Name</th>
      <th>Change</th>
      <th>Seasonal_noise_90D</th>
      <th>seasonal_90D</th>
      <th>Seasonal_noise_7D</th>
      <th>seasonal_7D</th>
      <th>Seasonal_noise_14D</th>
      <th>seasonal_14D</th>
      <th>Seasonal_noise_4D</th>
      <th>seasonal_4D</th>
      <th>Seasonal_noise_Month</th>
      <th>seasonal_Month</th>
      <th>residual</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2006-01-03</th>
      <td>211.47</td>
      <td>218.05</td>
      <td>209.32</td>
      <td>217.83</td>
      <td>13137450.0</td>
      <td>GOOGL</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.458579</td>
      <td>0.000000</td>
      <td>0.150230</td>
      <td>0.000000</td>
      <td>-0.450857</td>
      <td>0.000000</td>
      <td>0.576742</td>
      <td>0.000000</td>
      <td>-1.404333</td>
      <td>-0.458579</td>
    </tr>
    <tr>
      <th>2006-01-04</th>
      <td>222.17</td>
      <td>224.70</td>
      <td>220.09</td>
      <td>222.84</td>
      <td>15292353.0</td>
      <td>GOOGL</td>
      <td>1.030498</td>
      <td>3.325000</td>
      <td>0.458579</td>
      <td>3.325000</td>
      <td>0.150230</td>
      <td>3.325000</td>
      <td>-0.450857</td>
      <td>3.325000</td>
      <td>0.576742</td>
      <td>3.325000</td>
      <td>-1.404333</td>
      <td>2.866421</td>
    </tr>
    <tr>
      <th>2006-01-05</th>
      <td>223.22</td>
      <td>226.00</td>
      <td>220.97</td>
      <td>225.85</td>
      <td>10815661.0</td>
      <td>GOOGL</td>
      <td>1.005785</td>
      <td>3.083333</td>
      <td>0.458579</td>
      <td>3.083333</td>
      <td>0.150230</td>
      <td>3.083333</td>
      <td>-0.450857</td>
      <td>3.083333</td>
      <td>-0.523652</td>
      <td>3.083333</td>
      <td>-1.404333</td>
      <td>2.624755</td>
    </tr>
    <tr>
      <th>2006-01-06</th>
      <td>228.66</td>
      <td>235.49</td>
      <td>226.85</td>
      <td>233.06</td>
      <td>17759521.0</td>
      <td>GOOGL</td>
      <td>1.041991</td>
      <td>9.430000</td>
      <td>0.458579</td>
      <td>9.430000</td>
      <td>0.150230</td>
      <td>9.430000</td>
      <td>-0.450857</td>
      <td>9.430000</td>
      <td>-0.523652</td>
      <td>9.430000</td>
      <td>-1.404333</td>
      <td>8.971421</td>
    </tr>
    <tr>
      <th>2006-01-07</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-12-25</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-12-26</th>
      <td>1068.64</td>
      <td>1068.86</td>
      <td>1058.64</td>
      <td>1065.85</td>
      <td>918767.0</td>
      <td>GOOGL</td>
      <td>NaN</td>
      <td>33.054839</td>
      <td>18.129306</td>
      <td>-5.975000</td>
      <td>0.680592</td>
      <td>-5.431111</td>
      <td>0.843998</td>
      <td>0.000000</td>
      <td>0.856250</td>
      <td>7.276667</td>
      <td>4.391946</td>
      <td>14.925533</td>
    </tr>
    <tr>
      <th>2017-12-27</th>
      <td>1066.60</td>
      <td>1068.27</td>
      <td>1058.38</td>
      <td>1060.20</td>
      <td>1116203.0</td>
      <td>GOOGL</td>
      <td>0.999448</td>
      <td>30.818226</td>
      <td>18.129306</td>
      <td>-3.322500</td>
      <td>0.680592</td>
      <td>-7.442222</td>
      <td>0.843998</td>
      <td>-0.295000</td>
      <td>-0.264314</td>
      <td>6.913810</td>
      <td>4.391946</td>
      <td>12.688920</td>
    </tr>
    <tr>
      <th>2017-12-28</th>
      <td>1062.25</td>
      <td>1064.84</td>
      <td>1053.38</td>
      <td>1055.95</td>
      <td>994249.0</td>
      <td>GOOGL</td>
      <td>0.996789</td>
      <td>25.952258</td>
      <td>18.129306</td>
      <td>-3.582500</td>
      <td>0.680592</td>
      <td>-10.623333</td>
      <td>0.843998</td>
      <td>-2.483333</td>
      <td>-0.264314</td>
      <td>4.205714</td>
      <td>4.391946</td>
      <td>7.822952</td>
    </tr>
    <tr>
      <th>2017-12-29</th>
      <td>1055.49</td>
      <td>1058.05</td>
      <td>1052.70</td>
      <td>1053.40</td>
      <td>1180340.0</td>
      <td>GOOGL</td>
      <td>0.993623</td>
      <td>18.858095</td>
      <td>18.129306</td>
      <td>-6.955000</td>
      <td>0.680592</td>
      <td>-15.502222</td>
      <td>0.843998</td>
      <td>-6.955000</td>
      <td>-0.264314</td>
      <td>-2.550000</td>
      <td>4.391946</td>
      <td>0.728789</td>
    </tr>
  </tbody>
</table>
<p>4379 rows × 18 columns</p>
</div>



#### DatatimeIndex

- When transform Index, must keep the length.
    - Don't use date_range for group the period.


```python
# Index of date
google_stock.index.dayofyear
```




    Index([  3,   4,   5,   6,   9,  10,  11,  12,  13,  17,
           ...
           349, 352, 353, 354, 355, 356, 360, 361, 362, 363],
          dtype='int32', name='Date', length=3019)




```python
# Index of date of week
google_stock.index.dayofweek
```




    Index([1, 2, 3, 4, 0, 1, 2, 3, 4, 1,
           ...
           4, 0, 1, 2, 3, 4, 1, 2, 3, 4],
          dtype='int32', name='Date', length=3019)




```python
# Index of month
google_stock.index.month
```




    Index([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
           ...
           12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
          dtype='int32', name='Date', length=3019)




```python
# Index of week
google_stock.index.isocalendar().week
```




    Date
    2006-01-03     1
    2006-01-04     1
    2006-01-05     1
    2006-01-06     1
    2006-01-09     2
                  ..
    2017-12-22    51
    2017-12-26    52
    2017-12-27    52
    2017-12-28    52
    2017-12-29    52
    Name: week, Length: 3019, dtype: UInt32




```python
# Index of 3D
(google_stock.index.dayofweek - 1)//3 + 1
```




    Index([1, 1, 1, 2, 0, 1, 1, 1, 2, 1,
           ...
           2, 0, 1, 1, 1, 2, 1, 1, 1, 2],
          dtype='int32', name='Date', length=3019)



### 3.2.2.Time series Transformation [Application]


```python
class Timeseries_lag_transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lag_dfs = {}
        dataframe = X.copy()
        
        # List of lag days
        list_number_of_day = [1, 7, 30]
        
        for number_of_day in list_number_of_day: 
            # Lag dataframe
            lag_dfs[f'google_stock_{number_of_day}D'] = dataframe.asfreq('D').shift(number_of_day)
            
            # Rename column in Lag dataframe
            col_new_list = []
            for col in dataframe.columns:
                col_new = f'{col}_{number_of_day}D'
                col_new_list.append(col_new)
            lag_dfs[f'google_stock_{number_of_day}D'].columns = col_new_list
        
        # List of lag dataframe's name
        lag_dfs_list = list(lag_dfs.keys())

        # Concat lag dataframe to original dataframe
        for i in range(len(lag_dfs_list)):
            dataframe = pd.concat([dataframe, lag_dfs[lag_dfs_list[i]]], axis=1)  
        return dataframe
```


```python
class Timeseries_period_transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dataframe = X.copy()
        dataframe['date'] = dataframe.index
        dataframe['period_day'] = dataframe.index.to_period(freq='D')
        dataframe['period_month'] = dataframe.index.to_period(freq='M')
        dataframe['period_year'] = dataframe.index.to_period(freq='Y')
        return dataframe

```


```python
numertical_features = google_stock.select_dtypes(include = ['float64', 'int64']).columns

# List of lag days
list_number_of_day = [1, 7, 30]

col_new_list = []
for day in list_number_of_day:
    for col in numertical_features:
        col_new = f'{col}_{day}D'
        col_new_list.append(col_new)

numertical_features = numertical_features.append(pd.Index(col_new_list))
numertical_features
```




    Index(['Open', 'High', 'Low', 'Close', 'Volume', 'Change',
           'Seasonal_noise_90D', 'seasonal_90D', 'Seasonal_noise_7D',
           'seasonal_7D', 'Seasonal_noise_14D', 'seasonal_14D',
           'Seasonal_noise_4D', 'seasonal_4D', 'Seasonal_noise_Month',
           'seasonal_Month', 'residual', 'Open_1D', 'High_1D', 'Low_1D',
           'Close_1D', 'Volume_1D', 'Change_1D', 'Seasonal_noise_90D_1D',
           'seasonal_90D_1D', 'Seasonal_noise_7D_1D', 'seasonal_7D_1D',
           'Seasonal_noise_14D_1D', 'seasonal_14D_1D', 'Seasonal_noise_4D_1D',
           'seasonal_4D_1D', 'Seasonal_noise_Month_1D', 'seasonal_Month_1D',
           'residual_1D', 'Open_7D', 'High_7D', 'Low_7D', 'Close_7D', 'Volume_7D',
           'Change_7D', 'Seasonal_noise_90D_7D', 'seasonal_90D_7D',
           'Seasonal_noise_7D_7D', 'seasonal_7D_7D', 'Seasonal_noise_14D_7D',
           'seasonal_14D_7D', 'Seasonal_noise_4D_7D', 'seasonal_4D_7D',
           'Seasonal_noise_Month_7D', 'seasonal_Month_7D', 'residual_7D',
           'Open_30D', 'High_30D', 'Low_30D', 'Close_30D', 'Volume_30D',
           'Change_30D', 'Seasonal_noise_90D_30D', 'seasonal_90D_30D',
           'Seasonal_noise_7D_30D', 'seasonal_7D_30D', 'Seasonal_noise_14D_30D',
           'seasonal_14D_30D', 'Seasonal_noise_4D_30D', 'seasonal_4D_30D',
           'Seasonal_noise_Month_30D', 'seasonal_Month_30D', 'residual_30D'],
          dtype='object')




```python
column_preprocessing = ColumnTransformer(transformers = [
    ('numerical', Pipeline(steps = [
        ('scaler', MinMaxScaler())
        ]), numertical_features)
], remainder = 'passthrough')
```


```python
preprocessing_pipeline = Pipeline(steps = [
    ('timeseries_period', Timeseries_period_transformer()),
    ('timeseries_lag', Timeseries_lag_transformer()),
    ('column_preprocessing', column_preprocessing)
])
```


```python
preprocessing_pipeline
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.estimator-table summary {
    padding: .5rem;
    font-family: monospace;
    cursor: pointer;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover {
    background-color: #e0e0e0;
}

.estimator-table table td {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left;
}

.user-set td.value pre {
    color:rgb(255, 94, 0) !important;
    background-color: transparent !important;
}

.default td {
    color: black;
    text-align: left;
}

.user-set td i,
.default td i {
    color: black;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}
</style><body><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;timeseries_period&#x27;, Timeseries_period_transformer()),
                (&#x27;timeseries_lag&#x27;, Timeseries_lag_transformer()),
                (&#x27;column_preprocessing&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;numerical&#x27;,
                                                  Pipeline(steps=[(&#x27;scaler&#x27;,
                                                                   MinMaxScaler())]),
                                                  Index([&#x27;Open&#x27;, &#x27;High&#x27;, &#x27;Low&#x27;, &#x27;Close&#x27;, &#x27;Volume&#x27;, &#x27;Change&#x27;,
       &#x27;Seasonal_noise_90D&#x27;, &#x27;seasonal_90D&#x27;, &#x27;Seaso...
       &#x27;Seasonal_noise_Month_7D&#x27;, &#x27;seasonal_Month_7D&#x27;, &#x27;residual_7D&#x27;,
       &#x27;Open_30D&#x27;, &#x27;High_30D&#x27;, &#x27;Low_30D&#x27;, &#x27;Close_30D&#x27;, &#x27;Volume_30D&#x27;,
       &#x27;Change_30D&#x27;, &#x27;Seasonal_noise_90D_30D&#x27;, &#x27;seasonal_90D_30D&#x27;,
       &#x27;Seasonal_noise_7D_30D&#x27;, &#x27;seasonal_7D_30D&#x27;, &#x27;Seasonal_noise_14D_30D&#x27;,
       &#x27;seasonal_14D_30D&#x27;, &#x27;Seasonal_noise_4D_30D&#x27;, &#x27;seasonal_4D_30D&#x27;,
       &#x27;Seasonal_noise_Month_30D&#x27;, &#x27;seasonal_Month_30D&#x27;, &#x27;residual_30D&#x27;],
      dtype=&#x27;object&#x27;))]))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>Pipeline</div></div><div><a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link ">i<span>Not fitted</span></span></div></label><div class="sk-toggleable__content " data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('steps',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">steps&nbsp;</td>
            <td class="value">[(&#x27;timeseries_period&#x27;, ...), (&#x27;timeseries_lag&#x27;, ...), ...]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('transform_input',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">transform_input&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('memory',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">memory&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">verbose&nbsp;</td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>Timeseries_period_transformer</div></div></label><div class="sk-toggleable__content " data-param-prefix="timeseries_period__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>Timeseries_lag_transformer</div></div></label><div class="sk-toggleable__content " data-param-prefix="timeseries_lag__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>column_preprocessing: ColumnTransformer</div></div><div><a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for column_preprocessing: ColumnTransformer</span></a></div></label><div class="sk-toggleable__content " data-param-prefix="column_preprocessing__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('transformers',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">transformers&nbsp;</td>
            <td class="value">[(&#x27;numerical&#x27;, ...)]</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('remainder',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">remainder&nbsp;</td>
            <td class="value">&#x27;passthrough&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('sparse_threshold',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">sparse_threshold&nbsp;</td>
            <td class="value">0.3</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_jobs',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">n_jobs&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('transformer_weights',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">transformer_weights&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">verbose&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose_feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">verbose_feature_names_out&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('force_int_remainder_cols',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">force_int_remainder_cols&nbsp;</td>
            <td class="value">&#x27;deprecated&#x27;</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>numerical</div></div></label><div class="sk-toggleable__content " data-param-prefix="column_preprocessing__numerical__"><pre>Index([&#x27;Open&#x27;, &#x27;High&#x27;, &#x27;Low&#x27;, &#x27;Close&#x27;, &#x27;Volume&#x27;, &#x27;Change&#x27;,
       &#x27;Seasonal_noise_90D&#x27;, &#x27;seasonal_90D&#x27;, &#x27;Seasonal_noise_7D&#x27;,
       &#x27;seasonal_7D&#x27;, &#x27;Seasonal_noise_14D&#x27;, &#x27;seasonal_14D&#x27;,
       &#x27;Seasonal_noise_4D&#x27;, &#x27;seasonal_4D&#x27;, &#x27;Seasonal_noise_Month&#x27;,
       &#x27;seasonal_Month&#x27;, &#x27;residual&#x27;, &#x27;Open_1D&#x27;, &#x27;High_1D&#x27;, &#x27;Low_1D&#x27;,
       &#x27;Close_1D&#x27;, &#x27;Volume_1D&#x27;, &#x27;Change_1D&#x27;, &#x27;Seasonal_noise_90D_1D&#x27;,
       &#x27;seasonal_90D_1D&#x27;, &#x27;Seasonal_noise_7D_1D&#x27;, &#x27;seasonal_7D_1D&#x27;,
       &#x27;Seasonal_noise_14D_1D&#x27;, &#x27;seasonal_14D_1D&#x27;, &#x27;Seasonal_noise_4D_1D&#x27;,
       &#x27;seasonal_4D_1D&#x27;, &#x27;Seasonal_noise_Month_1D&#x27;, &#x27;seasonal_Month_1D&#x27;,
       &#x27;residual_1D&#x27;, &#x27;Open_7D&#x27;, &#x27;High_7D&#x27;, &#x27;Low_7D&#x27;, &#x27;Close_7D&#x27;, &#x27;Volume_7D&#x27;,
       &#x27;Change_7D&#x27;, &#x27;Seasonal_noise_90D_7D&#x27;, &#x27;seasonal_90D_7D&#x27;,
       &#x27;Seasonal_noise_7D_7D&#x27;, &#x27;seasonal_7D_7D&#x27;, &#x27;Seasonal_noise_14D_7D&#x27;,
       &#x27;seasonal_14D_7D&#x27;, &#x27;Seasonal_noise_4D_7D&#x27;, &#x27;seasonal_4D_7D&#x27;,
       &#x27;Seasonal_noise_Month_7D&#x27;, &#x27;seasonal_Month_7D&#x27;, &#x27;residual_7D&#x27;,
       &#x27;Open_30D&#x27;, &#x27;High_30D&#x27;, &#x27;Low_30D&#x27;, &#x27;Close_30D&#x27;, &#x27;Volume_30D&#x27;,
       &#x27;Change_30D&#x27;, &#x27;Seasonal_noise_90D_30D&#x27;, &#x27;seasonal_90D_30D&#x27;,
       &#x27;Seasonal_noise_7D_30D&#x27;, &#x27;seasonal_7D_30D&#x27;, &#x27;Seasonal_noise_14D_30D&#x27;,
       &#x27;seasonal_14D_30D&#x27;, &#x27;Seasonal_noise_4D_30D&#x27;, &#x27;seasonal_4D_30D&#x27;,
       &#x27;Seasonal_noise_Month_30D&#x27;, &#x27;seasonal_Month_30D&#x27;, &#x27;residual_30D&#x27;],
      dtype=&#x27;object&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>MinMaxScaler</div></div><div><a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.MinMaxScaler.html">?<span>Documentation for MinMaxScaler</span></a></div></label><div class="sk-toggleable__content " data-param-prefix="column_preprocessing__numerical__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('feature_range',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">feature_range&nbsp;</td>
            <td class="value">(0, ...)</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">copy&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('clip',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">clip&nbsp;</td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>remainder</div></div></label><div class="sk-toggleable__content " data-param-prefix="column_preprocessing__remainder__"></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label  sk-toggleable__label-arrow"><div><div>passthrough</div></div></label><div class="sk-toggleable__content " data-param-prefix="column_preprocessing__remainder__"><pre>passthrough</pre></div></div></div></div></div></div></div></div></div></div></div></div><script>function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.fa-regular.fa-copy').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const paramName = element.parentElement.nextElementSibling.textContent.trim();
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});
</script></body>




```python
preprocessing_pipeline.fit_transform(google_stock)
```




    array([[0.08415035255298803, 0.08745678649111563, 0.09019504480759094,
            ..., NaT, NaT, NaT],
           [0.09539421834116199, 0.09444450282135614, 0.101549815498155, ...,
            NaT, NaT, NaT],
           [0.09649758834841271, 0.09581052255508737, 0.10247759620453348,
            ..., NaT, NaT, NaT],
           ...,
           [0.9827453947437552, 0.9808547080395513, 0.9853558249868214, ...,
            Period('2017-11-27', 'D'), Period('2017-11', 'M'),
            Period('2017', 'Y-DEC')],
           [0.9781742904280024, 0.977250517511322, 0.9800843437005801, ...,
            Period('2017-11-28', 'D'), Period('2017-11', 'M'),
            Period('2017', 'Y-DEC')],
           [0.9710706892384644, 0.9701156913636027, 0.9793674222456511, ...,
            Period('2017-11-29', 'D'), Period('2017-11', 'M'),
            Period('2017', 'Y-DEC')]], shape=(4379, 88), dtype=object)




```python
preprocessing_pipeline.fit_transform(google_stock).shape
```




    (4379, 88)




```python

```

## 3.3. Feature Creation


## 3.4. Feature Extraction

# 4. Modelling

> **Bản chất:** Classical linear time-series models (AR, MA, ARMA, ARIMA, SARIMA) thường dự báo dựa trên chính chuỗi lịch sử (lag values) bằng các mối quan hệ tuyến tính, nên chúng có thể hoạt động tốt chỉ với dữ liệu chuỗi. Các mô hình hiện đại (ML/DL, hybrid) không bắt buộc phải dùng features ngoại sinh nhưng thường hưởng lợi mạnh khi được cung cấp nhiều feature (exogenous variables, calendar features, lags, rolling stats) vì chúng có khả năng học tương tác phi tuyến và patterns phức tạp mà mô hình tuyến tính không nắm được.

- **Misunderstanding Point:**
    - **Classical models có thể dùng biến ngoài**
        * ARIMA → có biến ngoại sinh → gọi là **ARIMAX / SARIMAX**.
        * **VAR / State-space** thường dùng nhiều biến đồng thời (multivariate) và có thể model exogenous inputs.
        * Như vậy: *“không cần biến ngoài”* là *không bắt buộc* chứ không phải *không thể*.

    - **ML/DL không bắt buộc phải có biến ngoài**
        * Bạn có thể train XGBoost / LSTM chỉ với lag-features (lag1, lag2, rolling_mean, ...) — mô hình vẫn chạy.
        * Tuy nhiên ML/DL thường **cần nhiều dữ liệu** và *thường vận hành tốt hơn* khi có thêm features (holiday, price, marketing, weather,...).

    - **Điểm khác biệt cốt lõi không chỉ là feature**
        * **Giả định tuyến tính vs phi-tuyến:** Classical giả định tuyến tính; ML/DL học phi-tuyến.
        * **Stationarity:** Classical thường yêu cầu/khuyên stationarity (vì lý thuyết suy luận) — dẫn tới differencing, detrending trước khi modeling. ML/DL không bắt buộc nhưng có thể benefit khi dữ liệu được “ổn định” hơn.
        * **Interpretability vs Predictive Power:** Classical dễ giải thích; ML/DL thường mạnh về predictive performance nhưng khó giải thích.

    - **Bias/Variance & Data size**
        * Classical ít tham số → tốt với dữ liệu ít.
        * ML/DL có nhiều tham số → cần dữ liệu lớn để tránh overfitting; nếu dữ liệu nhỏ, thêm nhiều features có thể khiến mô hình tệ hơn.

    - **Seasonality / Exogenous effects**
        * Nếu chuỗi có seasonality rõ ràng, classical (SARIMA) xử lý trực tiếp; ML/DL có thể học seasonal nếu đủ data hoặc khi thêm calendar features.

    - **Baseline & nguyên tắc thực hành**
        * Luôn so sánh với **baseline đơn giản** (naive, random walk, AR(1), seasonal naive).
        * Thử cả classical và ML/DL — đôi khi ARIMA thắng, đôi khi XGBoost/LSTM thắng, tùy dữ liệu.

- Advise:
    * **Hãy nói rõ điều kiện khi phát biểu:**
    Ví dụ: *“Với dữ liệu ngắn hạn, ít biến, no external covariates → classical models thường đủ. Với nhiều covariates, phi-tuyến interactions → ML/DL có lợi.”*

    * **Luôn kiểm tra stationarity trước khi dùng classical models.** Nếu non-stationary, dùng differencing (=> ARIMA) hoặc model returns/changes.

    * **Feature engineering vẫn quan trọng trong mọi trường hợp.** Dù dùng ARIMA hay LSTM, các features như lags, rolling mean, holiday flags, temperature… thường nâng hiệu năng.

    * **Chọn mô hình theo mục tiêu:**
        * Nếu cần *giải thích nguyên nhân* → ưu tiên classical/state-space.
        * Nếu cần *dự báo tối đa accuracy* với nhiều data → thử ML/DL/hybrid.

    * **Sử dụng hybrid khi hợp lý:** ARIMA để bắt trend/seasonal + LSTM/XGBoost để bắt phần residual phi-tuyến → thường cải thiện performance.

    * **Validation:** dùng time-series cross-validation (rolling / expanding window) thay vì random CV.

## 4.1. [Autoregressive Model](https://www.statsmodels.org/stable/examples/notebooks/generated/autoregressions.html)

- (adj) regressive /rɪˈɡres.ɪv/:  becoming or making something less advanced, returning to a previous and less advanced or worse state or way of behaving. 
    - VN: thoái lui, thụt lùi
- (prefix) auto /ˈɔː.təʊ/: of or by yourself, or operating independently and without needing help 
    - VN: tự
- (adj) autoregressive: tự hồi quy  -> giá trị hiện tại phụ thuộc vào các giá trị trong quá khứ của chính nó. 
    - Dự đoán hôm nay bằng cách nhìn các ngày hôm trước.
- (adj) linear /ˈlɪn.i.ər/: consisting of relating to lines or length.
    - VN: tuyến tính

- An autoregressive (AR) model is a representation of a type of random process; as such, it is used to describe certain time-varying processes in nature, economics, etc. The autoregressive model specifies that the output variable depends linearly on its own previous values and on a stochastic term (an imperfectly predictable term); thus the model is in the form of a stochastic difference equation.
    - AR Overall Model: $Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \ldots + \phi_p Y_{t-p} + \varepsilon_t$
        | Thành phần | Ý nghĩa |
        | --- | --- |
        | $Y_t$ | Giá trị tại thời điểm hiện tại |
        | $c$ | Hằng số (trung bình cố định) |
        | $\phi_1, \phi_2, ...\phi_p$ | Autoregressive coefficients (hệ số tự hồi quy) <=> Trọng số cho các giá trị trong quá khứ. Là hệ số $\phi_i$ thể hiện **mức độ ảnh hưởng của giá trị quá khứ $Y_{t-i}$** lên giá trị hiện tại $Y_t$.|
        | $\varepsilon_t$ | Nhiễu ngẫu nhiên (noise) |

        - Nếu $\phi_1 = 0.8$: giá trị hôm nay chịu ảnh hưởng **80% từ giá trị hôm qua**.
        - Nếu $\phi_1 = 0.1$: ảnh hưởng rất nhỏ, chuỗi thay đổi nhanh.
        - Nếu $\phi_1 = 1$: chuỗi có **tính dừng yếu**, dễ trở thành **random walk** (mất ổn định).
        - Nếu $|\phi_1| < 1$: chuỗi là **stationary** (ổn định quanh giá trị trung bình).

        | Tình huống     | Giá trị hệ số          | Đặc điểm chuỗi                              |
        | -------------- | ---------------------- | ------------------------------------------- |
        | $\phi = 0$     | Không có tự tương quan | Chuỗi ngẫu nhiên (white noise)              |
        | $0 < \phi < 1$ | Ảnh hưởng dương        | Chuỗi có xu hướng “theo đuôi” giá trị trước |
        | $\phi < 0$     | Ảnh hưởng ngược        | Chuỗi có dao động kiểu “zig-zag”            |

    - AR(1) Model: $R_t = \mu + \phi R_{t-1} + \epsilon_t$
        - As RHS has only one lagged value($R_{t-1}$)this is called AR model of order 1 where $\mu$ is mean and $\epsilon$ is noise at time t
        - If $\phi$ = 1, it is random walk. Else if $\phi$ = 0, it is white noise. Else if -1 < $\phi$ < 1, it is stationary. If ϕ is -ve, there is men reversion. If $\phi$ is +ve, there is momentum.
        - Example: AR(1) — chỉ nhìn lại 1 ngày trước: $Sales_t = c + \phi_1 \times Sales_{t-1} + \varepsilon_t$ 
            - Nếu $\phi_1 = 0.8$: → Nghĩa là **80% doanh số hôm nay được quyết định bởi hôm qua**, 20% là nhiễu (bất ngờ, khuyến mãi, v.v.)
    - AR(2) model: $R_t = \mu + R_{t-1} + \phi_2 R_{t-2} + \epsilon_t$
    - AR(3) model: $R_t = \mu + R_{t-1} + \phi_2 R_{t-2} + \phi_3 R_{t-3} + \epsilon_t$

    - Autoregression is the foundation of larger model like $ARIMA = AR + I (Integrated) + MA (Moving Average)$


- Mechanism of Autoregressive Model

    | Bước                                 | Hoạt động                                                                    |
    | ------------------------------------ | ---------------------------------------------------------------------------- |
    | **1. Input dữ liệu**                 | Chuỗi thời gian: giá, doanh thu, sản lượng...                                |
    | **2. Kiểm tra tính dừng (ADF Test)** | Nếu chuỗi không dừng → phải lấy sai phân (differencing).                     |
    | **3. Xác định độ trễ p**             | Dùng ACF/PACF plot hoặc AIC/BIC để chọn số lượng giá trị quá khứ quan trọng. <br> Là số bước thời gian ta quay ngược về quá khứ để xem giá trị trước đó ảnh hưởng đến hiện tại như thế nào. <br>  Là số lượng độ trễ (number of past periods) mà mô hình dùng để dự đoán hiện tại. |
    | **4. Huấn luyện mô hình**            | Ước lượng các hệ số $\phi_1, \phi_2, ...\phi_p$.                             |
    | **5. Dự đoán giá trị tương lai**     | Dựa trên các giá trị trước đó.                                               |



- ACF: Autocorrelation Function
    - ACF (hàm tự tương quan) đo **mức độ liên hệ giữa giá trị hiện tại** và **các giá trị trong quá khứ** (lag 1, lag 2, lag 3,…).
    - $ACF(k) = \text{correlation}(Y_t, Y_{t-k})$ hoặc $AR(p)$ (Autoregressive of order p)
- PACF: Partial Autocorrelation Function
    - PACF đo **mức độ ảnh hưởng trực tiếp** giữa $Y_t$ và $Y_{t-k}$ **sau khi loại bỏ ảnh hưởng của các lag trung gian**.
    - ACF xem “tổng ảnh hưởng” của quá khứ lên hiện tại.
    - PACF chỉ giữ “ảnh hưởng trực tiếp” sau khi bỏ các yếu tố trung gian.
- AIC: Akaike Information Criterion
    - AIC là **chỉ số đánh giá độ phù hợp của mô hình**, có phạt mô hình phức tạp.
    - $AIC = 2k - 2\ln(L)$
        - $k$: số lượng tham số trong mô hình
        - $L$: độ hợp lý (likelihood)
    - Simple Explanation:
        - Mô hình nào **fit tốt** dữ liệu và **càng đơn giản**, thì AIC càng **thấp** → mô hình đó tốt hơn.
    - Usage:
        - Khi bạn thử nhiều mô hình AR(p), MA(q), ARIMA(p,d,q) → chọn mô hình có **AIC nhỏ nhất**.
- BIC: Bayesian Information Criterion
    - Tương tự AIC nhưng **phạt mô hình phức tạp nặng hơn**.
    - $BIC = k \ln(n) - 2\ln(L)$
        - $n$: số lượng quan sát.
    - Simple Explanation: 
        - Khi dữ liệu nhiều, BIC khuyến khích chọn mô hình **càng đơn giản càng tốt**.
    - Usage:
        - Khi bạn muốn mô hình có **khả năng khái quát tốt**, không overfit.
        - Dùng song song với AIC để đối chiếu.


- Why choose the suitable number of past periods is important?
    | Mục tiêu                                   | Giải thích                                                                   |
    | ------------------------------------------ | ---------------------------------------------------------------------------- |
    | **Tìm số giá trị quá khứ quan trọng nhất** | Không phải mọi giá trị trong quá khứ đều ảnh hưởng mạnh đến hiện tại.        |
    | **Giảm overfitting**                       | Nếu chọn p quá lớn, mô hình sẽ phức tạp và khớp quá mức dữ liệu huấn luyện.  |
    | **Giảm underfitting**                      | Nếu chọn p quá nhỏ, mô hình không đủ thông tin quá khứ để dự đoán chính xác. |

- How to choose the suitable number of past periods?
    | Phương pháp          | Mô tả                                                                                                                           | Gợi ý lựa chọn                            |
    | -------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
    | **ACF/PACF plot**    | - **ACF**: tự tương quan toàn phần.<br>- **PACF**: tự tương quan riêng phần.<br>→ Xem bao nhiêu lag vẫn còn tương quan đáng kể. | Số lag mà PACF cắt giảm về 0 thường là p. |
    | **AIC / BIC**        | So sánh độ phù hợp của mô hình với các giá trị p khác nhau.                                                                     | Chọn p mà AIC hoặc BIC nhỏ nhất.          |
    | **Cross-validation** | Đo sai số dự báo với từng p khác nhau.                                                                                          | Chọn p cho sai số nhỏ nhất.               |




```python
# AR(1) MA(1) model:AR parameter = +0.9
rcParams['figure.figsize'] = 16, 12
plt.subplot(4,1,1)
ar1 = np.array([1, -0.9]) # We choose -0.9 as AR parameter is +0.9
ma1 = np.array([1])
AR1 = ArmaProcess(ar1, ma1)
sim1 = AR1.generate_sample(nsample=1000)
plt.title('AR(1) model: AR parameter = +0.9')
plt.plot(sim1)
```




    [<matplotlib.lines.Line2D at 0x2415cd639d0>]




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_202_1.png)
    


- `ArmaProcess` is a class used to represent and simulate ARMA (Autoregressive Moving Average) time series models.
    - It allows the creation of synthetic data from specified autoregressive (AR) and moving average (MA) parameters, commonly used in statistical and econometric analysis.
    - Params:
        - `ar`: Array-like, **autoregressive coefficients**, including the leading 1 (e.g., [1, -φ] for AR(1) with parameter φ).
        - `ma`: Array-like, **moving average coefficients**, including the leading 1 (e.g., [1, θ] for MA(1)).
    - Returns: An ArmaProcess instance that can generate samples or compute theoretical properties (e.g., impulse response, spectral density).
    - Side effects: None — the object is stateless; sample generation is done via method calls.

- (adj) synthetic /sɪnˈθet.ɪk/: Synthetic products are made from artificial substances, often copying a natural product
    - VD: tổng hợp từ các thành phần, tạo thành một bản copy, không làm thay đổi bản gốc.


### 4.1.1. AutoRegressive Model (1) Simulation


```python
# AR(1) MA(1) model:AR parameter = +0.9
rcParams['figure.figsize'] = 16, 12
plt.subplot(4,1,1)
ar1 = np.array([1, -0.9]) # We choose -0.9 as AR parameter is +0.9
ma1 = np.array([1])
AR1 = ArmaProcess(ar1, ma1)
sim1 = AR1.generate_sample(nsample=1000)
plt.title('AR(1) model: AR parameter = +0.9')
plt.plot(sim1)

# AR(1) MA(1) AR parameter = -0.9
plt.subplot(4,1,2)
ar2 = np.array([1, 0.9]) # We choose +0.9 as AR parameter is -0.9
ma2 = np.array([1])
AR2 = ArmaProcess(ar2, ma2)
sim2 = AR2.generate_sample(nsample=1000)
plt.title('AR(1) model: AR parameter = -0.9')
plt.plot(sim2)

# AR(2) MA(1) AR parameter = 0.9
plt.subplot(4,1,3)
ar3 = np.array([2, -0.9]) # We choose -0.9 as AR parameter is +0.9
ma3 = np.array([1])
AR3 = ArmaProcess(ar3, ma3)
sim3 = AR3.generate_sample(nsample=1000)
plt.title('AR(2) model: AR parameter = +0.9')
plt.plot(sim3)

# AR(2) MA(1) AR parameter = -0.9
plt.subplot(4,1,4)
ar4 = np.array([2, 0.9]) # We choose +0.9 as AR parameter is -0.9
ma4 = np.array([1])
AR4 = ArmaProcess(ar4, ma4)
sim4 = AR4.generate_sample(nsample=1000)
plt.title('AR(2) model: AR parameter = -0.9')
plt.plot(sim4)
plt.show()
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_205_0.png)
    


#### Simulated Data Model Fitting


```python
model = AutoReg(sim1, lags = 1, old_names=False)
result = model.fit()
print(result.summary())
print("μ={} ,ϕ={}".format(result.params[0],result.params[1]))
```

                                AutoReg Model Results                             
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 1000
    Model:                     AutoReg(1)   Log Likelihood               -1400.607
    Method:               Conditional MLE   S.D. of innovations              0.983
    Date:                Fri, 17 Oct 2025   AIC                           2807.214
    Time:                        15:55:18   BIC                           2821.934
    Sample:                             1   HQIC                          2812.809
                                     1000                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.0066      0.031      0.212      0.832      -0.054       0.068
    y.L1           0.8986      0.014     64.736      0.000       0.871       0.926
                                        Roots                                    
    =============================================================================
                      Real          Imaginary           Modulus         Frequency
    -----------------------------------------------------------------------------
    AR.1            1.1129           +0.0000j            1.1129            0.0000
    -----------------------------------------------------------------------------
    μ=0.0066001852555156535 ,ϕ=0.8985649550456631
    

- Conclusion:
    - AutoRegressive Coefficient AR ($\phi$) ≈ **0.9131**. $\phi$ is around 0.9 which is what we chose as AR parameter in our first simulated model.
        - Nghĩa là 90% giá trị hiện tại được kế thừa từ giá trị của kỳ trước. Chuỗi này có tính “dễ đoán” và có quán tính mạnh (momentum).
    - Constant ($const = c = \mu$) ≈ **0.05547**
        - Là mức giá trị trung bình cơ bản mà dữ liệu sẽ quay về trong dài hạn nếu không có tác động bên ngoài.
    - P-value is extremely small 0.000 (< 0.05) → hệ số có ý nghĩa thống kê (statistically significant).
    - Final: 
        - Kết quả AR(1) với hệ số 0.9 cho thấy chuỗi dữ liệu có độ bền cao, nghĩa là phần lớn biến động hiện tại được giải thích bởi giá trị của kỳ trước. Mô hình này cho phép dự báo ngắn hạn với độ tin cậy khá cao, vì dữ liệu có xu hướng duy trì đà (momentum) thay vì thay đổi ngẫu nhiên.
        - Example: “Nếu hôm qua bạn bán được 100 sản phẩm, thì hôm nay mô hình dự báo bạn sẽ bán khoảng 90 + (noise) sản phẩm — nghĩa là kết quả hôm nay gần như đi theo đà của hôm qua.”

- `from statsmodels.tsa.ar_model import AutoReg` is a class used to fit an autoregressive time series model.
    - It estimates linear relationships between an observation and its past values.
    - Params:
        - `sim1`: Array-like time series data to be modeled
        - `lags=1`: Uses one previous time step (i.e., AR(1)) as predictor
        - `old_names=False`: Disables legacy parameter naming convention in output
    - Side effects: None directly; creates a model object ready for fitting
    - Returns: An AutoReg model instance that can be fitted to estimate parameters
    - Typical usage: Followed by .fit() to obtain results (e.g., coefficients, diagnostics)

#### Simulated Data Model Prediction


```python
# Predicting simulated AR(1) model 
result.plot_predict(start=900, end=1010, figsize = (15, 5))
plt.show()
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_211_0.png)
    



```python
rmse = math.sqrt(mean_squared_error(sim1[900:1011], result.predict(start=900,end=999)))
print("The root mean squared error is {}.".format(rmse))
```

    The root mean squared error is 0.9682846341269183.
    

### 4.1.2. Google Close Stock Price - AutoRegressive Model

Stages:
1. Stationarity Testing for Google stock price => Output: p-value > 0.05 => Random Walk >> None Stationary
2. If the series is not stationary -> phải lấy sai phân (differencing)
3. Define the selected lag: Use ACF/PACF or AIC/BIC to select the lag
4. Fit the model: Estimate the parameters of the model
5. Forecast

- ADF Test:

    | Concept                         | Meaning                                                                                                      |
    | ------------------------------- | ------------------------------------------------------------------------------------------------------------ |
    | **Null hypothesis (H₀)**        | The series is **non-stationary** (has a unit root).                                                          |
    | **Alternative hypothesis (H₁)** | The series is **stationary** (no unit root).                                                                 |
    | **Decision rule**               | If **p-value < 0.05**, reject H₀ → stationary.<br>If **p-value ≥ 0.05**, fail to reject H₀ → non-stationary. |



```python
gg_close_adf = adfuller(google_stock['Close'])
print(f"p-value of Google Close Stock Price: {float(gg_close_adf[1]):.10f}")
```

    p-value of Google Close Stock Price: 0.9967315858
    

- Result:

    | Metric       | Value                               | Interpretation                 |
    | ------------ | ----------------------------------- | ------------------------------ |
    | **p-value**  | 0.9967                              | Much **greater than 0.05**     |
    | **Decision** | Fail to reject H₀                   | The data **is not stationary** |
    | **Meaning**  | The series contains a **unit root** |                                |

- Conclusion
    - The Google Close Stock Price time series is non-stationary.
    - That means:
        - Its mean and variance change over time.
        - There’s a trend or drift in prices (they’re not mean-reverting).
        - The data behaves like a random walk — typical for stock prices.

- Implication for Modeling:
    | Step                | Action                                                                   | Explanation                                         |
    | ------------------- | ------------------------------------------------------------------------ | --------------------------------------------------- |
    | **1. Differencing** | Apply first difference: `google_stock['Close'].diff()`                   | This removes the trend and stabilizes the mean.     |
    | **2. Re-test ADF**  | Run ADF again on the differenced data                                    | Expect **p-value < 0.05** if it becomes stationary. |
    | **3. Use ARIMA**    | Since data is non-stationary → use **ARIMA(p,1,q)** instead of **AR(p)** | The `d=1` term accounts for differencing.           |




```python
google_stock["Close"]
```




    Date
    2006-01-03     217.83
    2006-01-04     222.84
    2006-01-05     225.85
    2006-01-06     233.06
    2006-01-09     233.68
                   ...   
    2017-12-22    1068.86
    2017-12-26    1065.85
    2017-12-27    1060.20
    2017-12-28    1055.95
    2017-12-29    1053.40
    Name: Close, Length: 3019, dtype: float64




```python
google_stock["Close"].diff().iloc[1:]
```




    Date
    2006-01-04    5.01
    2006-01-05    3.01
    2006-01-06    7.21
    2006-01-09    0.62
    2006-01-10    1.43
                  ... 
    2017-12-22   -1.99
    2017-12-26   -3.01
    2017-12-27   -5.65
    2017-12-28   -4.25
    2017-12-29   -2.55
    Name: Close, Length: 3018, dtype: float64



- The number of past periods = 1


```python
# Predicting closing prices of google
humid = AutoReg(google_stock["Close"].diff().iloc[1:].values, lags = 1, old_names=False)
res = humid.fit()
res.plot_predict(start=900, end=1010, figsize=(15,5))
plt.show()
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_221_0.png)
    



```python
print(res.summary())
print("μ={} ,ϕ={}".format(res.params[0],res.params[1]))
```

                                AutoReg Model Results                             
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 3018
    Model:                     AutoReg(1)   Log Likelihood              -10111.410
    Method:               Conditional MLE   S.D. of innovations              6.907
    Date:                Fri, 17 Oct 2025   AIC                          20228.820
    Time:                        15:55:19   BIC                          20246.856
    Sample:                             1   HQIC                         20235.305
                                     3018                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.2675      0.126      2.126      0.034       0.021       0.514
    y.L1           0.0280      0.018      1.541      0.123      -0.008       0.064
                                        Roots                                    
    =============================================================================
                      Real          Imaginary           Modulus         Frequency
    -----------------------------------------------------------------------------
    AR.1           35.6632           +0.0000j           35.6632            0.0000
    -----------------------------------------------------------------------------
    μ=0.2675038275538285 ,ϕ=0.02804008050171732
    

- Conclusion:
    | Element | Meaning | Interpretation |
    | --- | --- | --- |
    | **Model:** AutoReg(1) | Autoregressive model of order 1 | The closing price today is predicted from **yesterday’s closing price** |
    | **Equation:** | $y_t = 0.2675 + 0.0280 \times y_{t-1} + \epsilon_t$ | Indicates a **small autoregressive dependence** |
    | **const = 0.2675** | Constant term | The baseline (intercept) value added to each prediction |
    | **y.L1 = 0.0280** | Coefficient for previous day’s price | A 1-unit increase in yesterday’s price increases today’s expected price by **0.028 units**, very small |
    | **p-value = 0.123** | Significance of y.L1 | The lag effect is **not statistically significant** at 95% confidence (p > 0.05) |
    | **AIC/BIC** | Model selection criteria | AIC = 20228.82, BIC = 20246.86 → used to compare with more complex models (lower = better) |
    | **Roots (AR.1 = 35.66)** | Model stability | Because the root modulus > 1, the process is **stationary and stable** |

    - Business Interpretation
        - The relationship between yesterday’s and today’s Google closing price is weak $\phi ≈ 0.028$. <br/>
        → This means the stock price does not strongly depend on its immediate past — it behaves more like a random walk.
        - The constant term (0.2675) adds a small positive drift — a general upward tendency on average, but not strong enough to be a deterministic trend.
        - Since p-value = 0.123, the model fails to show a significant predictive relationship. <br/>
        → The day-to-day price movements are mostly driven by new information, not by past prices (efficient market hypothesis effect).

- The number of past periods = 7


```python
# Predicting closing prices of google
humid = AutoReg(google_stock["Close"].diff().iloc[1:].values, lags = 7, old_names=False)
res = humid.fit()
res.plot_predict(start=900, end=1010, figsize=(15,5))
plt.show()
print(res.summary())
print("μ={} ,ϕ={}".format(res.params[0],res.params[1:]))
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_225_0.png)
    


                                AutoReg Model Results                             
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 3018
    Model:                     AutoReg(7)   Log Likelihood              -10090.193
    Method:               Conditional MLE   S.D. of innovations              6.904
    Date:                Fri, 17 Oct 2025   AIC                          20198.386
    Time:                        15:55:19   BIC                          20252.476
    Sample:                             7   HQIC                         20217.838
                                     3018                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.2817      0.127      2.226      0.026       0.034       0.530
    y.L1           0.0270      0.018      1.480      0.139      -0.009       0.063
    y.L2           0.0125      0.018      0.688      0.491      -0.023       0.048
    y.L3          -0.0353      0.018     -1.936      0.053      -0.071       0.000
    y.L4          -0.0187      0.018     -1.028      0.304      -0.054       0.017
    y.L5          -0.0109      0.018     -0.596      0.551      -0.047       0.025
    y.L6          -0.0150      0.018     -0.822      0.411      -0.051       0.021
    y.L7           0.0087      0.018      0.479      0.632      -0.027       0.044
                                        Roots                                    
    =============================================================================
                      Real          Imaginary           Modulus         Frequency
    -----------------------------------------------------------------------------
    AR.1           -1.6311           -0.8000j            1.8167           -0.4274
    AR.2           -1.6311           +0.8000j            1.8167            0.4274
    AR.3           -0.2657           -1.8987j            1.9172           -0.2721
    AR.4           -0.2657           +1.8987j            1.9172            0.2721
    AR.5            1.3513           -1.2395j            1.8337           -0.1181
    AR.6            1.3513           +1.2395j            1.8337            0.1181
    AR.7            2.8076           -0.0000j            2.8076           -0.0000
    -----------------------------------------------------------------------------
    μ=0.28174501411663927 ,ϕ=[ 0.02696487  0.01254804 -0.03530257 -0.01874867 -0.01085571 -0.01498869
      0.00873151]
    

- The number of past periods = 30


```python
# Predicting closing prices of google
humid = AutoReg(google_stock["Close"].diff().iloc[1:].values, lags = 30, old_names=False)
res = humid.fit()
res.plot_predict(start=900, end=1010, figsize=(15,5))
plt.show()
print(res.summary())
print("μ={} ,ϕ={}".format(res.params[0],res.params[1:]))
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_227_0.png)
    


                                AutoReg Model Results                             
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 3018
    Model:                    AutoReg(30)   Log Likelihood               -9980.925
    Method:               Conditional MLE   S.D. of innovations              6.830
    Date:                Fri, 17 Oct 2025   AIC                          20025.850
    Time:                        15:55:19   BIC                          20217.925
    Sample:                            30   HQIC                         20094.953
                                     3018                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.3287      0.128      2.562      0.010       0.077       0.580
    y.L1           0.0239      0.018      1.305      0.192      -0.012       0.060
    y.L2           0.0122      0.018      0.667      0.505      -0.024       0.048
    y.L3          -0.0346      0.018     -1.897      0.058      -0.070       0.001
    y.L4          -0.0229      0.018     -1.253      0.210      -0.059       0.013
    y.L5          -0.0122      0.018     -0.669      0.503      -0.048       0.024
    y.L6          -0.0128      0.018     -0.704      0.481      -0.049       0.023
    y.L7           0.0104      0.018      0.572      0.568      -0.025       0.046
    y.L8          -0.0343      0.018     -1.880      0.060      -0.070       0.001
    y.L9          -0.0119      0.018     -0.653      0.514      -0.048       0.024
    y.L10         -0.0029      0.018     -0.160      0.873      -0.039       0.033
    y.L11          0.0057      0.018      0.315      0.753      -0.030       0.041
    y.L12          0.0040      0.018      0.223      0.824      -0.032       0.040
    y.L13         -0.0164      0.018     -0.901      0.368      -0.052       0.019
    y.L14         -0.0069      0.018     -0.381      0.703      -0.043       0.029
    y.L15          0.0029      0.018      0.157      0.875      -0.033       0.038
    y.L16          0.0015      0.018      0.081      0.935      -0.034       0.037
    y.L17          0.0486      0.018      2.671      0.008       0.013       0.084
    y.L18         -0.0145      0.018     -0.799      0.424      -0.050       0.021
    y.L19          0.0333      0.018      1.833      0.067      -0.002       0.069
    y.L20          0.0481      0.018      2.641      0.008       0.012       0.084
    y.L21          0.0202      0.018      1.108      0.268      -0.016       0.056
    y.L22         -0.0124      0.018     -0.680      0.496      -0.048       0.023
    y.L23         -0.0284      0.018     -1.557      0.119      -0.064       0.007
    y.L24         -0.0048      0.018     -0.263      0.792      -0.041       0.031
    y.L25         -0.0543      0.018     -2.973      0.003      -0.090      -0.019
    y.L26         -0.0237      0.018     -1.296      0.195      -0.060       0.012
    y.L27          0.0139      0.018      0.761      0.447      -0.022       0.050
    y.L28          0.0463      0.018      2.528      0.011       0.010       0.082
    y.L29         -0.0536      0.018     -2.922      0.003      -0.090      -0.018
    y.L30         -0.0353      0.018     -1.920      0.055      -0.071       0.001
                                        Roots                                     
    ==============================================================================
                       Real          Imaginary           Modulus         Frequency
    ------------------------------------------------------------------------------
    AR.1             1.0875           -0.1134j            1.0934           -0.0165
    AR.2             1.0875           +0.1134j            1.0934            0.0165
    AR.3             1.0103           -0.3642j            1.0739           -0.0551
    AR.4             1.0103           +0.3642j            1.0739            0.0551
    AR.5             0.8466           -0.6887j            1.0913           -0.1087
    AR.6             0.8466           +0.6887j            1.0913            0.1087
    AR.7             1.0048           -0.6134j            1.1772           -0.0872
    AR.8             1.0048           +0.6134j            1.1772            0.0872
    AR.9             0.6397           -0.8603j            1.0721           -0.1482
    AR.10            0.6397           +0.8603j            1.0721            0.1482
    AR.11            0.4314           -0.9788j            1.0696           -0.1839
    AR.12            0.4314           +0.9788j            1.0696            0.1839
    AR.13            0.2205           -1.0629j            1.0856           -0.2174
    AR.14            0.2205           +1.0629j            1.0856            0.2174
    AR.15           -0.0205           -1.0656j            1.0658           -0.2531
    AR.16           -0.0205           +1.0656j            1.0658            0.2531
    AR.17           -0.2734           -1.0344j            1.0700           -0.2911
    AR.18           -0.2734           +1.0344j            1.0700            0.2911
    AR.19           -0.5026           -0.9826j            1.1037           -0.3252
    AR.20           -0.5026           +0.9826j            1.1037            0.3252
    AR.21           -0.6919           -0.8557j            1.1005           -0.3582
    AR.22           -0.6919           +0.8557j            1.1005            0.3582
    AR.23           -0.8881           -0.6989j            1.1301           -0.3939
    AR.24           -0.8881           +0.6989j            1.1301            0.3939
    AR.25           -1.1113           -0.0000j            1.1113           -0.5000
    AR.26           -1.0596           -0.2488j            1.0884           -0.4633
    AR.27           -1.0596           +0.2488j            1.0884            0.4633
    AR.28           -0.9725           -0.5071j            1.0967           -0.4235
    AR.29           -0.9725           +0.5071j            1.0967            0.4235
    AR.30           -2.0725           -0.0000j            2.0725           -0.5000
    ------------------------------------------------------------------------------
    μ=0.32868389125474945 ,ϕ=[ 0.02386693  0.012182   -0.03459527 -0.0228581  -0.01221168 -0.0128357
      0.01041627 -0.03425436 -0.01191265 -0.0029249   0.00572866  0.00404887
     -0.01638481 -0.00692461  0.00285975  0.00147289  0.0485616  -0.01454116
      0.03332099  0.0480662   0.02018708 -0.01241448 -0.02842343 -0.00481013
     -0.05433042 -0.02372808  0.0139333   0.04628325 -0.05357589 -0.03526   ]
    

- The number of past periods = 365


```python
# Predicting closing prices of google
humid = AutoReg(google_stock["Close"].diff().iloc[1:].values, lags = 365, old_names=False)
res = humid.fit()
res.plot_predict(start=900, end=1010, figsize=(15,5))
plt.show()
print(res.summary())
print("μ={} ,ϕ={}".format(res.params[0],res.params[1:]))
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_229_0.png)
    


                                AutoReg Model Results                             
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 3018
    Model:                   AutoReg(365)   Log Likelihood               -8747.851
    Method:               Conditional MLE   S.D. of innovations              6.543
    Date:                Fri, 17 Oct 2025   AIC                          18229.703
    Time:                        15:55:19   BIC                          20388.927
    Sample:                           365   HQIC                         19011.240
                                     3018                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.3532      0.173      2.042      0.041       0.014       0.692
    y.L1           0.0198      0.019      1.022      0.307      -0.018       0.058
    y.L2           0.0116      0.019      0.595      0.552      -0.026       0.050
    y.L3          -0.0238      0.019     -1.227      0.220      -0.062       0.014
    y.L4          -0.0263      0.019     -1.355      0.175      -0.064       0.012
    y.L5          -0.0134      0.019     -0.690      0.490      -0.051       0.025
    y.L6           0.0032      0.019      0.163      0.871      -0.035       0.041
    y.L7           0.0102      0.019      0.528      0.598      -0.028       0.048
    y.L8          -0.0229      0.019     -1.185      0.236      -0.061       0.015
    y.L9          -0.0002      0.019     -0.010      0.992      -0.038       0.038
    y.L10          0.0047      0.019      0.245      0.807      -0.033       0.043
    y.L11          0.0043      0.019      0.220      0.826      -0.034       0.042
    y.L12         -0.0025      0.019     -0.131      0.895      -0.040       0.035
    y.L13         -0.0193      0.019     -0.996      0.319      -0.057       0.019
    y.L14         -0.0011      0.019     -0.055      0.956      -0.039       0.037
    y.L15         -0.0022      0.019     -0.113      0.910      -0.040       0.036
    y.L16          0.0055      0.019      0.285      0.776      -0.032       0.043
    y.L17          0.0429      0.019      2.216      0.027       0.005       0.081
    y.L18         -0.0189      0.019     -0.976      0.329      -0.057       0.019
    y.L19          0.0339      0.019      1.749      0.080      -0.004       0.072
    y.L20          0.0641      0.019      3.301      0.001       0.026       0.102
    y.L21         -0.0059      0.019     -0.304      0.761      -0.044       0.032
    y.L22         -0.0272      0.019     -1.397      0.162      -0.065       0.011
    y.L23         -0.0315      0.019     -1.615      0.106      -0.070       0.007
    y.L24          0.0058      0.020      0.298      0.766      -0.032       0.044
    y.L25         -0.0546      0.020     -2.800      0.005      -0.093      -0.016
    y.L26         -0.0216      0.020     -1.104      0.270      -0.060       0.017
    y.L27          0.0156      0.020      0.798      0.425      -0.023       0.054
    y.L28          0.0407      0.020      2.079      0.038       0.002       0.079
    y.L29         -0.0478      0.020     -2.443      0.015      -0.086      -0.009
    y.L30         -0.0297      0.020     -1.518      0.129      -0.068       0.009
    y.L31          0.0081      0.020      0.413      0.680      -0.030       0.047
    y.L32          0.0306      0.020      1.562      0.118      -0.008       0.069
    y.L33          0.0280      0.020      1.429      0.153      -0.010       0.066
    y.L34         -0.0164      0.020     -0.836      0.403      -0.055       0.022
    y.L35          0.0345      0.020      1.761      0.078      -0.004       0.073
    y.L36          0.0165      0.020      0.843      0.399      -0.022       0.055
    y.L37         -0.0526      0.020     -2.683      0.007      -0.091      -0.014
    y.L38         -0.0368      0.020     -1.875      0.061      -0.075       0.002
    y.L39          0.0282      0.020      1.437      0.151      -0.010       0.067
    y.L40          0.0115      0.020      0.585      0.558      -0.027       0.050
    y.L41          0.0195      0.020      0.992      0.321      -0.019       0.058
    y.L42          0.0169      0.020      0.859      0.391      -0.022       0.055
    y.L43         -0.0168      0.020     -0.854      0.393      -0.055       0.022
    y.L44         -0.0103      0.020     -0.522      0.602      -0.049       0.028
    y.L45          0.0047      0.020      0.238      0.812      -0.034       0.043
    y.L46         -0.0066      0.020     -0.332      0.740      -0.045       0.032
    y.L47          0.0057      0.020      0.289      0.772      -0.033       0.044
    y.L48         -0.0282      0.020     -1.426      0.154      -0.067       0.011
    y.L49         -0.0191      0.020     -0.965      0.335      -0.058       0.020
    y.L50         -0.0431      0.020     -2.177      0.029      -0.082      -0.004
    y.L51         -0.0490      0.020     -2.469      0.014      -0.088      -0.010
    y.L52          0.0182      0.020      0.916      0.360      -0.021       0.057
    y.L53          0.0264      0.020      1.329      0.184      -0.013       0.065
    y.L54         -0.0111      0.020     -0.558      0.577      -0.050       0.028
    y.L55          0.0244      0.020      1.228      0.219      -0.015       0.063
    y.L56          0.0172      0.020      0.865      0.387      -0.022       0.056
    y.L57          0.0409      0.020      2.055      0.040       0.002       0.080
    y.L58          0.0120      0.020      0.603      0.547      -0.027       0.051
    y.L59          0.0088      0.020      0.444      0.657      -0.030       0.048
    y.L60          0.0045      0.020      0.226      0.821      -0.035       0.044
    y.L61          0.0013      0.020      0.068      0.946      -0.038       0.040
    y.L62         -0.0256      0.020     -1.288      0.198      -0.065       0.013
    y.L63         -0.0470      0.020     -2.356      0.018      -0.086      -0.008
    y.L64         -0.0434      0.020     -2.177      0.029      -0.083      -0.004
    y.L65         -0.0221      0.020     -1.105      0.269      -0.061       0.017
    y.L66         -0.0608      0.020     -3.039      0.002      -0.100      -0.022
    y.L67         -0.0162      0.020     -0.809      0.419      -0.055       0.023
    y.L68          0.0060      0.020      0.302      0.763      -0.033       0.045
    y.L69          0.0241      0.020      1.207      0.227      -0.015       0.063
    y.L70          0.0056      0.020      0.279      0.780      -0.034       0.045
    y.L71          0.0109      0.020      0.546      0.585      -0.028       0.050
    y.L72         -0.0162      0.020     -0.811      0.417      -0.055       0.023
    y.L73          0.0179      0.020      0.896      0.370      -0.021       0.057
    y.L74         -0.0263      0.020     -1.315      0.188      -0.066       0.013
    y.L75         -0.0210      0.020     -1.051      0.293      -0.060       0.018
    y.L76          0.0161      0.020      0.805      0.421      -0.023       0.055
    y.L77          0.0329      0.020      1.644      0.100      -0.006       0.072
    y.L78          0.0370      0.020      1.850      0.064      -0.002       0.076
    y.L79         -0.0432      0.020     -2.156      0.031      -0.082      -0.004
    y.L80          0.0098      0.020      0.490      0.624      -0.029       0.049
    y.L81          0.0016      0.020      0.082      0.935      -0.038       0.041
    y.L82          0.0034      0.020      0.169      0.866      -0.036       0.043
    y.L83          0.0169      0.020      0.846      0.398      -0.022       0.056
    y.L84         -0.0422      0.020     -2.106      0.035      -0.081      -0.003
    y.L85          0.0140      0.020      0.698      0.485      -0.025       0.053
    y.L86          0.0253      0.020      1.262      0.207      -0.014       0.065
    y.L87          0.0040      0.020      0.201      0.841      -0.035       0.043
    y.L88         -0.0419      0.020     -2.092      0.036      -0.081      -0.003
    y.L89         -0.0023      0.020     -0.115      0.909      -0.042       0.037
    y.L90         -0.0130      0.020     -0.647      0.517      -0.052       0.026
    y.L91          0.0180      0.020      0.894      0.371      -0.021       0.057
    y.L92         -0.0081      0.020     -0.405      0.685      -0.048       0.031
    y.L93          0.0060      0.020      0.299      0.765      -0.033       0.045
    y.L94         -0.0038      0.020     -0.188      0.851      -0.043       0.036
    y.L95          0.0507      0.020      2.522      0.012       0.011       0.090
    y.L96          0.0049      0.020      0.246      0.806      -0.034       0.044
    y.L97         -0.0071      0.020     -0.355      0.723      -0.047       0.032
    y.L98          0.0138      0.020      0.687      0.492      -0.026       0.053
    y.L99          0.0409      0.020      2.032      0.042       0.001       0.080
    y.L100        -0.0208      0.020     -1.033      0.302      -0.060       0.019
    y.L101        -0.0103      0.020     -0.512      0.609      -0.050       0.029
    y.L102        -0.0089      0.020     -0.441      0.659      -0.048       0.031
    y.L103        -0.0269      0.020     -1.337      0.181      -0.066       0.013
    y.L104         0.0094      0.020      0.468      0.640      -0.030       0.049
    y.L105        -0.0140      0.020     -0.695      0.487      -0.053       0.025
    y.L106         0.0388      0.020      1.925      0.054      -0.001       0.078
    y.L107         0.0494      0.020      2.451      0.014       0.010       0.089
    y.L108        -0.0050      0.020     -0.248      0.804      -0.045       0.035
    y.L109         0.0156      0.020      0.774      0.439      -0.024       0.055
    y.L110        -0.0314      0.020     -1.557      0.120      -0.071       0.008
    y.L111        -0.0118      0.020     -0.582      0.561      -0.051       0.028
    y.L112        -0.0053      0.020     -0.261      0.794      -0.045       0.034
    y.L113         0.0182      0.020      0.898      0.369      -0.022       0.058
    y.L114         0.0338      0.020      1.671      0.095      -0.006       0.073
    y.L115        -0.0568      0.020     -2.804      0.005      -0.097      -0.017
    y.L116         0.0570      0.020      2.812      0.005       0.017       0.097
    y.L117        -0.0230      0.020     -1.133      0.257      -0.063       0.017
    y.L118        -0.0356      0.020     -1.754      0.079      -0.075       0.004
    y.L119         0.0063      0.020      0.309      0.757      -0.034       0.046
    y.L120         0.0067      0.020      0.327      0.744      -0.033       0.047
    y.L121        -0.0186      0.020     -0.916      0.360      -0.059       0.021
    y.L122         0.0052      0.020      0.254      0.799      -0.035       0.045
    y.L123         0.0337      0.020      1.654      0.098      -0.006       0.074
    y.L124        -0.0013      0.020     -0.063      0.950      -0.041       0.039
    y.L125        -0.0407      0.020     -1.996      0.046      -0.081      -0.001
    y.L126        -0.0337      0.020     -1.654      0.098      -0.074       0.006
    y.L127         0.0296      0.020      1.452      0.147      -0.010       0.070
    y.L128        -0.0052      0.020     -0.256      0.798      -0.045       0.035
    y.L129        -0.0004      0.020     -0.021      0.983      -0.040       0.040
    y.L130      1.123e-05      0.020      0.001      1.000      -0.040       0.040
    y.L131        -0.0084      0.020     -0.411      0.681      -0.049       0.032
    y.L132        -0.0318      0.020     -1.552      0.121      -0.072       0.008
    y.L133        -0.0410      0.020     -2.004      0.045      -0.081      -0.001
    y.L134         0.0453      0.020      2.213      0.027       0.005       0.086
    y.L135         0.0318      0.021      1.549      0.121      -0.008       0.072
    y.L136         0.0204      0.021      0.996      0.319      -0.020       0.061
    y.L137         0.0056      0.021      0.271      0.787      -0.035       0.046
    y.L138        -0.0343      0.021     -1.670      0.095      -0.075       0.006
    y.L139         0.0002      0.021      0.011      0.991      -0.040       0.040
    y.L140        -0.0245      0.021     -1.193      0.233      -0.065       0.016
    y.L141         0.0037      0.021      0.182      0.856      -0.037       0.044
    y.L142        -0.0281      0.021     -1.362      0.173      -0.069       0.012
    y.L143         0.0020      0.021      0.097      0.922      -0.038       0.042
    y.L144         0.0052      0.021      0.252      0.801      -0.035       0.046
    y.L145         0.0223      0.021      1.082      0.279      -0.018       0.063
    y.L146      6.419e-05      0.021      0.003      0.998      -0.040       0.041
    y.L147         0.0213      0.021      1.032      0.302      -0.019       0.062
    y.L148        -0.0416      0.021     -2.015      0.044      -0.082      -0.001
    y.L149         0.0119      0.021      0.577      0.564      -0.029       0.052
    y.L150         0.0456      0.021      2.207      0.027       0.005       0.086
    y.L151        -0.0043      0.021     -0.207      0.836      -0.045       0.036
    y.L152        -0.0060      0.021     -0.289      0.773      -0.047       0.035
    y.L153         0.0295      0.021      1.425      0.154      -0.011       0.070
    y.L154         0.0090      0.021      0.436      0.663      -0.032       0.050
    y.L155        -0.0382      0.021     -1.848      0.065      -0.079       0.002
    y.L156         0.0324      0.021      1.565      0.118      -0.008       0.073
    y.L157         0.0237      0.021      1.144      0.253      -0.017       0.064
    y.L158         0.0270      0.021      1.298      0.194      -0.014       0.068
    y.L159        -0.0245      0.021     -1.178      0.239      -0.065       0.016
    y.L160        -0.0187      0.021     -0.902      0.367      -0.059       0.022
    y.L161         0.0010      0.021      0.046      0.963      -0.040       0.042
    y.L162        -0.0135      0.021     -0.650      0.515      -0.054       0.027
    y.L163        -0.0124      0.021     -0.597      0.550      -0.053       0.028
    y.L164        -0.0065      0.021     -0.315      0.752      -0.047       0.034
    y.L165         0.0007      0.021      0.031      0.975      -0.040       0.041
    y.L166         0.0214      0.021      1.035      0.301      -0.019       0.062
    y.L167        -0.0100      0.021     -0.482      0.629      -0.051       0.031
    y.L168         0.0195      0.021      0.942      0.346      -0.021       0.060
    y.L169        -0.0466      0.021     -2.251      0.024      -0.087      -0.006
    y.L170        -0.0123      0.021     -0.595      0.552      -0.053       0.028
    y.L171         0.0201      0.021      0.964      0.335      -0.021       0.061
    y.L172        -0.0205      0.021     -0.982      0.326      -0.061       0.020
    y.L173         0.0205      0.021      0.983      0.326      -0.020       0.061
    y.L174        -0.0132      0.021     -0.632      0.528      -0.054       0.028
    y.L175         0.0564      0.021      2.703      0.007       0.016       0.097
    y.L176        -0.0280      0.021     -1.340      0.180      -0.069       0.013
    y.L177         0.0285      0.021      1.362      0.173      -0.013       0.069
    y.L178        -0.0048      0.021     -0.228      0.819      -0.046       0.036
    y.L179         0.0275      0.021      1.313      0.189      -0.014       0.069
    y.L180         0.0239      0.021      1.140      0.254      -0.017       0.065
    y.L181         0.0227      0.021      1.083      0.279      -0.018       0.064
    y.L182         0.0036      0.021      0.172      0.864      -0.037       0.045
    y.L183         0.0187      0.021      0.892      0.372      -0.022       0.060
    y.L184         0.0043      0.021      0.205      0.837      -0.037       0.045
    y.L185         0.0509      0.021      2.429      0.015       0.010       0.092
    y.L186        -0.0272      0.021     -1.299      0.194      -0.068       0.014
    y.L187         0.0109      0.021      0.521      0.602      -0.030       0.052
    y.L188         0.0021      0.021      0.101      0.920      -0.039       0.043
    y.L189        -0.0150      0.021     -0.715      0.475      -0.056       0.026
    y.L190        -0.0104      0.021     -0.496      0.620      -0.051       0.031
    y.L191        -0.0349      0.021     -1.666      0.096      -0.076       0.006
    y.L192         0.0071      0.021      0.337      0.736      -0.034       0.048
    y.L193        -0.0083      0.021     -0.396      0.692      -0.049       0.033
    y.L194        -0.0202      0.021     -0.965      0.335      -0.061       0.021
    y.L195         0.0151      0.021      0.719      0.472      -0.026       0.056
    y.L196         0.0027      0.021      0.129      0.897      -0.038       0.044
    y.L197         0.0090      0.021      0.428      0.669      -0.032       0.050
    y.L198         0.0296      0.021      1.412      0.158      -0.011       0.071
    y.L199        -0.0194      0.021     -0.924      0.355      -0.061       0.022
    y.L200        -0.0059      0.021     -0.283      0.777      -0.047       0.035
    y.L201        -0.0269      0.021     -1.282      0.200      -0.068       0.014
    y.L202         0.0303      0.021      1.444      0.149      -0.011       0.072
    y.L203         0.0035      0.021      0.165      0.869      -0.038       0.045
    y.L204         0.0186      0.021      0.884      0.377      -0.023       0.060
    y.L205         0.0402      0.021      1.914      0.056      -0.001       0.081
    y.L206        -0.0183      0.021     -0.869      0.385      -0.060       0.023
    y.L207         0.0401      0.021      1.905      0.057      -0.001       0.081
    y.L208        -0.0134      0.021     -0.636      0.525      -0.055       0.028
    y.L209        -0.0002      0.021     -0.011      0.991      -0.041       0.041
    y.L210         0.0052      0.021      0.246      0.806      -0.036       0.046
    y.L211         0.0187      0.021      0.890      0.373      -0.023       0.060
    y.L212        -0.0272      0.021     -1.291      0.197      -0.068       0.014
    y.L213         0.0008      0.021      0.040      0.968      -0.040       0.042
    y.L214         0.0044      0.021      0.211      0.833      -0.037       0.046
    y.L215        -0.0393      0.021     -1.867      0.062      -0.081       0.002
    y.L216        -0.0005      0.021     -0.026      0.980      -0.042       0.041
    y.L217        -0.0153      0.021     -0.728      0.466      -0.057       0.026
    y.L218         0.0326      0.021      1.553      0.121      -0.009       0.074
    y.L219         0.0128      0.021      0.607      0.544      -0.028       0.054
    y.L220        -0.0359      0.021     -1.706      0.088      -0.077       0.005
    y.L221        -0.0307      0.021     -1.459      0.145      -0.072       0.011
    y.L222        -0.0189      0.021     -0.896      0.370      -0.060       0.022
    y.L223         0.0062      0.021      0.295      0.768      -0.035       0.047
    y.L224        -0.0049      0.021     -0.233      0.816      -0.046       0.036
    y.L225         0.0145      0.021      0.688      0.492      -0.027       0.056
    y.L226        -0.0380      0.021     -1.807      0.071      -0.079       0.003
    y.L227         0.0022      0.021      0.102      0.919      -0.039       0.043
    y.L228        -0.0049      0.021     -0.231      0.818      -0.046       0.036
    y.L229         0.0061      0.021      0.291      0.771      -0.035       0.047
    y.L230         0.0069      0.021      0.327      0.743      -0.034       0.048
    y.L231        -0.0247      0.021     -1.173      0.241      -0.066       0.017
    y.L232        -0.0233      0.021     -1.105      0.269      -0.065       0.018
    y.L233         0.0391      0.021      1.859      0.063      -0.002       0.080
    y.L234        -0.0102      0.021     -0.486      0.627      -0.052       0.031
    y.L235        -0.0297      0.021     -1.411      0.158      -0.071       0.012
    y.L236        -0.0092      0.021     -0.435      0.664      -0.050       0.032
    y.L237        -0.0120      0.021     -0.571      0.568      -0.053       0.029
    y.L238        -0.0292      0.021     -1.383      0.167      -0.071       0.012
    y.L239         0.0157      0.021      0.741      0.459      -0.026       0.057
    y.L240         0.0012      0.021      0.058      0.953      -0.040       0.043
    y.L241        -0.0034      0.021     -0.160      0.873      -0.045       0.038
    y.L242        -0.0275      0.021     -1.300      0.194      -0.069       0.014
    y.L243        -0.0199      0.021     -0.943      0.346      -0.061       0.021
    y.L244        -0.0193      0.021     -0.912      0.362      -0.061       0.022
    y.L245         0.0021      0.021      0.101      0.920      -0.039       0.044
    y.L246         0.0237      0.021      1.123      0.261      -0.018       0.065
    y.L247         0.0079      0.021      0.374      0.709      -0.034       0.049
    y.L248         0.0387      0.021      1.829      0.067      -0.003       0.080
    y.L249        -0.0189      0.021     -0.896      0.370      -0.060       0.023
    y.L250         0.0260      0.021      1.233      0.218      -0.015       0.067
    y.L251         0.0081      0.021      0.385      0.700      -0.033       0.050
    y.L252         0.0099      0.021      0.468      0.640      -0.031       0.051
    y.L253        -0.0075      0.021     -0.357      0.721      -0.049       0.034
    y.L254        -0.0132      0.021     -0.627      0.531      -0.055       0.028
    y.L255         0.0100      0.021      0.476      0.634      -0.031       0.051
    y.L256        -0.0074      0.021     -0.351      0.726      -0.049       0.034
    y.L257        -0.0345      0.021     -1.636      0.102      -0.076       0.007
    y.L258         0.0326      0.021      1.545      0.122      -0.009       0.074
    y.L259        -0.0055      0.021     -0.260      0.795      -0.047       0.036
    y.L260        -0.0342      0.021     -1.624      0.104      -0.075       0.007
    y.L261         0.0309      0.021      1.466      0.143      -0.010       0.072
    y.L262         0.0014      0.021      0.064      0.949      -0.040       0.043
    y.L263        -0.0351      0.021     -1.667      0.096      -0.076       0.006
    y.L264        -0.0188      0.021     -0.891      0.373      -0.060       0.023
    y.L265        -0.0229      0.021     -1.089      0.276      -0.064       0.018
    y.L266         0.0063      0.021      0.298      0.765      -0.035       0.048
    y.L267        -0.0011      0.021     -0.052      0.958      -0.042       0.040
    y.L268        -0.0077      0.021     -0.365      0.715      -0.049       0.034
    y.L269         0.0344      0.021      1.629      0.103      -0.007       0.076
    y.L270        -0.0269      0.021     -1.277      0.202      -0.068       0.014
    y.L271         0.0292      0.021      1.383      0.167      -0.012       0.071
    y.L272         0.0256      0.021      1.211      0.226      -0.016       0.067
    y.L273         0.0075      0.021      0.356      0.722      -0.034       0.049
    y.L274        -0.0109      0.021     -0.515      0.606      -0.052       0.031
    y.L275        -0.0549      0.021     -2.595      0.009      -0.096      -0.013
    y.L276        -0.0242      0.021     -1.142      0.254      -0.066       0.017
    y.L277         0.0116      0.021      0.549      0.583      -0.030       0.053
    y.L278        -0.0143      0.021     -0.678      0.498      -0.056       0.027
    y.L279         0.0215      0.021      1.016      0.310      -0.020       0.063
    y.L280        -0.0579      0.021     -2.736      0.006      -0.099      -0.016
    y.L281        -0.0152      0.021     -0.716      0.474      -0.057       0.026
    y.L282        -0.0075      0.021     -0.357      0.721      -0.049       0.034
    y.L283         0.0322      0.021      1.519      0.129      -0.009       0.074
    y.L284         0.0333      0.021      1.571      0.116      -0.008       0.075
    y.L285         0.0115      0.021      0.543      0.587      -0.030       0.053
    y.L286        -0.0044      0.021     -0.208      0.835      -0.046       0.037
    y.L287         0.0067      0.021      0.314      0.753      -0.035       0.048
    y.L288         0.0219      0.021      1.032      0.302      -0.020       0.064
    y.L289         0.0027      0.021      0.129      0.897      -0.039       0.044
    y.L290         0.0056      0.021      0.263      0.793      -0.036       0.047
    y.L291        -0.0525      0.021     -2.470      0.013      -0.094      -0.011
    y.L292         0.0018      0.021      0.085      0.932      -0.040       0.044
    y.L293        -0.0020      0.021     -0.096      0.924      -0.044       0.040
    y.L294         0.0032      0.021      0.150      0.881      -0.039       0.045
    y.L295         0.0268      0.021      1.257      0.209      -0.015       0.068
    y.L296         0.0230      0.021      1.082      0.279      -0.019       0.065
    y.L297         0.0428      0.021      2.008      0.045       0.001       0.085
    y.L298         0.0127      0.021      0.597      0.551      -0.029       0.055
    y.L299         0.0540      0.021      2.530      0.011       0.012       0.096
    y.L300        -0.0112      0.021     -0.523      0.601      -0.053       0.031
    y.L301        -0.0213      0.021     -0.999      0.318      -0.063       0.020
    y.L302        -0.0228      0.021     -1.068      0.285      -0.065       0.019
    y.L303         0.0109      0.021      0.514      0.607      -0.031       0.053
    y.L304        -0.0357      0.021     -1.675      0.094      -0.077       0.006
    y.L305        -0.0418      0.021     -1.959      0.050      -0.084    1.76e-05
    y.L306        -0.0180      0.021     -0.844      0.399      -0.060       0.024
    y.L307        -0.0164      0.021     -0.769      0.442      -0.058       0.025
    y.L308         0.0421      0.021      1.974      0.048       0.000       0.084
    y.L309         0.0128      0.021      0.597      0.551      -0.029       0.055
    y.L310         0.0174      0.021      0.816      0.415      -0.024       0.059
    y.L311        -0.0220      0.021     -1.030      0.303      -0.064       0.020
    y.L312         0.0009      0.021      0.041      0.967      -0.041       0.043
    y.L313         0.0086      0.021      0.401      0.689      -0.033       0.050
    y.L314         0.0167      0.021      0.780      0.435      -0.025       0.059
    y.L315         0.0179      0.021      0.838      0.402      -0.024       0.060
    y.L316         0.0135      0.021      0.633      0.527      -0.028       0.055
    y.L317         0.0088      0.021      0.412      0.680      -0.033       0.051
    y.L318        -0.0099      0.021     -0.467      0.641      -0.052       0.032
    y.L319        -0.0411      0.021     -1.931      0.054      -0.083       0.001
    y.L320         0.0286      0.021      1.344      0.179      -0.013       0.070
    y.L321         0.0235      0.021      1.101      0.271      -0.018       0.065
    y.L322        -0.0033      0.021     -0.155      0.877      -0.045       0.038
    y.L323        -0.0492      0.021     -2.309      0.021      -0.091      -0.007
    y.L324         0.0148      0.021      0.694      0.488      -0.027       0.057
    y.L325        -0.0286      0.021     -1.343      0.179      -0.070       0.013
    y.L326        -0.0058      0.021     -0.272      0.786      -0.048       0.036
    y.L327        -0.0022      0.021     -0.105      0.916      -0.044       0.040
    y.L328        -0.0309      0.021     -1.449      0.147      -0.073       0.011
    y.L329        -0.0112      0.021     -0.524      0.601      -0.053       0.031
    y.L330        -0.0393      0.021     -1.844      0.065      -0.081       0.002
    y.L331         0.0067      0.021      0.314      0.754      -0.035       0.048
    y.L332        -0.0104      0.021     -0.487      0.626      -0.052       0.031
    y.L333         0.0194      0.021      0.908      0.364      -0.022       0.061
    y.L334        -0.0501      0.021     -2.346      0.019      -0.092      -0.008
    y.L335         0.0138      0.021      0.648      0.517      -0.028       0.056
    y.L336         0.0145      0.021      0.679      0.497      -0.027       0.056
    y.L337        -0.0182      0.021     -0.855      0.393      -0.060       0.024
    y.L338         0.0009      0.021      0.042      0.967      -0.041       0.043
    y.L339         0.0034      0.021      0.159      0.874      -0.038       0.045
    y.L340         0.0087      0.021      0.409      0.683      -0.033       0.050
    y.L341        -0.0354      0.021     -1.668      0.095      -0.077       0.006
    y.L342        -0.0042      0.021     -0.198      0.843      -0.046       0.037
    y.L343         0.0377      0.021      1.773      0.076      -0.004       0.079
    y.L344         0.0155      0.021      0.731      0.465      -0.026       0.057
    y.L345         0.0256      0.021      1.206      0.228      -0.016       0.067
    y.L346        -0.0082      0.021     -0.384      0.701      -0.050       0.033
    y.L347        -0.0263      0.021     -1.241      0.215      -0.068       0.015
    y.L348        -0.0307      0.021     -1.449      0.147      -0.072       0.011
    y.L349         0.0016      0.021      0.076      0.939      -0.040       0.043
    y.L350         0.0350      0.021      1.653      0.098      -0.007       0.076
    y.L351         0.0197      0.021      0.928      0.353      -0.022       0.061
    y.L352        -0.0373      0.021     -1.762      0.078      -0.079       0.004
    y.L353        -0.0006      0.021     -0.030      0.976      -0.042       0.041
    y.L354         0.0458      0.021      2.168      0.030       0.004       0.087
    y.L355        -0.0246      0.021     -1.163      0.245      -0.066       0.017
    y.L356        -0.0068      0.021     -0.320      0.749      -0.048       0.035
    y.L357        -0.0055      0.021     -0.259      0.795      -0.047       0.036
    y.L358         0.0656      0.021      3.102      0.002       0.024       0.107
    y.L359        -0.0222      0.021     -1.044      0.296      -0.064       0.019
    y.L360        -0.0318      0.021     -1.498      0.134      -0.074       0.010
    y.L361         0.0468      0.021      2.201      0.028       0.005       0.089
    y.L362         0.0281      0.021      1.320      0.187      -0.014       0.070
    y.L363         0.0230      0.021      1.079      0.280      -0.019       0.065
    y.L364        -0.0070      0.021     -0.327      0.744      -0.049       0.035
    y.L365         0.0004      0.021      0.018      0.986      -0.041       0.042
                                         Roots                                     
    ===============================================================================
                        Real          Imaginary           Modulus         Frequency
    -------------------------------------------------------------------------------
    AR.1              1.0000           -0.0674j            1.0023           -0.0107
    AR.2              1.0000           +0.0674j            1.0023            0.0107
    AR.3              1.0050           -0.0363j            1.0057           -0.0057
    AR.4              1.0050           +0.0363j            1.0057            0.0057
    AR.5              1.0076           -0.0487j            1.0087           -0.0077
    AR.6              1.0076           +0.0487j            1.0087            0.0077
    AR.7              1.0074           -0.0137j            1.0075           -0.0022
    AR.8              1.0074           +0.0137j            1.0075            0.0022
    AR.9              1.0131           -0.0000j            1.0131           -0.0000
    AR.10             1.0014           -0.0863j            1.0051           -0.0137
    AR.11             1.0014           +0.0863j            1.0051            0.0137
    AR.12             0.9986           -0.1049j            1.0041           -0.0167
    AR.13             0.9986           +0.1049j            1.0041            0.0167
    AR.14             0.9949           -0.1252j            1.0027           -0.0199
    AR.15             0.9949           +0.1252j            1.0027            0.0199
    AR.16             0.9965           -0.1439j            1.0068           -0.0228
    AR.17             0.9965           +0.1439j            1.0068            0.0228
    AR.18             0.9961           -0.1581j            1.0086           -0.0250
    AR.19             0.9961           +0.1581j            1.0086            0.0250
    AR.20             0.9885           -0.1761j            1.0040           -0.0281
    AR.21             0.9885           +0.1761j            1.0040            0.0281
    AR.22             0.7957           -0.6106j            1.0030           -0.1042
    AR.23             0.7957           +0.6106j            1.0030            0.1042
    AR.24             0.8059           -0.6009j            1.0053           -0.1020
    AR.25             0.8059           +0.6009j            1.0053            0.1020
    AR.26             0.8233           -0.5727j            1.0029           -0.0967
    AR.27             0.8233           +0.5727j            1.0029            0.0967
    AR.28             0.8377           -0.5588j            1.0070           -0.0936
    AR.29             0.8377           +0.5588j            1.0070            0.0936
    AR.30             0.8446           -0.5445j            1.0050           -0.0911
    AR.31             0.8446           +0.5445j            1.0050            0.0911
    AR.32             0.8537           -0.5259j            1.0027           -0.0879
    AR.33             0.8537           +0.5259j            1.0027            0.0879
    AR.34             0.8659           -0.5078j            1.0038           -0.0844
    AR.35             0.8659           +0.5078j            1.0038            0.0844
    AR.36             0.9744           -0.2402j            1.0036           -0.0385
    AR.37             0.9744           +0.2402j            1.0036            0.0385
    AR.38             0.9634           -0.2757j            1.0021           -0.0444
    AR.39             0.9634           +0.2757j            1.0021            0.0444
    AR.40             0.9727           -0.2547j            1.0054           -0.0408
    AR.41             0.9727           +0.2547j            1.0054            0.0408
    AR.42             0.7823           -0.6288j            1.0037           -0.1077
    AR.43             0.7823           +0.6288j            1.0037            0.1077
    AR.44             0.8211           -0.5944j            1.0137           -0.0997
    AR.45             0.8211           +0.5944j            1.0137            0.0997
    AR.46             0.8768           -0.4899j            1.0043           -0.0811
    AR.47             0.8768           +0.4899j            1.0043            0.0811
    AR.48             0.9828           -0.2167j            1.0064           -0.0345
    AR.49             0.9828           +0.2167j            1.0064            0.0345
    AR.50             0.8883           -0.4679j            1.0040           -0.0772
    AR.51             0.8883           +0.4679j            1.0040            0.0772
    AR.52             0.9604           -0.2976j            1.0055           -0.0478
    AR.53             0.9604           +0.2976j            1.0055            0.0478
    AR.54             0.8976           -0.4470j            1.0028           -0.0735
    AR.55             0.8976           +0.4470j            1.0028            0.0735
    AR.56             0.7701           -0.6453j            1.0047           -0.1110
    AR.57             0.7701           +0.6453j            1.0047            0.1110
    AR.58             0.7585           -0.6584j            1.0044           -0.1138
    AR.59             0.7585           +0.6584j            1.0044            0.1138
    AR.60             0.9066           -0.4303j            1.0036           -0.0705
    AR.61             0.9066           +0.4303j            1.0036            0.0705
    AR.62             0.9906           -0.1998j            1.0106           -0.0317
    AR.63             0.9906           +0.1998j            1.0106            0.0317
    AR.64             0.9565           -0.3161j            1.0074           -0.0508
    AR.65             0.9565           +0.3161j            1.0074            0.0508
    AR.66             0.9150           -0.4097j            1.0025           -0.0670
    AR.67             0.9150           +0.4097j            1.0025            0.0670
    AR.68             0.9488           -0.3290j            1.0042           -0.0531
    AR.69             0.9488           +0.3290j            1.0042            0.0531
    AR.70             0.7486           -0.6714j            1.0056           -0.1164
    AR.71             0.7486           +0.6714j            1.0056            0.1164
    AR.72             0.9242           -0.3923j            1.0040           -0.0639
    AR.73             0.9242           +0.3923j            1.0040            0.0639
    AR.74             0.7384           -0.6853j            1.0074           -0.1191
    AR.75             0.7384           +0.6853j            1.0074            0.1191
    AR.76             0.5831           -0.8163j            1.0032           -0.1513
    AR.77             0.5831           +0.8163j            1.0032            0.1513
    AR.78             0.6083           -0.7978j            1.0033           -0.1463
    AR.79             0.6083           +0.7978j            1.0033            0.1463
    AR.80             0.5983           -0.8074j            1.0050           -0.1485
    AR.81             0.5983           +0.8074j            1.0050            0.1485
    AR.82             0.6277           -0.7838j            1.0041           -0.1425
    AR.83             0.6277           +0.7838j            1.0041            0.1425
    AR.84             0.6410           -0.7747j            1.0055           -0.1400
    AR.85             0.6410           +0.7747j            1.0055            0.1400
    AR.86             0.9333           -0.3713j            1.0044           -0.0603
    AR.87             0.9333           +0.3713j            1.0044            0.0603
    AR.88             0.7255           -0.6981j            1.0068           -0.1219
    AR.89             0.7255           +0.6981j            1.0068            0.1219
    AR.90             0.6552           -0.7657j            1.0078           -0.1373
    AR.91             0.6552           +0.7657j            1.0078            0.1373
    AR.92             0.6735           -0.7434j            1.0031           -0.1328
    AR.93             0.6735           +0.7434j            1.0031            0.1328
    AR.94             0.5675           -0.8303j            1.0057           -0.1546
    AR.95             0.5675           +0.8303j            1.0057            0.1546
    AR.96             0.5250           -0.8539j            1.0024           -0.1623
    AR.97             0.5250           +0.8539j            1.0024            0.1623
    AR.98             0.5573           -0.8400j            1.0081           -0.1568
    AR.99             0.5573           +0.8400j            1.0081            0.1568
    AR.100            0.5066           -0.8644j            1.0019           -0.1656
    AR.101            0.5066           +0.8644j            1.0019            0.1656
    AR.102            0.5439           -0.8473j            1.0068           -0.1592
    AR.103            0.5439           +0.8473j            1.0068            0.1592
    AR.104            0.9460           -0.3456j            1.0072           -0.0558
    AR.105            0.9460           +0.3456j            1.0072            0.0558
    AR.106            0.6879           -0.7371j            1.0083           -0.1305
    AR.107            0.6879           +0.7371j            1.0083            0.1305
    AR.108            0.7127           -0.7126j            1.0079           -0.1250
    AR.109            0.7127           +0.7126j            1.0079            0.1250
    AR.110            0.4895           -0.8753j            1.0029           -0.1688
    AR.111            0.4895           +0.8753j            1.0029            0.1688
    AR.112            0.4741           -0.8825j            1.0018           -0.1715
    AR.113            0.4741           +0.8825j            1.0018            0.1715
    AR.114            0.4599           -0.8938j            1.0052           -0.1744
    AR.115            0.4599           +0.8938j            1.0052            0.1744
    AR.116            0.4424           -0.9025j            1.0051           -0.1775
    AR.117            0.4424           +0.9025j            1.0051            0.1775
    AR.118            0.4292           -0.9074j            1.0038           -0.1797
    AR.119            0.4292           +0.9074j            1.0038            0.1797
    AR.120            0.4047           -0.9172j            1.0025           -0.1839
    AR.121            0.4047           +0.9172j            1.0025            0.1839
    AR.122            0.3844           -0.9246j            1.0013           -0.1873
    AR.123            0.3844           +0.9246j            1.0013            0.1873
    AR.124            0.3623           -0.9363j            1.0040           -0.1912
    AR.125            0.3623           +0.9363j            1.0040            0.1912
    AR.126            0.7025           -0.7254j            1.0098           -0.1276
    AR.127            0.7025           +0.7254j            1.0098            0.1276
    AR.128            0.3458           -0.9435j            1.0049           -0.1941
    AR.129            0.3458           +0.9435j            1.0049            0.1941
    AR.130            0.3247           -0.9496j            1.0036           -0.1976
    AR.131            0.3247           +0.9496j            1.0036            0.1976
    AR.132            0.2195           -0.9788j            1.0031           -0.2149
    AR.133            0.2195           +0.9788j            1.0031            0.2149
    AR.134            0.2007           -0.9832j            1.0035           -0.2180
    AR.135            0.2007           +0.9832j            1.0035            0.2180
    AR.136            0.2397           -0.9750j            1.0040           -0.2116
    AR.137            0.2397           +0.9750j            1.0040            0.2116
    AR.138            0.1825           -0.9859j            1.0026           -0.2209
    AR.139            0.1825           +0.9859j            1.0026            0.2209
    AR.140            0.1683           -0.9889j            1.0031           -0.2232
    AR.141            0.1683           +0.9889j            1.0031            0.2232
    AR.142            0.1466           -0.9960j            1.0068           -0.2267
    AR.143            0.1466           +0.9960j            1.0068            0.2267
    AR.144            0.1326           -0.9956j            1.0044           -0.2289
    AR.145            0.1326           +0.9956j            1.0044            0.2289
    AR.146            0.1134           -0.9988j            1.0053           -0.2320
    AR.147            0.1134           +0.9988j            1.0053            0.2320
    AR.148            0.0991           -0.9990j            1.0039           -0.2343
    AR.149            0.0991           +0.9990j            1.0039            0.2343
    AR.150            0.0405           -1.0024j            1.0032           -0.2436
    AR.151            0.0405           +1.0024j            1.0032            0.2436
    AR.152            0.0798           -1.0023j            1.0055           -0.2373
    AR.153            0.0798           +1.0023j            1.0055            0.2373
    AR.154            0.0594           -1.0029j            1.0047           -0.2406
    AR.155            0.0594           +1.0029j            1.0047            0.2406
    AR.156            0.3032           -0.9575j            1.0044           -0.2012
    AR.157            0.3032           +0.9575j            1.0044            0.2012
    AR.158            0.2562           -0.9727j            1.0059           -0.2090
    AR.159            0.2562           +0.9727j            1.0059            0.2090
    AR.160            0.0076           -1.0042j            1.0042           -0.2488
    AR.161            0.0076           +1.0042j            1.0042            0.2488
    AR.162            0.0240           -1.0036j            1.0039           -0.2462
    AR.163            0.0240           +1.0036j            1.0039            0.2462
    AR.164            0.2760           -0.9685j            1.0071           -0.2058
    AR.165            0.2760           +0.9685j            1.0071            0.2058
    AR.166           -0.0109           -1.0046j            1.0047           -0.2517
    AR.167           -0.0109           +1.0046j            1.0047            0.2517
    AR.168           -0.0236           -1.0047j            1.0050           -0.2537
    AR.169           -0.0236           +1.0047j            1.0050            0.2537
    AR.170           -0.0424           -1.0031j            1.0040           -0.2567
    AR.171           -0.0424           +1.0031j            1.0040            0.2567
    AR.172           -0.0613           -1.0003j            1.0022           -0.2597
    AR.173           -0.0613           +1.0003j            1.0022            0.2597
    AR.174           -0.0772           -1.0008j            1.0038           -0.2623
    AR.175           -0.0772           +1.0008j            1.0038            0.2623
    AR.176           -0.0978           -0.9987j            1.0035           -0.2655
    AR.177           -0.0978           +0.9987j            1.0035            0.2655
    AR.178           -0.1122           -0.9969j            1.0032           -0.2678
    AR.179           -0.1122           +0.9969j            1.0032            0.2678
    AR.180           -0.1328           -0.9940j            1.0028           -0.2711
    AR.181           -0.1328           +0.9940j            1.0028            0.2711
    AR.182           -0.1536           -0.9896j            1.0015           -0.2745
    AR.183           -0.1536           +0.9896j            1.0015            0.2745
    AR.184           -0.1692           -0.9910j            1.0054           -0.2769
    AR.185           -0.1692           +0.9910j            1.0054            0.2769
    AR.186           -0.1885           -0.9860j            1.0038           -0.2801
    AR.187           -0.1885           +0.9860j            1.0038            0.2801
    AR.188           -0.2067           -0.9809j            1.0025           -0.2831
    AR.189           -0.2067           +0.9809j            1.0025            0.2831
    AR.190           -0.2256           -0.9823j            1.0079           -0.2859
    AR.191           -0.2256           +0.9823j            1.0079            0.2859
    AR.192           -0.2329           -0.9792j            1.0065           -0.2872
    AR.193           -0.2329           +0.9792j            1.0065            0.2872
    AR.194           -0.2565           -0.9697j            1.0031           -0.2912
    AR.195           -0.2565           +0.9697j            1.0031            0.2912
    AR.196           -0.2703           -0.9659j            1.0030           -0.2934
    AR.197           -0.2703           +0.9659j            1.0030            0.2934
    AR.198           -0.7758           -0.6374j            1.0040           -0.3905
    AR.199           -0.7758           +0.6374j            1.0040            0.3905
    AR.200           -0.7911           -0.6180j            1.0038           -0.3945
    AR.201           -0.7911           +0.6180j            1.0038            0.3945
    AR.202           -0.8000           -0.6119j            1.0072           -0.3961
    AR.203           -0.8000           +0.6119j            1.0072            0.3961
    AR.204           -0.8098           -0.5946j            1.0047           -0.3992
    AR.205           -0.8098           +0.5946j            1.0047            0.3992
    AR.206           -0.8207           -0.5801j            1.0051           -0.4021
    AR.207           -0.8207           +0.5801j            1.0051            0.4021
    AR.208           -0.8290           -0.5638j            1.0026           -0.4049
    AR.209           -0.8290           +0.5638j            1.0026            0.4049
    AR.210           -0.7612           -0.6537j            1.0033           -0.3871
    AR.211           -0.7612           +0.6537j            1.0033            0.3871
    AR.212           -0.8447           -0.5537j            1.0100           -0.4077
    AR.213           -0.8447           +0.5537j            1.0100            0.4077
    AR.214           -0.8502           -0.5332j            1.0035           -0.4109
    AR.215           -0.8502           +0.5332j            1.0035            0.4109
    AR.216           -0.8608           -0.5259j            1.0087           -0.4127
    AR.217           -0.8608           +0.5259j            1.0087            0.4127
    AR.218           -0.8670           -0.5044j            1.0030           -0.4161
    AR.219           -0.8670           +0.5044j            1.0030            0.4161
    AR.220           -0.9183           -0.3980j            1.0009           -0.4349
    AR.221           -0.9183           +0.3980j            1.0009            0.4349
    AR.222           -0.9129           -0.4172j            1.0038           -0.4318
    AR.223           -0.9129           +0.4172j            1.0038            0.4318
    AR.224           -0.9097           -0.4320j            1.0071           -0.4294
    AR.225           -0.9097           +0.4320j            1.0071            0.4294
    AR.226           -0.9012           -0.4482j            1.0065           -0.4265
    AR.227           -0.9012           +0.4482j            1.0065            0.4265
    AR.228           -0.9289           -0.3784j            1.0030           -0.4384
    AR.229           -0.9289           +0.3784j            1.0030            0.4384
    AR.230           -0.9332           -0.3643j            1.0018           -0.4408
    AR.231           -0.9332           +0.3643j            1.0018            0.4408
    AR.232           -0.8901           -0.4665j            1.0050           -0.4232
    AR.233           -0.8901           +0.4665j            1.0050            0.4232
    AR.234           -0.9432           -0.3435j            1.0038           -0.4444
    AR.235           -0.9432           +0.3435j            1.0038            0.4444
    AR.236           -0.7582           -0.6670j            1.0099           -0.3852
    AR.237           -0.7582           +0.6670j            1.0099            0.3852
    AR.238           -0.7468           -0.6745j            1.0063           -0.3831
    AR.239           -0.7468           +0.6745j            1.0063            0.3831
    AR.240           -0.8872           -0.4765j            1.0071           -0.4216
    AR.241           -0.8872           +0.4765j            1.0071            0.4216
    AR.242           -0.7288           -0.6883j            1.0024           -0.3795
    AR.243           -0.7288           +0.6883j            1.0024            0.3795
    AR.244           -0.9534           -0.3240j            1.0070           -0.4479
    AR.245           -0.9534           +0.3240j            1.0070            0.4479
    AR.246           -0.4354           -0.9045j            1.0038           -0.3214
    AR.247           -0.4354           +0.9045j            1.0038            0.3214
    AR.248           -0.4548           -0.8964j            1.0052           -0.3247
    AR.249           -0.4548           +0.8964j            1.0052            0.3247
    AR.250           -0.4675           -0.8873j            1.0029           -0.3272
    AR.251           -0.4675           +0.8873j            1.0029            0.3272
    AR.252           -0.9563           -0.3164j            1.0072           -0.4491
    AR.253           -0.9563           +0.3164j            1.0072            0.4491
    AR.254           -0.7157           -0.7054j            1.0049           -0.3762
    AR.255           -0.7157           +0.7054j            1.0049            0.3762
    AR.256           -0.9622           -0.2980j            1.0072           -0.4522
    AR.257           -0.9622           +0.2980j            1.0072            0.4522
    AR.258           -0.4163           -0.9144j            1.0047           -0.3180
    AR.259           -0.4163           +0.9144j            1.0047            0.3180
    AR.260           -0.4863           -0.8822j            1.0074           -0.3302
    AR.261           -0.4863           +0.8822j            1.0074            0.3302
    AR.262           -0.3881           -0.9264j            1.0044           -0.3131
    AR.263           -0.3881           +0.9264j            1.0044            0.3131
    AR.264           -0.4038           -0.9275j            1.0116           -0.3154
    AR.265           -0.4038           +0.9275j            1.0116            0.3154
    AR.266           -0.9652           -0.2792j            1.0048           -0.4552
    AR.267           -0.9652           +0.2792j            1.0048            0.4552
    AR.268           -0.5044           -0.8676j            1.0035           -0.3338
    AR.269           -0.5044           +0.8676j            1.0035            0.3338
    AR.270           -0.7049           -0.7197j            1.0073           -0.3733
    AR.271           -0.7049           +0.7197j            1.0073            0.3733
    AR.272           -0.6949           -0.7310j            1.0086           -0.3710
    AR.273           -0.6949           +0.7310j            1.0086            0.3710
    AR.274           -0.9710           -0.2579j            1.0047           -0.4587
    AR.275           -0.9710           +0.2579j            1.0047            0.4587
    AR.276           -0.9764           -0.2328j            1.0038           -0.4627
    AR.277           -0.9764           +0.2328j            1.0038            0.4627
    AR.278           -0.5241           -0.8587j            1.0060           -0.3372
    AR.279           -0.5241           +0.8587j            1.0060            0.3372
    AR.280           -0.3677           -0.9375j            1.0071           -0.3095
    AR.281           -0.3677           +0.9375j            1.0071            0.3095
    AR.282           -0.9826           -0.2198j            1.0069           -0.4650
    AR.283           -0.9826           +0.2198j            1.0069            0.4650
    AR.284           -0.2950           -0.9649j            1.0090           -0.2972
    AR.285           -0.2950           +0.9649j            1.0090            0.2972
    AR.286           -0.9823           -0.1963j            1.0017           -0.4686
    AR.287           -0.9823           +0.1963j            1.0017            0.4686
    AR.288           -0.6787           -0.7416j            1.0053           -0.3679
    AR.289           -0.6787           +0.7416j            1.0053            0.3679
    AR.290           -0.9893           -0.1766j            1.0049           -0.4719
    AR.291           -0.9893           +0.1766j            1.0049            0.4719
    AR.292           -0.5551           -0.8393j            1.0063           -0.3430
    AR.293           -0.5551           +0.8393j            1.0063            0.3430
    AR.294           -0.5399           -0.8537j            1.0101           -0.3397
    AR.295           -0.5399           +0.8537j            1.0101            0.3397
    AR.296           -0.5736           -0.8285j            1.0077           -0.3464
    AR.297           -0.5736           +0.8285j            1.0077            0.3464
    AR.298           -0.9896           -0.1583j            1.0022           -0.4748
    AR.299           -0.9896           +0.1583j            1.0022            0.4748
    AR.300           -0.5835           -0.8235j            1.0093           -0.3481
    AR.301           -0.5835           +0.8235j            1.0093            0.3481
    AR.302           -0.3568           -0.9430j            1.0082           -0.3076
    AR.303           -0.3568           +0.9430j            1.0082            0.3076
    AR.304           -0.6001           -0.8040j            1.0033           -0.3521
    AR.305           -0.6001           +0.8040j            1.0033            0.3521
    AR.306           -0.9990           -0.1441j            1.0094           -0.4772
    AR.307           -0.9990           +0.1441j            1.0094            0.4772
    AR.308           -0.6645           -0.7561j            1.0066           -0.3648
    AR.309           -0.6645           +0.7561j            1.0066            0.3648
    AR.310           -0.9952           -0.1217j            1.0026           -0.4806
    AR.311           -0.9952           +0.1217j            1.0026            0.4806
    AR.312            0.9506           -0.3576j            1.0157           -0.0573
    AR.313            0.9506           +0.3576j            1.0157            0.0573
    AR.314            0.2841           -0.9728j            1.0135           -0.2048
    AR.315            0.2841           +0.9728j            1.0135            0.2048
    AR.316           -0.3216           -0.9551j            1.0078           -0.3017
    AR.317           -0.3216           +0.9551j            1.0078            0.3017
    AR.318           -0.9998           -0.1011j            1.0049           -0.4840
    AR.319           -0.9998           +0.1011j            1.0049            0.4840
    AR.320           -0.3399           -0.9487j            1.0078           -0.3048
    AR.321           -0.3399           +0.9487j            1.0078            0.3048
    AR.322           -0.6428           -0.7716j            1.0043           -0.3605
    AR.323           -0.6428           +0.7716j            1.0043            0.3605
    AR.324           -0.6259           -0.7911j            1.0087           -0.3565
    AR.325           -0.6259           +0.7911j            1.0087            0.3565
    AR.326           -0.6251           -0.7978j            1.0136           -0.3558
    AR.327           -0.6251           +0.7978j            1.0136            0.3558
    AR.328           -0.3081           -0.9593j            1.0075           -0.2995
    AR.329           -0.3081           +0.9593j            1.0075            0.2995
    AR.330           -1.0046           -0.0803j            1.0078           -0.4873
    AR.331           -1.0046           +0.0803j            1.0078            0.4873
    AR.332           -1.0061           -0.0169j            1.0062           -0.4973
    AR.333           -1.0061           +0.0169j            1.0062            0.4973
    AR.334           -1.0060           -0.0544j            1.0074           -0.4914
    AR.335           -1.0060           +0.0544j            1.0074            0.4914
    AR.336           -0.8858           -0.4936j            1.0141           -0.4191
    AR.337           -0.8858           +0.4936j            1.0141            0.4191
    AR.338           -1.0095           -0.0389j            1.0103           -0.4939
    AR.339           -1.0095           +0.0389j            1.0103            0.4939
    AR.340            0.9295           -0.4441j            1.0301           -0.0709
    AR.341            0.9295           +0.4441j            1.0301            0.0709
    AR.342            1.0001           -0.2115j            1.0222           -0.0332
    AR.343            1.0001           +0.2115j            1.0222            0.0332
    AR.344           -0.6602           -0.7739j            1.0173           -0.3624
    AR.345           -0.6602           +0.7739j            1.0173            0.3624
    AR.346           -1.0273           -0.0000j            1.0273           -0.5000
    AR.347           -0.9895           -0.2528j            1.0213           -0.4602
    AR.348           -0.9895           +0.2528j            1.0213            0.4602
    AR.349           -1.0290           -0.0712j            1.0314           -0.4890
    AR.350           -1.0290           +0.0712j            1.0314            0.4890
    AR.351            0.3468           -0.9705j            1.0306           -0.1954
    AR.352            0.3468           +0.9705j            1.0306            0.1954
    AR.353            0.8945           -0.5237j            1.0365           -0.0843
    AR.354            0.8945           +0.5237j            1.0365            0.0843
    AR.355            0.4147           -0.9441j            1.0311           -0.1841
    AR.356            0.4147           +0.9441j            1.0311            0.1841
    AR.357           -0.5414           -0.8867j            1.0389           -0.3372
    AR.358           -0.5414           +0.8867j            1.0389            0.3372
    AR.359            0.7081           -0.7905j            1.0613           -0.1337
    AR.360            0.7081           +0.7905j            1.0613            0.1337
    AR.361           -1.0961           -0.0000j            1.0961           -0.5000
    AR.362           -0.7748           -1.3433j            1.5507           -0.3333
    AR.363           -0.7748           +1.3433j            1.5507            0.3333
    AR.364            6.1466           -0.0000j            6.1466           -0.0000
    AR.365           13.6813           -0.0000j           13.6813           -0.0000
    -------------------------------------------------------------------------------
    μ=0.35317353294416043 ,ϕ=[ 1.98381693e-02  1.15551378e-02 -2.38222688e-02 -2.63066504e-02
     -1.33832958e-02  3.15840442e-03  1.02311209e-02 -2.29319005e-02
     -1.91226355e-04  4.73978025e-03  4.26506965e-03 -2.54360512e-03
     -1.92709163e-02 -1.07091912e-03 -2.18036925e-03  5.51565464e-03
      4.29130414e-02 -1.89061517e-02  3.39115848e-02  6.41116555e-02
     -5.90841232e-03 -2.72339556e-02 -3.14897597e-02  5.81329856e-03
     -5.46280186e-02 -2.15633840e-02  1.55992045e-02  4.06504739e-02
     -4.77864240e-02 -2.97282086e-02  8.09433746e-03  3.05789344e-02
      2.79794482e-02 -1.63831249e-02  3.45000596e-02  1.65280258e-02
     -5.26071791e-02 -3.67989901e-02  2.82297146e-02  1.15042935e-02
      1.95027792e-02  1.68828419e-02 -1.67665082e-02 -1.03186611e-02
      4.70948823e-03 -6.55729932e-03  5.71487579e-03 -2.82218063e-02
     -1.91116902e-02 -4.31381927e-02 -4.89787229e-02  1.81787475e-02
      2.63945722e-02 -1.10957940e-02  2.44039950e-02  1.72061752e-02
      4.08758426e-02  1.19918893e-02  8.83932012e-03  4.50905336e-03
      1.34469888e-03 -2.56400267e-02 -4.69782536e-02 -4.34443972e-02
     -2.20663904e-02 -6.07959966e-02 -1.61811232e-02  6.04560109e-03
      2.41405953e-02  5.58691362e-03  1.09245033e-02 -1.62249303e-02
      1.79220525e-02 -2.63189533e-02 -2.10279198e-02  1.61117456e-02
      3.28867490e-02  3.70145472e-02 -4.31730208e-02  9.82507152e-03
      1.63563924e-03  3.37628264e-03  1.69390949e-02 -4.22015382e-02
      1.40014323e-02  2.52795100e-02  4.02688845e-03 -4.19307702e-02
     -2.30420677e-03 -1.29897743e-02  1.79624359e-02 -8.13525923e-03
      6.00882478e-03 -3.78675796e-03  5.06645445e-02  4.94593090e-03
     -7.13210858e-03  1.38062372e-02  4.09234408e-02 -2.08155822e-02
     -1.03047578e-02 -8.88535219e-03 -2.69121017e-02  9.42606317e-03
     -1.40008815e-02  3.87585148e-02  4.94163408e-02 -4.99556405e-03
      1.56060495e-02 -3.14107057e-02 -1.17755836e-02 -5.28339839e-03
      1.81754887e-02  3.38162301e-02 -5.68037409e-02  5.70432421e-02
     -2.30073424e-02 -3.56180722e-02  6.28197445e-03  6.65268921e-03
     -1.86343701e-02  5.17138600e-03  3.36845805e-02 -1.28908390e-03
     -4.06706457e-02 -3.37202016e-02  2.96145735e-02 -5.23596549e-03
     -4.24215691e-04  1.12339574e-05 -8.42272781e-03 -3.17768886e-02
     -4.10166521e-02  4.53453547e-02  3.17591604e-02  2.04492555e-02
      5.56103199e-03 -3.42887569e-02  2.34162471e-04 -2.44972265e-02
      3.73602849e-03 -2.81204006e-02  2.00958795e-03  5.19672016e-03
      2.23478399e-02  6.41907161e-05  2.13205932e-02 -4.15955648e-02
      1.19259966e-02  4.56181682e-02 -4.27884769e-03 -5.97043683e-03
      2.94823758e-02  9.00351393e-03 -3.81984164e-02  3.23738900e-02
      2.36854510e-02  2.69613417e-02 -2.44538672e-02 -1.87402595e-02
      9.53364606e-04 -1.34958874e-02 -1.23851518e-02 -6.53072677e-03
      6.51726809e-04  2.14326045e-02 -9.98996118e-03  1.95016100e-02
     -4.66055633e-02 -1.23450968e-02  2.00873956e-02 -2.04580710e-02
      2.04909321e-02 -1.31771141e-02  5.64485356e-02 -2.80263958e-02
      2.84884881e-02 -4.77916841e-03  2.74830352e-02  2.38728751e-02
      2.26843565e-02  3.59409936e-03  1.86893421e-02  4.30234147e-03
      5.08649385e-02 -2.72272321e-02  1.09211336e-02  2.10721160e-03
     -1.49877089e-02 -1.04059523e-02 -3.48949398e-02  7.06642524e-03
     -8.30426899e-03 -2.02123500e-02  1.50568710e-02  2.70419500e-03
      8.95786704e-03  2.96274485e-02 -1.93961398e-02 -5.94954601e-03
     -2.69289412e-02  3.03499322e-02  3.46029764e-03  1.85898660e-02
      4.02433018e-02 -1.82853008e-02  4.00739571e-02 -1.33745344e-02
     -2.24424745e-04  5.17667290e-03  1.87317544e-02 -2.71698337e-02
      8.44559163e-04  4.44968332e-03 -3.92804748e-02 -5.38730029e-04
     -1.53219129e-02  3.26492355e-02  1.27737771e-02 -3.58929709e-02
     -3.07041933e-02 -1.88584222e-02  6.20525403e-03 -4.90012206e-03
      1.44827944e-02 -3.80352871e-02  2.15466836e-03 -4.85496844e-03
      6.11602906e-03  6.88905017e-03 -2.46975043e-02 -2.32675916e-02
      3.91300794e-02 -1.02303646e-02 -2.97213159e-02 -9.16538299e-03
     -1.20423247e-02 -2.92293919e-02  1.56578486e-02  1.23341124e-03
     -3.37423540e-03 -2.74603785e-02 -1.99258099e-02 -1.92635902e-02
      2.13082536e-03  2.37412597e-02  7.89670669e-03  3.86629818e-02
     -1.89409776e-02  2.60251031e-02  8.13574739e-03  9.88299566e-03
     -7.53783582e-03 -1.32271836e-02  1.00311834e-02 -7.40047733e-03
     -3.45025999e-02  3.25860405e-02 -5.48594212e-03 -3.41838706e-02
      3.08739500e-02  1.35834271e-03 -3.50895614e-02 -1.87645877e-02
     -2.29365367e-02  6.28942587e-03 -1.10348663e-03 -7.68478101e-03
      3.43593526e-02 -2.69472718e-02  2.91850812e-02  2.55636269e-02
      7.52912881e-03 -1.08873434e-02 -5.48691863e-02 -2.41721010e-02
      1.16195549e-02 -1.43339385e-02  2.14954252e-02 -5.78640279e-02
     -1.51707845e-02 -7.54517732e-03  3.21680323e-02  3.33311430e-02
      1.15242408e-02 -4.42770383e-03  6.67884377e-03  2.19175292e-02
      2.74891632e-03  5.59127578e-03 -5.25357420e-02  1.81722026e-03
     -2.04289055e-03  3.19088485e-03  2.67639308e-02  2.30496379e-02
      4.27769041e-02  1.27272363e-02  5.39734124e-02 -1.11582114e-02
     -2.12994548e-02 -2.27629908e-02  1.09494779e-02 -3.57062807e-02
     -4.17795624e-02 -1.80156562e-02 -1.64135747e-02  4.21423766e-02
      1.27503773e-02  1.74350300e-02 -2.20021171e-02  8.80473138e-04
      8.55683476e-03  1.66564550e-02  1.78572075e-02  1.34904741e-02
      8.77109488e-03 -9.93191503e-03 -4.11094034e-02  2.86297564e-02
      2.34728965e-02 -3.30749795e-03 -4.92024709e-02  1.48022136e-02
     -2.86441849e-02 -5.80453859e-03 -2.24499014e-03 -3.08689382e-02
     -1.11532206e-02 -3.93029730e-02  6.68874317e-03 -1.03870800e-02
      1.93764738e-02 -5.00819502e-02  1.38284758e-02  1.44887354e-02
     -1.82292090e-02  8.90267176e-04  3.37617997e-03  8.69317096e-03
     -3.54076436e-02 -4.20394688e-03  3.76503474e-02  1.55320680e-02
      2.56260994e-02 -8.15024478e-03 -2.63006079e-02 -3.07301149e-02
      1.60861764e-03  3.49968041e-02  1.96506066e-02 -3.72990092e-02
     -6.32353459e-04  4.58287999e-02 -2.45983366e-02 -6.76827159e-03
     -5.48015586e-03  6.56385677e-02 -2.21975177e-02 -3.18407131e-02
      4.68180731e-02  2.80954561e-02  2.29693541e-02 -6.95270756e-03
      3.73238618e-04]
    

- The number of past periods = 730


```python
# Predicting closing prices of google
humid = AutoReg(google_stock["Close"].diff().iloc[1:].values, lags = 730, old_names=False)
res = humid.fit()
res.plot_predict(start=900, end=1010, figsize=(15,5))
plt.show()
print(res.summary())
print("μ={} ,ϕ={}".format(res.params[0],res.params[1:]))
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_231_0.png)
    


                                AutoReg Model Results                             
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 3018
    Model:                   AutoReg(730)   Log Likelihood               -7269.816
    Method:               Conditional MLE   S.D. of innovations              5.803
    Date:                Fri, 17 Oct 2025   AIC                          16003.631
    Time:                        15:55:20   BIC                          20201.968
    Sample:                           730   HQIC                         17534.699
                                     3018                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.3326      0.192      1.729      0.084      -0.044       0.710
    y.L1           0.0259      0.021      1.238      0.216      -0.015       0.067
    y.L2           0.0150      0.021      0.718      0.473      -0.026       0.056
    y.L3          -0.0284      0.021     -1.357      0.175      -0.069       0.013
    y.L4          -0.0238      0.021     -1.140      0.254      -0.065       0.017
    y.L5          -0.0094      0.021     -0.450      0.652      -0.050       0.032
    y.L6          -0.0019      0.021     -0.090      0.928      -0.043       0.039
    y.L7           0.0140      0.021      0.672      0.502      -0.027       0.055
    y.L8          -0.0382      0.021     -1.832      0.067      -0.079       0.003
    y.L9           0.0072      0.021      0.347      0.728      -0.034       0.048
    y.L10          0.0014      0.021      0.069      0.945      -0.039       0.042
    y.L11          0.0031      0.021      0.151      0.880      -0.038       0.044
    y.L12         -0.0107      0.021     -0.516      0.606      -0.052       0.030
    y.L13         -0.0335      0.021     -1.608      0.108      -0.074       0.007
    y.L14          0.0041      0.021      0.197      0.844      -0.037       0.045
    y.L15         -0.0142      0.021     -0.682      0.495      -0.055       0.027
    y.L16          0.0050      0.021      0.240      0.811      -0.036       0.046
    y.L17          0.0494      0.021      2.378      0.017       0.009       0.090
    y.L18         -0.0099      0.021     -0.473      0.636      -0.051       0.031
    y.L19          0.0185      0.021      0.890      0.373      -0.022       0.059
    y.L20          0.0566      0.021      2.717      0.007       0.016       0.097
    y.L21         -0.0053      0.021     -0.252      0.801      -0.046       0.036
    y.L22         -0.0358      0.021     -1.721      0.085      -0.077       0.005
    y.L23         -0.0297      0.021     -1.424      0.154      -0.071       0.011
    y.L24          0.0150      0.021      0.719      0.472      -0.026       0.056
    y.L25         -0.0741      0.021     -3.554      0.000      -0.115      -0.033
    y.L26         -0.0301      0.021     -1.440      0.150      -0.071       0.011
    y.L27          0.0028      0.021      0.132      0.895      -0.038       0.044
    y.L28          0.0432      0.021      2.066      0.039       0.002       0.084
    y.L29         -0.0530      0.021     -2.536      0.011      -0.094      -0.012
    y.L30         -0.0321      0.021     -1.534      0.125      -0.073       0.009
    y.L31          0.0125      0.021      0.595      0.552      -0.029       0.054
    y.L32          0.0268      0.021      1.285      0.199      -0.014       0.068
    y.L33          0.0434      0.021      2.079      0.038       0.002       0.084
    y.L34         -0.0173      0.021     -0.830      0.407      -0.058       0.024
    y.L35          0.0031      0.021      0.146      0.884      -0.038       0.044
    y.L36          0.0150      0.021      0.717      0.473      -0.026       0.056
    y.L37         -0.0421      0.021     -2.015      0.044      -0.083      -0.001
    y.L38         -0.0556      0.021     -2.656      0.008      -0.097      -0.015
    y.L39          0.0567      0.021      2.708      0.007       0.016       0.098
    y.L40         -0.0032      0.021     -0.151      0.880      -0.044       0.038
    y.L41          0.0452      0.021      2.155      0.031       0.004       0.086
    y.L42          0.0091      0.021      0.437      0.662      -0.032       0.050
    y.L43         -0.0067      0.021     -0.319      0.749      -0.048       0.034
    y.L44         -0.0159      0.021     -0.755      0.450      -0.057       0.025
    y.L45          0.0123      0.021      0.586      0.558      -0.029       0.053
    y.L46         -0.0135      0.021     -0.642      0.521      -0.055       0.028
    y.L47          0.0337      0.021      1.602      0.109      -0.008       0.075
    y.L48         -0.0529      0.021     -2.508      0.012      -0.094      -0.012
    y.L49         -0.0289      0.021     -1.374      0.169      -0.070       0.012
    y.L50         -0.0662      0.021     -3.141      0.002      -0.107      -0.025
    y.L51         -0.0286      0.021     -1.353      0.176      -0.070       0.013
    y.L52          0.0393      0.021      1.863      0.062      -0.002       0.081
    y.L53         -0.0099      0.021     -0.469      0.639      -0.051       0.031
    y.L54       2.531e-05      0.021      0.001      0.999      -0.041       0.041
    y.L55          0.0109      0.021      0.518      0.605      -0.030       0.052
    y.L56          0.0110      0.021      0.523      0.601      -0.030       0.052
    y.L57          0.0311      0.021      1.478      0.139      -0.010       0.072
    y.L58          0.0193      0.021      0.918      0.359      -0.022       0.061
    y.L59          0.0303      0.021      1.438      0.150      -0.011       0.072
    y.L60          0.0073      0.021      0.347      0.728      -0.034       0.049
    y.L61         -0.0131      0.021     -0.621      0.534      -0.054       0.028
    y.L62         -0.0018      0.021     -0.086      0.931      -0.043       0.040
    y.L63         -0.0399      0.021     -1.890      0.059      -0.081       0.001
    y.L64         -0.0241      0.021     -1.143      0.253      -0.066       0.017
    y.L65         -0.0297      0.021     -1.407      0.159      -0.071       0.012
    y.L66         -0.0654      0.021     -3.088      0.002      -0.107      -0.024
    y.L67         -0.0138      0.021     -0.652      0.514      -0.055       0.028
    y.L68          0.0082      0.021      0.385      0.700      -0.033       0.050
    y.L69          0.0411      0.021      1.939      0.052      -0.000       0.083
    y.L70         -0.0176      0.021     -0.828      0.407      -0.059       0.024
    y.L71          0.0189      0.021      0.891      0.373      -0.023       0.060
    y.L72          0.0090      0.021      0.427      0.670      -0.033       0.051
    y.L73          0.0193      0.021      0.909      0.363      -0.022       0.061
    y.L74         -0.0238      0.021     -1.125      0.261      -0.065       0.018
    y.L75         -0.0183      0.021     -0.864      0.388      -0.060       0.023
    y.L76          0.0054      0.021      0.256      0.798      -0.036       0.047
    y.L77          0.0567      0.021      2.670      0.008       0.015       0.098
    y.L78          0.0275      0.021      1.293      0.196      -0.014       0.069
    y.L79         -0.0373      0.021     -1.754      0.079      -0.079       0.004
    y.L80         -0.0258      0.021     -1.211      0.226      -0.068       0.016
    y.L81          0.0272      0.021      1.277      0.202      -0.015       0.069
    y.L82          0.0057      0.021      0.269      0.788      -0.036       0.048
    y.L83          0.0331      0.021      1.554      0.120      -0.009       0.075
    y.L84         -0.0635      0.021     -2.977      0.003      -0.105      -0.022
    y.L85          0.0254      0.021      1.190      0.234      -0.016       0.067
    y.L86          0.0134      0.021      0.627      0.530      -0.028       0.055
    y.L87          0.0018      0.021      0.084      0.933      -0.040       0.044
    y.L88         -0.0288      0.021     -1.347      0.178      -0.071       0.013
    y.L89          0.0218      0.021      1.019      0.308      -0.020       0.064
    y.L90         -0.0274      0.021     -1.283      0.199      -0.069       0.014
    y.L91          0.0160      0.021      0.748      0.454      -0.026       0.058
    y.L92         -0.0019      0.021     -0.089      0.929      -0.044       0.040
    y.L93          0.0068      0.021      0.316      0.752      -0.035       0.049
    y.L94          0.0154      0.021      0.719      0.472      -0.027       0.057
    y.L95          0.0301      0.021      1.409      0.159      -0.012       0.072
    y.L96          0.0078      0.021      0.366      0.714      -0.034       0.050
    y.L97         -0.0347      0.021     -1.622      0.105      -0.077       0.007
    y.L98          0.0077      0.021      0.359      0.719      -0.034       0.050
    y.L99          0.0252      0.021      1.179      0.238      -0.017       0.067
    y.L100        -0.0158      0.021     -0.742      0.458      -0.058       0.026
    y.L101        -0.0236      0.021     -1.103      0.270      -0.065       0.018
    y.L102        -0.0200      0.021     -0.937      0.349      -0.062       0.022
    y.L103        -0.0420      0.021     -1.969      0.049      -0.084      -0.000
    y.L104         0.0021      0.021      0.097      0.923      -0.040       0.044
    y.L105        -0.0278      0.021     -1.299      0.194      -0.070       0.014
    y.L106         0.0571      0.021      2.670      0.008       0.015       0.099
    y.L107         0.0555      0.021      2.593      0.010       0.014       0.097
    y.L108         0.0087      0.021      0.407      0.684      -0.033       0.051
    y.L109         0.0064      0.021      0.298      0.766      -0.036       0.048
    y.L110        -0.0227      0.021     -1.061      0.289      -0.065       0.019
    y.L111        -0.0217      0.021     -1.012      0.311      -0.064       0.020
    y.L112         0.0136      0.021      0.637      0.524      -0.028       0.056
    y.L113         0.0095      0.021      0.442      0.659      -0.032       0.051
    y.L114         0.0184      0.021      0.860      0.390      -0.024       0.060
    y.L115        -0.0552      0.021     -2.582      0.010      -0.097      -0.013
    y.L116         0.0421      0.021      1.968      0.049       0.000       0.084
    y.L117        -0.0152      0.021     -0.710      0.478      -0.057       0.027
    y.L118        -0.0500      0.021     -2.335      0.020      -0.092      -0.008
    y.L119         0.0068      0.021      0.317      0.752      -0.035       0.049
    y.L120        -0.0208      0.021     -0.969      0.333      -0.063       0.021
    y.L121        -0.0184      0.021     -0.858      0.391      -0.060       0.024
    y.L122         0.0079      0.021      0.368      0.713      -0.034       0.050
    y.L123         0.0382      0.021      1.782      0.075      -0.004       0.080
    y.L124        -0.0229      0.021     -1.068      0.286      -0.065       0.019
    y.L125        -0.0219      0.021     -1.022      0.307      -0.064       0.020
    y.L126        -0.0366      0.021     -1.709      0.087      -0.079       0.005
    y.L127         0.0163      0.021      0.759      0.448      -0.026       0.058
    y.L128        -0.0381      0.021     -1.775      0.076      -0.080       0.004
    y.L129        -0.0043      0.021     -0.198      0.843      -0.046       0.038
    y.L130         0.0014      0.021      0.064      0.949      -0.041       0.043
    y.L131        -0.0101      0.022     -0.471      0.638      -0.052       0.032
    y.L132        -0.0323      0.021     -1.502      0.133      -0.074       0.010
    y.L133        -0.0528      0.021     -2.456      0.014      -0.095      -0.011
    y.L134         0.0393      0.022      1.826      0.068      -0.003       0.082
    y.L135         0.0376      0.022      1.747      0.081      -0.005       0.080
    y.L136         0.0323      0.022      1.493      0.135      -0.010       0.075
    y.L137         0.0156      0.022      0.723      0.470      -0.027       0.058
    y.L138        -0.0141      0.022     -0.654      0.513      -0.057       0.028
    y.L139         0.0010      0.022      0.045      0.964      -0.041       0.043
    y.L140        -0.0468      0.022     -2.167      0.030      -0.089      -0.004
    y.L141        -0.0026      0.022     -0.120      0.905      -0.045       0.040
    y.L142        -0.0225      0.022     -1.041      0.298      -0.065       0.020
    y.L143         0.0009      0.022      0.043      0.966      -0.041       0.043
    y.L144         0.0163      0.022      0.755      0.450      -0.026       0.059
    y.L145         0.0052      0.022      0.242      0.809      -0.037       0.048
    y.L146        -0.0273      0.022     -1.262      0.207      -0.070       0.015
    y.L147         0.0125      0.022      0.580      0.562      -0.030       0.055
    y.L148        -0.0271      0.022     -1.253      0.210      -0.069       0.015
    y.L149         0.0365      0.022      1.690      0.091      -0.006       0.079
    y.L150         0.0287      0.022      1.327      0.184      -0.014       0.071
    y.L151        -0.0274      0.022     -1.268      0.205      -0.070       0.015
    y.L152        -0.0140      0.022     -0.645      0.519      -0.056       0.028
    y.L153         0.0078      0.022      0.361      0.718      -0.035       0.050
    y.L154         0.0052      0.022      0.239      0.811      -0.037       0.048
    y.L155        -0.0178      0.021     -0.829      0.407      -0.060       0.024
    y.L156         0.0385      0.022      1.788      0.074      -0.004       0.081
    y.L157         0.0119      0.022      0.551      0.582      -0.030       0.054
    y.L158         0.0055      0.022      0.255      0.799      -0.037       0.048
    y.L159        -0.0095      0.022     -0.442      0.659      -0.052       0.033
    y.L160        -0.0236      0.022     -1.093      0.274      -0.066       0.019
    y.L161         0.0189      0.022      0.875      0.382      -0.023       0.061
    y.L162        -0.0026      0.022     -0.120      0.905      -0.045       0.040
    y.L163        -0.0213      0.022     -0.989      0.323      -0.064       0.021
    y.L164         0.0009      0.022      0.044      0.965      -0.041       0.043
    y.L165         0.0072      0.022      0.333      0.739      -0.035       0.049
    y.L166        -0.0058      0.022     -0.267      0.789      -0.048       0.036
    y.L167         0.0248      0.022      1.150      0.250      -0.017       0.067
    y.L168         0.0153      0.022      0.714      0.475      -0.027       0.057
    y.L169        -0.0510      0.021     -2.375      0.018      -0.093      -0.009
    y.L170        -0.0189      0.022     -0.878      0.380      -0.061       0.023
    y.L171         0.0270      0.022      1.252      0.211      -0.015       0.069
    y.L172        -0.0240      0.022     -1.111      0.267      -0.066       0.018
    y.L173         0.0015      0.022      0.068      0.946      -0.041       0.044
    y.L174        -0.0152      0.022     -0.703      0.482      -0.057       0.027
    y.L175         0.0581      0.022      2.687      0.007       0.016       0.101
    y.L176        -0.0221      0.022     -1.020      0.308      -0.065       0.020
    y.L177         0.0031      0.022      0.143      0.887      -0.039       0.046
    y.L178         0.0022      0.022      0.101      0.919      -0.040       0.045
    y.L179         0.0182      0.022      0.841      0.401      -0.024       0.061
    y.L180         0.0368      0.022      1.700      0.089      -0.006       0.079
    y.L181         0.0256      0.022      1.182      0.237      -0.017       0.068
    y.L182        -0.0034      0.022     -0.159      0.874      -0.046       0.039
    y.L183         0.0267      0.022      1.234      0.217      -0.016       0.069
    y.L184         0.0046      0.022      0.213      0.831      -0.038       0.047
    y.L185         0.0578      0.022      2.674      0.007       0.015       0.100
    y.L186        -0.0026      0.022     -0.119      0.905      -0.045       0.040
    y.L187         0.0175      0.022      0.811      0.417      -0.025       0.060
    y.L188        -0.0027      0.022     -0.124      0.901      -0.045       0.040
    y.L189        -0.0151      0.022     -0.699      0.484      -0.057       0.027
    y.L190        -0.0144      0.022     -0.668      0.504      -0.057       0.028
    y.L191        -0.0335      0.022     -1.550      0.121      -0.076       0.009
    y.L192         0.0155      0.022      0.715      0.474      -0.027       0.058
    y.L193        -0.0051      0.022     -0.238      0.812      -0.047       0.037
    y.L194        -0.0359      0.022     -1.660      0.097      -0.078       0.006
    y.L195         0.0126      0.022      0.581      0.561      -0.030       0.055
    y.L196         0.0190      0.022      0.881      0.378      -0.023       0.061
    y.L197         0.0057      0.022      0.264      0.792      -0.037       0.048
    y.L198         0.0282      0.022      1.304      0.192      -0.014       0.071
    y.L199        -0.0054      0.022     -0.248      0.804      -0.048       0.037
    y.L200        -0.0125      0.022     -0.576      0.564      -0.055       0.030
    y.L201        -0.0274      0.022     -1.266      0.206      -0.070       0.015
    y.L202         0.0342      0.022      1.576      0.115      -0.008       0.077
    y.L203         0.0019      0.022      0.089      0.929      -0.041       0.044
    y.L204         0.0163      0.022      0.753      0.452      -0.026       0.059
    y.L205         0.0436      0.022      2.015      0.044       0.001       0.086
    y.L206        -0.0131      0.022     -0.607      0.544      -0.056       0.029
    y.L207         0.0246      0.022      1.135      0.256      -0.018       0.067
    y.L208        -0.0529      0.022     -2.453      0.014      -0.095      -0.011
    y.L209         0.0129      0.022      0.597      0.551      -0.029       0.055
    y.L210         0.0064      0.022      0.299      0.765      -0.036       0.049
    y.L211         0.0237      0.022      1.098      0.272      -0.019       0.066
    y.L212        -0.0331      0.022     -1.531      0.126      -0.075       0.009
    y.L213         0.0020      0.022      0.094      0.926      -0.040       0.044
    y.L214         0.0099      0.022      0.456      0.648      -0.032       0.052
    y.L215        -0.0248      0.022     -1.153      0.249      -0.067       0.017
    y.L216         0.0079      0.022      0.366      0.714      -0.034       0.050
    y.L217        -0.0101      0.022     -0.467      0.641      -0.052       0.032
    y.L218         0.0384      0.022      1.785      0.074      -0.004       0.081
    y.L219         0.0123      0.022      0.570      0.569      -0.030       0.054
    y.L220        -0.0298      0.022     -1.384      0.166      -0.072       0.012
    y.L221        -0.0292      0.022     -1.357      0.175      -0.071       0.013
    y.L222        -0.0375      0.022     -1.740      0.082      -0.080       0.005
    y.L223        -0.0149      0.022     -0.689      0.491      -0.057       0.027
    y.L224        -0.0166      0.022     -0.771      0.440      -0.059       0.026
    y.L225         0.0181      0.022      0.842      0.400      -0.024       0.060
    y.L226        -0.0498      0.022     -2.312      0.021      -0.092      -0.008
    y.L227        -0.0245      0.022     -1.137      0.256      -0.067       0.018
    y.L228        -0.0098      0.022     -0.456      0.648      -0.052       0.032
    y.L229         0.0113      0.022      0.526      0.599      -0.031       0.054
    y.L230     -2.213e-05      0.022     -0.001      0.999      -0.042       0.042
    y.L231        -0.0068      0.022     -0.315      0.753      -0.049       0.035
    y.L232        -0.0265      0.022     -1.233      0.217      -0.069       0.016
    y.L233         0.0389      0.022      1.807      0.071      -0.003       0.081
    y.L234         0.0077      0.022      0.359      0.720      -0.035       0.050
    y.L235        -0.0325      0.022     -1.504      0.133      -0.075       0.010
    y.L236     -9.445e-05      0.022     -0.004      0.997      -0.042       0.042
    y.L237        -0.0077      0.022     -0.359      0.720      -0.050       0.035
    y.L238        -0.0237      0.022     -1.096      0.273      -0.066       0.019
    y.L239         0.0070      0.022      0.322      0.747      -0.035       0.049
    y.L240         0.0007      0.022      0.034      0.973      -0.042       0.043
    y.L241         0.0277      0.022      1.283      0.200      -0.015       0.070
    y.L242        -0.0308      0.022     -1.423      0.155      -0.073       0.012
    y.L243        -0.0395      0.022     -1.828      0.068      -0.082       0.003
    y.L244        -0.0443      0.022     -2.047      0.041      -0.087      -0.002
    y.L245         0.0066      0.022      0.304      0.761      -0.036       0.049
    y.L246         0.0202      0.022      0.933      0.351      -0.022       0.063
    y.L247         0.0218      0.022      1.013      0.311      -0.020       0.064
    y.L248         0.0414      0.022      1.923      0.054      -0.001       0.084
    y.L249        -0.0245      0.022     -1.138      0.255      -0.067       0.018
    y.L250         0.0319      0.022      1.479      0.139      -0.010       0.074
    y.L251         0.0158      0.022      0.733      0.463      -0.026       0.058
    y.L252         0.0051      0.022      0.238      0.812      -0.037       0.047
    y.L253        -0.0169      0.022     -0.784      0.433      -0.059       0.025
    y.L254        -0.0062      0.022     -0.287      0.774      -0.049       0.036
    y.L255        -0.0045      0.022     -0.208      0.835      -0.047       0.038
    y.L256        -0.0349      0.022     -1.619      0.105      -0.077       0.007
    y.L257        -0.0533      0.022     -2.470      0.014      -0.096      -0.011
    y.L258         0.0310      0.022      1.434      0.151      -0.011       0.073
    y.L259         0.0035      0.022      0.160      0.873      -0.039       0.046
    y.L260        -0.0311      0.022     -1.439      0.150      -0.073       0.011
    y.L261         0.0239      0.022      1.108      0.268      -0.018       0.066
    y.L262         0.0063      0.022      0.291      0.771      -0.036       0.049
    y.L263        -0.0385      0.022     -1.785      0.074      -0.081       0.004
    y.L264        -0.0201      0.022     -0.934      0.350      -0.062       0.022
    y.L265         0.0038      0.022      0.179      0.858      -0.038       0.046
    y.L266        -0.0068      0.022     -0.315      0.753      -0.049       0.035
    y.L267        -0.0063      0.021     -0.294      0.769      -0.048       0.036
    y.L268        -0.0106      0.021     -0.493      0.622      -0.053       0.032
    y.L269         0.0310      0.021      1.443      0.149      -0.011       0.073
    y.L270        -0.0291      0.022     -1.354      0.176      -0.071       0.013
    y.L271         0.0240      0.022      1.115      0.265      -0.018       0.066
    y.L272         0.0308      0.022      1.434      0.152      -0.011       0.073
    y.L273         0.0254      0.022      1.183      0.237      -0.017       0.068
    y.L274        -0.0108      0.022     -0.500      0.617      -0.053       0.031
    y.L275        -0.0537      0.022     -2.493      0.013      -0.096      -0.011
    y.L276        -0.0115      0.022     -0.532      0.595      -0.054       0.031
    y.L277         0.0039      0.022      0.183      0.855      -0.038       0.046
    y.L278        -0.0204      0.022     -0.946      0.344      -0.063       0.022
    y.L279         0.0117      0.022      0.544      0.586      -0.031       0.054
    y.L280        -0.0476      0.022     -2.204      0.028      -0.090      -0.005
    y.L281        -0.0286      0.022     -1.324      0.186      -0.071       0.014
    y.L282        -0.0062      0.022     -0.287      0.774      -0.049       0.036
    y.L283         0.0370      0.022      1.714      0.087      -0.005       0.079
    y.L284         0.0295      0.022      1.367      0.172      -0.013       0.072
    y.L285         0.0365      0.022      1.689      0.091      -0.006       0.079
    y.L286         0.0147      0.022      0.680      0.497      -0.028       0.057
    y.L287        -0.0015      0.022     -0.068      0.945      -0.044       0.041
    y.L288         0.0365      0.022      1.691      0.091      -0.006       0.079
    y.L289         0.0249      0.022      1.149      0.251      -0.018       0.067
    y.L290         0.0149      0.022      0.690      0.490      -0.028       0.057
    y.L291        -0.0339      0.022     -1.567      0.117      -0.076       0.009
    y.L292        -0.0375      0.022     -1.729      0.084      -0.080       0.005
    y.L293        -0.0125      0.022     -0.578      0.563      -0.055       0.030
    y.L294         0.0148      0.022      0.680      0.496      -0.028       0.057
    y.L295         0.0508      0.022      2.344      0.019       0.008       0.093
    y.L296         0.0172      0.022      0.793      0.428      -0.025       0.060
    y.L297         0.0405      0.022      1.870      0.062      -0.002       0.083
    y.L298         0.0214      0.022      0.988      0.323      -0.021       0.064
    y.L299         0.0554      0.022      2.552      0.011       0.013       0.098
    y.L300         0.0082      0.022      0.377      0.706      -0.034       0.051
    y.L301        -0.0150      0.022     -0.689      0.491      -0.058       0.028
    y.L302        -0.0296      0.022     -1.364      0.173      -0.072       0.013
    y.L303         0.0153      0.022      0.701      0.484      -0.027       0.058
    y.L304        -0.0421      0.022     -1.933      0.053      -0.085       0.001
    y.L305        -0.0409      0.022     -1.878      0.060      -0.084       0.002
    y.L306        -0.0305      0.022     -1.397      0.162      -0.073       0.012
    y.L307        -0.0039      0.022     -0.180      0.857      -0.047       0.039
    y.L308         0.0510      0.022      2.335      0.020       0.008       0.094
    y.L309         0.0103      0.022      0.469      0.639      -0.033       0.053
    y.L310         0.0286      0.022      1.306      0.192      -0.014       0.071
    y.L311         0.0013      0.022      0.059      0.953      -0.042       0.044
    y.L312        -0.0051      0.022     -0.234      0.815      -0.048       0.038
    y.L313         0.0298      0.022      1.365      0.172      -0.013       0.073
    y.L314         0.0280      0.022      1.279      0.201      -0.015       0.071
    y.L315         0.0287      0.022      1.312      0.190      -0.014       0.072
    y.L316         0.0036      0.022      0.165      0.869      -0.039       0.046
    y.L317         0.0248      0.022      1.135      0.256      -0.018       0.068
    y.L318        -0.0214      0.022     -0.977      0.329      -0.064       0.022
    y.L319        -0.0434      0.022     -1.983      0.047      -0.086      -0.001
    y.L320         0.0322      0.022      1.469      0.142      -0.011       0.075
    y.L321         0.0233      0.022      1.061      0.289      -0.020       0.066
    y.L322        -0.0114      0.022     -0.519      0.604      -0.054       0.032
    y.L323        -0.0565      0.022     -2.578      0.010      -0.100      -0.014
    y.L324         0.0080      0.022      0.363      0.717      -0.035       0.051
    y.L325        -0.0357      0.022     -1.622      0.105      -0.079       0.007
    y.L326        -0.0154      0.022     -0.700      0.484      -0.059       0.028
    y.L327         0.0112      0.022      0.509      0.611      -0.032       0.054
    y.L328        -0.0366      0.022     -1.665      0.096      -0.080       0.006
    y.L329        -0.0240      0.022     -1.088      0.277      -0.067       0.019
    y.L330        -0.0587      0.022     -2.665      0.008      -0.102      -0.016
    y.L331         0.0345      0.022      1.562      0.118      -0.009       0.078
    y.L332         0.0074      0.022      0.333      0.739      -0.036       0.051
    y.L333        -0.0047      0.022     -0.212      0.832      -0.048       0.039
    y.L334        -0.0335      0.022     -1.517      0.129      -0.077       0.010
    y.L335         0.0017      0.022      0.079      0.937      -0.042       0.045
    y.L336         0.0104      0.022      0.468      0.640      -0.033       0.054
    y.L337        -0.0146      0.022     -0.658      0.510      -0.058       0.029
    y.L338         0.0229      0.022      1.034      0.301      -0.020       0.066
    y.L339         0.0139      0.022      0.629      0.529      -0.029       0.057
    y.L340        -0.0077      0.022     -0.348      0.728      -0.051       0.036
    y.L341        -0.0205      0.022     -0.926      0.354      -0.064       0.023
    y.L342        -0.0224      0.022     -1.010      0.312      -0.066       0.021
    y.L343         0.0638      0.022      2.883      0.004       0.020       0.107
    y.L344         0.0249      0.022      1.126      0.260      -0.018       0.068
    y.L345         0.0371      0.022      1.675      0.094      -0.006       0.081
    y.L346        -0.0179      0.022     -0.807      0.420      -0.061       0.026
    y.L347        -0.0212      0.022     -0.956      0.339      -0.065       0.022
    y.L348        -0.0327      0.022     -1.477      0.140      -0.076       0.011
    y.L349         0.0199      0.022      0.896      0.370      -0.024       0.063
    y.L350         0.0498      0.022      2.247      0.025       0.006       0.093
    y.L351         0.0063      0.022      0.284      0.776      -0.037       0.050
    y.L352        -0.0479      0.022     -2.160      0.031      -0.091      -0.004
    y.L353         0.0083      0.022      0.373      0.709      -0.035       0.052
    y.L354         0.0272      0.022      1.224      0.221      -0.016       0.071
    y.L355        -0.0362      0.022     -1.632      0.103      -0.080       0.007
    y.L356        -0.0033      0.022     -0.147      0.883      -0.047       0.040
    y.L357        -0.0220      0.022     -0.994      0.320      -0.066       0.021
    y.L358         0.0581      0.022      2.618      0.009       0.015       0.102
    y.L359        -0.0244      0.022     -1.095      0.274      -0.068       0.019
    y.L360        -0.0522      0.022     -2.349      0.019      -0.096      -0.009
    y.L361         0.0577      0.022      2.593      0.010       0.014       0.101
    y.L362         0.0156      0.022      0.702      0.483      -0.028       0.059
    y.L363         0.0221      0.022      0.992      0.321      -0.022       0.066
    y.L364         0.0109      0.022      0.491      0.624      -0.033       0.055
    y.L365         0.0062      0.022      0.281      0.779      -0.037       0.050
    y.L366        -0.0469      0.022     -2.108      0.035      -0.091      -0.003
    y.L367        -0.0099      0.022     -0.445      0.656      -0.054       0.034
    y.L368        -0.0050      0.022     -0.223      0.824      -0.049       0.039
    y.L369         0.0272      0.022      1.217      0.224      -0.017       0.071
    y.L370         0.0452      0.022      2.025      0.043       0.001       0.089
    y.L371        -0.0527      0.022     -2.354      0.019      -0.096      -0.009
    y.L372        -0.0384      0.022     -1.716      0.086      -0.082       0.005
    y.L373         0.0281      0.022      1.256      0.209      -0.016       0.072
    y.L374         0.0067      0.022      0.302      0.763      -0.037       0.051
    y.L375         0.0145      0.022      0.646      0.518      -0.029       0.058
    y.L376         0.0320      0.022      1.432      0.152      -0.012       0.076
    y.L377         0.0046      0.022      0.207      0.836      -0.039       0.048
    y.L378        -0.0288      0.022     -1.286      0.198      -0.073       0.015
    y.L379         0.0387      0.022      1.731      0.084      -0.005       0.082
    y.L380         0.0282      0.022      1.260      0.208      -0.016       0.072
    y.L381        -0.0098      0.022     -0.440      0.660      -0.054       0.034
    y.L382         0.0324      0.022      1.450      0.147      -0.011       0.076
    y.L383        -0.0214      0.022     -0.956      0.339      -0.065       0.022
    y.L384        -0.0335      0.022     -1.494      0.135      -0.077       0.010
    y.L385         0.0364      0.022      1.622      0.105      -0.008       0.080
    y.L386        -0.0317      0.022     -1.413      0.158      -0.076       0.012
    y.L387         0.0042      0.022      0.186      0.853      -0.040       0.048
    y.L388         0.0317      0.022      1.415      0.157      -0.012       0.076
    y.L389        -0.0271      0.022     -1.210      0.226      -0.071       0.017
    y.L390         0.0128      0.022      0.571      0.568      -0.031       0.057
    y.L391        -0.0109      0.022     -0.489      0.625      -0.055       0.033
    y.L392         0.0184      0.022      0.819      0.413      -0.026       0.062
    y.L393         0.0315      0.022      1.404      0.160      -0.012       0.075
    y.L394        -0.0118      0.022     -0.526      0.599      -0.056       0.032
    y.L395        -0.0127      0.022     -0.568      0.570      -0.057       0.031
    y.L396        -0.0112      0.022     -0.501      0.616      -0.055       0.033
    y.L397         0.0220      0.022      0.982      0.326      -0.022       0.066
    y.L398         0.0413      0.022      1.838      0.066      -0.003       0.085
    y.L399         0.0015      0.022      0.068      0.946      -0.042       0.046
    y.L400         0.0063      0.022      0.279      0.780      -0.038       0.050
    y.L401         0.0102      0.022      0.454      0.650      -0.034       0.054
    y.L402         0.0120      0.022      0.533      0.594      -0.032       0.056
    y.L403         0.0095      0.022      0.425      0.671      -0.034       0.054
    y.L404         0.0231      0.022      1.031      0.303      -0.021       0.067
    y.L405         0.0260      0.023      1.155      0.248      -0.018       0.070
    y.L406         0.0043      0.023      0.191      0.849      -0.040       0.048
    y.L407         0.0130      0.023      0.575      0.565      -0.031       0.057
    y.L408        -0.0137      0.022     -0.611      0.541      -0.058       0.030
    y.L409        -0.0848      0.022     -3.771      0.000      -0.129      -0.041
    y.L410        -0.0258      0.023     -1.143      0.253      -0.070       0.018
    y.L411         0.0581      0.023      2.575      0.010       0.014       0.102
    y.L412         0.0230      0.023      1.018      0.308      -0.021       0.067
    y.L413         0.0110      0.023      0.488      0.626      -0.033       0.055
    y.L414        -0.0271      0.023     -1.200      0.230      -0.071       0.017
    y.L415         0.0023      0.023      0.102      0.919      -0.042       0.047
    y.L416        -0.0208      0.023     -0.922      0.356      -0.065       0.023
    y.L417         0.0284      0.023      1.258      0.208      -0.016       0.073
    y.L418        -0.0062      0.023     -0.276      0.783      -0.050       0.038
    y.L419         0.0256      0.023      1.134      0.257      -0.019       0.070
    y.L420         0.0094      0.023      0.416      0.678      -0.035       0.054
    y.L421        -0.0508      0.023     -2.248      0.025      -0.095      -0.006
    y.L422        -0.0415      0.023     -1.834      0.067      -0.086       0.003
    y.L423         0.0067      0.023      0.295      0.768      -0.038       0.051
    y.L424        -0.0393      0.023     -1.734      0.083      -0.084       0.005
    y.L425         0.0363      0.023      1.602      0.109      -0.008       0.081
    y.L426        -0.0211      0.023     -0.928      0.353      -0.066       0.023
    y.L427         0.0028      0.023      0.121      0.904      -0.042       0.048
    y.L428         0.0219      0.023      0.960      0.337      -0.023       0.067
    y.L429        -0.0006      0.023     -0.027      0.979      -0.045       0.044
    y.L430         0.0267      0.023      1.171      0.242      -0.018       0.071
    y.L431         0.0080      0.023      0.348      0.728      -0.037       0.053
    y.L432         0.0091      0.023      0.400      0.689      -0.036       0.054
    y.L433        -0.0317      0.023     -1.390      0.165      -0.076       0.013
    y.L434        -0.0139      0.023     -0.608      0.543      -0.059       0.031
    y.L435         0.0342      0.023      1.500      0.134      -0.010       0.079
    y.L436        -0.0309      0.023     -1.358      0.175      -0.076       0.014
    y.L437         0.0098      0.023      0.429      0.668      -0.035       0.054
    y.L438         0.0449      0.023      1.969      0.049       0.000       0.090
    y.L439        -0.0397      0.023     -1.740      0.082      -0.084       0.005
    y.L440         0.0418      0.023      1.832      0.067      -0.003       0.086
    y.L441         0.0217      0.023      0.953      0.341      -0.023       0.066
    y.L442         0.0207      0.023      0.906      0.365      -0.024       0.065
    y.L443         0.0244      0.023      1.070      0.285      -0.020       0.069
    y.L444        -0.0716      0.023     -3.143      0.002      -0.116      -0.027
    y.L445        -0.0350      0.023     -1.532      0.126      -0.080       0.010
    y.L446         0.0271      0.023      1.186      0.236      -0.018       0.072
    y.L447        -0.0103      0.023     -0.451      0.652      -0.055       0.034
    y.L448      9.685e-05      0.023      0.004      0.997      -0.045       0.045
    y.L449         0.0698      0.023      3.064      0.002       0.025       0.114
    y.L450        -0.0106      0.023     -0.464      0.643      -0.055       0.034
    y.L451         0.0109      0.023      0.479      0.632      -0.034       0.056
    y.L452        -0.0129      0.023     -0.566      0.572      -0.058       0.032
    y.L453         0.0452      0.023      1.981      0.048       0.000       0.090
    y.L454        -0.0076      0.023     -0.334      0.739      -0.052       0.037
    y.L455         0.0317      0.023      1.389      0.165      -0.013       0.076
    y.L456         0.0130      0.023      0.569      0.569      -0.032       0.058
    y.L457        -0.0672      0.023     -2.943      0.003      -0.112      -0.022
    y.L458         0.0059      0.023      0.260      0.795      -0.039       0.051
    y.L459        -0.0251      0.023     -1.098      0.272      -0.070       0.020
    y.L460         0.0546      0.023      2.387      0.017       0.010       0.099
    y.L461         0.0076      0.023      0.331      0.741      -0.037       0.053
    y.L462        -0.0265      0.023     -1.155      0.248      -0.071       0.018
    y.L463         0.0037      0.023      0.162      0.871      -0.041       0.049
    y.L464        -0.0293      0.023     -1.273      0.203      -0.074       0.016
    y.L465        -0.0158      0.023     -0.684      0.494      -0.061       0.029
    y.L466         0.0092      0.023      0.398      0.691      -0.036       0.054
    y.L467         0.0413      0.023      1.793      0.073      -0.004       0.086
    y.L468        -0.0060      0.023     -0.260      0.795      -0.051       0.039
    y.L469         0.0002      0.023      0.010      0.992      -0.045       0.045
    y.L470        -0.0498      0.023     -2.165      0.030      -0.095      -0.005
    y.L471         0.0232      0.023      1.009      0.313      -0.022       0.068
    y.L472        -0.0016      0.023     -0.069      0.945      -0.047       0.044
    y.L473         0.0135      0.023      0.586      0.558      -0.032       0.059
    y.L474         0.0202      0.023      0.875      0.382      -0.025       0.065
    y.L475        -0.0255      0.023     -1.106      0.269      -0.071       0.020
    y.L476         0.0290      0.023      1.258      0.208      -0.016       0.074
    y.L477         0.0367      0.023      1.588      0.112      -0.009       0.082
    y.L478        -0.0322      0.023     -1.392      0.164      -0.077       0.013
    y.L479        -0.0191      0.023     -0.826      0.409      -0.064       0.026
    y.L480        -0.0162      0.023     -0.700      0.484      -0.062       0.029
    y.L481         0.0204      0.023      0.880      0.379      -0.025       0.066
    y.L482         0.0149      0.023      0.642      0.521      -0.031       0.060
    y.L483         0.0132      0.023      0.566      0.571      -0.032       0.059
    y.L484        -0.0293      0.023     -1.258      0.208      -0.075       0.016
    y.L485         0.0275      0.023      1.179      0.238      -0.018       0.073
    y.L486      9.148e-05      0.023      0.004      0.997      -0.046       0.046
    y.L487        -0.0228      0.023     -0.975      0.329      -0.069       0.023
    y.L488        -0.0180      0.023     -0.770      0.441      -0.064       0.028
    y.L489         0.0154      0.023      0.661      0.509      -0.030       0.061
    y.L490        -0.0244      0.023     -1.044      0.297      -0.070       0.021
    y.L491         0.0521      0.023      2.228      0.026       0.006       0.098
    y.L492        -0.0449      0.023     -1.921      0.055      -0.091       0.001
    y.L493         0.0103      0.023      0.438      0.661      -0.036       0.056
    y.L494        -0.0423      0.023     -1.803      0.071      -0.088       0.004
    y.L495         0.0418      0.024      1.771      0.076      -0.004       0.088
    y.L496         0.0199      0.024      0.839      0.402      -0.027       0.066
    y.L497         0.0245      0.024      1.035      0.301      -0.022       0.071
    y.L498         0.0127      0.024      0.539      0.590      -0.034       0.059
    y.L499         0.0025      0.024      0.106      0.915      -0.044       0.049
    y.L500        -0.0085      0.024     -0.358      0.720      -0.055       0.038
    y.L501         0.0718      0.024      3.039      0.002       0.026       0.118
    y.L502        -0.0006      0.024     -0.025      0.980      -0.047       0.046
    y.L503         0.0218      0.024      0.917      0.359      -0.025       0.068
    y.L504         0.0342      0.024      1.437      0.151      -0.012       0.081
    y.L505         0.0270      0.024      1.137      0.256      -0.020       0.074
    y.L506         0.0605      0.024      2.542      0.011       0.014       0.107
    y.L507         0.0182      0.024      0.764      0.445      -0.029       0.065
    y.L508         0.0093      0.024      0.391      0.695      -0.037       0.056
    y.L509         0.0016      0.024      0.068      0.946      -0.045       0.048
    y.L510        -0.0225      0.024     -0.945      0.345      -0.069       0.024
    y.L511         0.0124      0.024      0.518      0.604      -0.034       0.059
    y.L512        -0.0283      0.024     -1.187      0.235      -0.075       0.018
    y.L513        -0.0291      0.024     -1.219      0.223      -0.076       0.018
    y.L514         0.0043      0.024      0.179      0.858      -0.043       0.051
    y.L515         0.0096      0.024      0.400      0.689      -0.037       0.056
    y.L516         0.0605      0.024      2.529      0.011       0.014       0.107
    y.L517         0.0065      0.024      0.272      0.785      -0.040       0.053
    y.L518        -0.0482      0.024     -2.012      0.044      -0.095      -0.001
    y.L519        -0.0397      0.024     -1.656      0.098      -0.087       0.007
    y.L520         0.0075      0.024      0.313      0.754      -0.040       0.055
    y.L521         0.0194      0.024      0.808      0.419      -0.028       0.067
    y.L522        -0.0107      0.024     -0.446      0.656      -0.058       0.036
    y.L523         0.0503      0.024      2.089      0.037       0.003       0.097
    y.L524        -0.0378      0.024     -1.569      0.117      -0.085       0.009
    y.L525         0.0300      0.024      1.245      0.213      -0.017       0.077
    y.L526        -0.0082      0.024     -0.338      0.736      -0.055       0.039
    y.L527         0.0299      0.024      1.238      0.216      -0.017       0.077
    y.L528        -0.0396      0.024     -1.638      0.101      -0.087       0.008
    y.L529        -0.0241      0.024     -1.000      0.317      -0.071       0.023
    y.L530         0.0054      0.024      0.222      0.824      -0.042       0.053
    y.L531         0.0121      0.024      0.503      0.615      -0.035       0.060
    y.L532        -0.0157      0.024     -0.651      0.515      -0.063       0.032
    y.L533         0.0583      0.024      2.414      0.016       0.011       0.106
    y.L534        -0.0303      0.024     -1.253      0.210      -0.078       0.017
    y.L535         0.0109      0.024      0.451      0.652      -0.037       0.058
    y.L536         0.0045      0.024      0.184      0.854      -0.043       0.052
    y.L537         0.0184      0.024      0.758      0.448      -0.029       0.066
    y.L538        -0.0531      0.024     -2.192      0.028      -0.101      -0.006
    y.L539         0.0454      0.024      1.870      0.061      -0.002       0.093
    y.L540         0.0305      0.024      1.255      0.210      -0.017       0.078
    y.L541        -0.0163      0.024     -0.671      0.502      -0.064       0.031
    y.L542        -0.0106      0.024     -0.437      0.662      -0.058       0.037
    y.L543         0.0020      0.024      0.080      0.936      -0.046       0.050
    y.L544        -0.0646      0.024     -2.655      0.008      -0.112      -0.017
    y.L545        -0.0335      0.024     -1.377      0.168      -0.081       0.014
    y.L546         0.0147      0.024      0.604      0.546      -0.033       0.062
    y.L547        -0.0185      0.024     -0.761      0.447      -0.066       0.029
    y.L548         0.0041      0.024      0.169      0.866      -0.044       0.052
    y.L549         0.0063      0.024      0.259      0.796      -0.041       0.054
    y.L550        -0.0298      0.024     -1.224      0.221      -0.078       0.018
    y.L551         0.0380      0.024      1.555      0.120      -0.010       0.086
    y.L552         0.0432      0.024      1.767      0.077      -0.005       0.091
    y.L553         0.0151      0.024      0.615      0.538      -0.033       0.063
    y.L554         0.0363      0.024      1.485      0.138      -0.012       0.084
    y.L555        -0.0205      0.024     -0.836      0.403      -0.068       0.027
    y.L556         0.0202      0.024      0.825      0.409      -0.028       0.068
    y.L557        -0.0454      0.024     -1.852      0.064      -0.093       0.003
    y.L558        -0.0010      0.025     -0.039      0.969      -0.049       0.047
    y.L559        -0.0284      0.025     -1.160      0.246      -0.076       0.020
    y.L560         0.0020      0.025      0.081      0.936      -0.046       0.050
    y.L561        -0.0170      0.025     -0.694      0.487      -0.065       0.031
    y.L562        -0.0148      0.025     -0.602      0.547      -0.063       0.033
    y.L563         0.0291      0.025      1.185      0.236      -0.019       0.077
    y.L564         0.0083      0.025      0.337      0.736      -0.040       0.056
    y.L565         0.0387      0.025      1.578      0.115      -0.009       0.087
    y.L566         0.0140      0.025      0.568      0.570      -0.034       0.062
    y.L567         0.0481      0.025      1.961      0.050    2.06e-05       0.096
    y.L568        -0.0209      0.025     -0.852      0.394      -0.069       0.027
    y.L569         0.0151      0.025      0.613      0.540      -0.033       0.063
    y.L570        -0.0283      0.025     -1.150      0.250      -0.077       0.020
    y.L571         0.0357      0.025      1.446      0.148      -0.013       0.084
    y.L572        -0.0268      0.025     -1.085      0.278      -0.075       0.022
    y.L573         0.0414      0.025      1.679      0.093      -0.007       0.090
    y.L574        -0.0573      0.025     -2.323      0.020      -0.106      -0.009
    y.L575         0.0015      0.025      0.061      0.952      -0.047       0.050
    y.L576         0.0540      0.025      2.185      0.029       0.006       0.102
    y.L577         0.0176      0.025      0.711      0.477      -0.031       0.066
    y.L578        -0.0326      0.025     -1.317      0.188      -0.081       0.016
    y.L579         0.0130      0.025      0.523      0.601      -0.036       0.062
    y.L580        -0.0099      0.025     -0.399      0.690      -0.058       0.039
    y.L581         0.0156      0.025      0.628      0.530      -0.033       0.064
    y.L582         0.0001      0.025      0.005      0.996      -0.048       0.049
    y.L583        -0.0428      0.025     -1.728      0.084      -0.091       0.006
    y.L584        -0.0155      0.025     -0.625      0.532      -0.064       0.033
    y.L585        -0.0084      0.025     -0.337      0.736      -0.057       0.040
    y.L586        -0.0148      0.025     -0.598      0.550      -0.063       0.034
    y.L587         0.0143      0.025      0.576      0.564      -0.034       0.063
    y.L588         0.0056      0.025      0.224      0.823      -0.043       0.054
    y.L589        -0.0424      0.025     -1.707      0.088      -0.091       0.006
    y.L590         0.0289      0.025      1.165      0.244      -0.020       0.078
    y.L591        -0.0262      0.025     -1.053      0.292      -0.075       0.023
    y.L592         0.0192      0.025      0.767      0.443      -0.030       0.068
    y.L593        -0.0065      0.025     -0.259      0.795      -0.056       0.043
    y.L594         0.0077      0.025      0.306      0.759      -0.041       0.057
    y.L595        -0.0232      0.025     -0.919      0.358      -0.073       0.026
    y.L596         0.0185      0.025      0.735      0.462      -0.031       0.068
    y.L597        -0.0031      0.025     -0.125      0.901      -0.053       0.046
    y.L598        -0.0583      0.025     -2.311      0.021      -0.108      -0.009
    y.L599        -0.0198      0.025     -0.785      0.432      -0.069       0.030
    y.L600         0.0337      0.025      1.334      0.182      -0.016       0.083
    y.L601         0.0388      0.025      1.538      0.124      -0.011       0.088
    y.L602         0.0458      0.025      1.816      0.069      -0.004       0.095
    y.L603         0.0023      0.025      0.091      0.928      -0.047       0.052
    y.L604        -0.0261      0.025     -1.028      0.304      -0.076       0.024
    y.L605        -0.0602      0.025     -2.376      0.017      -0.110      -0.011
    y.L606         0.0367      0.025      1.446      0.148      -0.013       0.086
    y.L607         0.0263      0.025      1.035      0.301      -0.024       0.076
    y.L608        -0.0272      0.025     -1.073      0.283      -0.077       0.023
    y.L609         0.0151      0.025      0.595      0.552      -0.035       0.065
    y.L610        -0.0042      0.025     -0.167      0.868      -0.054       0.046
    y.L611         0.0225      0.025      0.888      0.375      -0.027       0.072
    y.L612        -0.0023      0.025     -0.091      0.927      -0.052       0.047
    y.L613         0.0721      0.025      2.842      0.004       0.022       0.122
    y.L614         0.0172      0.025      0.677      0.498      -0.033       0.067
    y.L615        -0.0189      0.025     -0.746      0.456      -0.069       0.031
    y.L616         0.0029      0.025      0.116      0.907      -0.047       0.053
    y.L617         0.0407      0.025      1.605      0.109      -0.009       0.090
    y.L618        -0.0535      0.025     -2.112      0.035      -0.103      -0.004
    y.L619         0.0628      0.025      2.474      0.013       0.013       0.113
    y.L620        -0.0032      0.027     -0.121      0.904      -0.055       0.049
    y.L621        -0.0084      0.027     -0.315      0.752      -0.061       0.044
    y.L622         0.0126      0.027      0.471      0.638      -0.040       0.065
    y.L623         0.0537      0.027      2.011      0.044       0.001       0.106
    y.L624         0.0500      0.027      1.872      0.061      -0.002       0.102
    y.L625         0.0673      0.027      2.516      0.012       0.015       0.120
    y.L626         0.0017      0.027      0.064      0.949      -0.051       0.054
    y.L627        -0.0344      0.027     -1.286      0.198      -0.087       0.018
    y.L628         0.0389      0.027      1.456      0.145      -0.013       0.091
    y.L629         0.0234      0.027      0.876      0.381      -0.029       0.076
    y.L630        -0.0241      0.027     -0.902      0.367      -0.076       0.028
    y.L631        -0.0540      0.027     -2.022      0.043      -0.106      -0.002
    y.L632        -0.0004      0.027     -0.013      0.989      -0.053       0.052
    y.L633        -0.0754      0.027     -2.814      0.005      -0.128      -0.023
    y.L634        -0.0508      0.027     -1.894      0.058      -0.103       0.002
    y.L635        -0.0186      0.027     -0.693      0.488      -0.071       0.034
    y.L636         0.0640      0.027      2.389      0.017       0.011       0.117
    y.L637         0.0082      0.027      0.307      0.759      -0.044       0.061
    y.L638         0.0374      0.027      1.395      0.163      -0.015       0.090
    y.L639        -0.0431      0.027     -1.606      0.108      -0.096       0.009
    y.L640         0.0051      0.027      0.189      0.850      -0.048       0.058
    y.L641        -0.0184      0.027     -0.687      0.492      -0.071       0.034
    y.L642         0.0258      0.027      0.962      0.336      -0.027       0.078
    y.L643        -0.0287      0.027     -1.069      0.285      -0.081       0.024
    y.L644        -0.0128      0.027     -0.477      0.633      -0.065       0.040
    y.L645         0.0312      0.027      1.161      0.245      -0.021       0.084
    y.L646         0.0115      0.027      0.427      0.669      -0.041       0.064
    y.L647        -0.0217      0.027     -0.809      0.419      -0.074       0.031
    y.L648        -0.0487      0.027     -1.813      0.070      -0.101       0.004
    y.L649         0.0293      0.027      1.092      0.275      -0.023       0.082
    y.L650         0.0135      0.027      0.503      0.615      -0.039       0.066
    y.L651         0.0055      0.027      0.206      0.837      -0.047       0.058
    y.L652         0.0158      0.027      0.590      0.555      -0.037       0.068
    y.L653         0.0120      0.027      0.449      0.653      -0.040       0.065
    y.L654         0.0144      0.027      0.537      0.591      -0.038       0.067
    y.L655        -0.0047      0.027     -0.174      0.862      -0.057       0.048
    y.L656        -0.0107      0.027     -0.401      0.688      -0.063       0.042
    y.L657         0.0212      0.027      0.791      0.429      -0.031       0.074
    y.L658        -0.0321      0.027     -1.198      0.231      -0.085       0.020
    y.L659         0.0138      0.027      0.516      0.606      -0.039       0.066
    y.L660        -0.0084      0.027     -0.315      0.753      -0.061       0.044
    y.L661         0.0188      0.027      0.703      0.482      -0.034       0.071
    y.L662         0.0310      0.027      1.159      0.246      -0.021       0.084
    y.L663        -0.0113      0.027     -0.422      0.673      -0.064       0.041
    y.L664        -0.0402      0.027     -1.504      0.133      -0.093       0.012
    y.L665        -0.0054      0.027     -0.202      0.840      -0.058       0.047
    y.L666         0.0334      0.027      1.250      0.211      -0.019       0.086
    y.L667         0.0265      0.027      0.992      0.321      -0.026       0.079
    y.L668        -0.0210      0.027     -0.787      0.431      -0.073       0.031
    y.L669        -0.0311      0.027     -1.166      0.244      -0.084       0.021
    y.L670        -0.0178      0.027     -0.666      0.506      -0.070       0.035
    y.L671        -0.0332      0.027     -1.241      0.214      -0.086       0.019
    y.L672        -0.0123      0.027     -0.461      0.645      -0.065       0.040
    y.L673        -0.0207      0.027     -0.775      0.438      -0.073       0.032
    y.L674         0.0266      0.027      0.998      0.318      -0.026       0.079
    y.L675        -0.0334      0.027     -1.254      0.210      -0.086       0.019
    y.L676        -0.0326      0.027     -1.223      0.221      -0.085       0.020
    y.L677         0.0331      0.027      1.242      0.214      -0.019       0.085
    y.L678        -0.0332      0.027     -1.241      0.215      -0.086       0.019
    y.L679         0.0136      0.027      0.506      0.613      -0.039       0.066
    y.L680         0.0015      0.027      0.055      0.956      -0.051       0.054
    y.L681         0.0359      0.027      1.340      0.180      -0.017       0.088
    y.L682         0.0955      0.027      3.563      0.000       0.043       0.148
    y.L683         0.0005      0.027      0.019      0.985      -0.052       0.053
    y.L684        -0.0044      0.027     -0.162      0.871      -0.057       0.048
    y.L685        -0.0360      0.027     -1.341      0.180      -0.089       0.017
    y.L686        -0.0045      0.027     -0.167      0.867      -0.057       0.048
    y.L687         0.0153      0.027      0.569      0.570      -0.037       0.068
    y.L688        -0.0844      0.027     -3.139      0.002      -0.137      -0.032
    y.L689        -0.0007      0.027     -0.025      0.980      -0.054       0.052
    y.L690         0.0395      0.027      1.467      0.142      -0.013       0.092
    y.L691        -0.0127      0.027     -0.469      0.639      -0.066       0.040
    y.L692         0.0043      0.027      0.160      0.873      -0.049       0.057
    y.L693         0.0108      0.027      0.400      0.689      -0.042       0.063
    y.L694        -0.0279      0.027     -1.037      0.300      -0.080       0.025
    y.L695         0.0250      0.027      0.931      0.352      -0.028       0.078
    y.L696        -0.0512      0.027     -1.906      0.057      -0.104       0.001
    y.L697         0.0054      0.027      0.202      0.840      -0.047       0.058
    y.L698         0.0071      0.027      0.264      0.792      -0.046       0.060
    y.L699         0.0164      0.027      0.609      0.542      -0.036       0.069
    y.L700         0.0200      0.027      0.743      0.457      -0.033       0.073
    y.L701        -0.0321      0.027     -1.195      0.232      -0.085       0.021
    y.L702         0.0392      0.027      1.459      0.145      -0.013       0.092
    y.L703         0.0090      0.027      0.334      0.738      -0.044       0.062
    y.L704        -0.0160      0.027     -0.596      0.551      -0.069       0.037
    y.L705         0.0160      0.027      0.594      0.553      -0.037       0.069
    y.L706         0.0094      0.027      0.349      0.727      -0.043       0.062
    y.L707        -0.0056      0.027     -0.209      0.835      -0.058       0.047
    y.L708        -0.0011      0.027     -0.042      0.967      -0.054       0.051
    y.L709         0.0237      0.027      0.883      0.377      -0.029       0.076
    y.L710        -0.0450      0.027     -1.679      0.093      -0.098       0.008
    y.L711        -0.0017      0.027     -0.063      0.950      -0.054       0.051
    y.L712         0.0591      0.027      2.211      0.027       0.007       0.111
    y.L713        -0.0110      0.027     -0.412      0.680      -0.064       0.041
    y.L714        -0.0115      0.027     -0.430      0.667      -0.064       0.041
    y.L715         0.0482      0.027      1.802      0.072      -0.004       0.101
    y.L716         0.0143      0.027      0.533      0.594      -0.038       0.067
    y.L717         0.0029      0.027      0.110      0.912      -0.050       0.055
    y.L718         0.0215      0.027      0.804      0.421      -0.031       0.074
    y.L719         0.0580      0.027      2.172      0.030       0.006       0.110
    y.L720         0.0357      0.027      1.335      0.182      -0.017       0.088
    y.L721         0.0208      0.027      0.778      0.436      -0.032       0.073
    y.L722        -0.0306      0.027     -1.141      0.254      -0.083       0.022
    y.L723        -0.0856      0.027     -3.198      0.001      -0.138      -0.033
    y.L724        -0.0046      0.027     -0.171      0.864      -0.057       0.048
    y.L725         0.0422      0.027      1.573      0.116      -0.010       0.095
    y.L726        -0.0333      0.027     -1.239      0.215      -0.086       0.019
    y.L727         0.0248      0.027      0.924      0.355      -0.028       0.078
    y.L728        -0.0024      0.027     -0.091      0.928      -0.055       0.050
    y.L729        -0.0071      0.027     -0.264      0.792      -0.060       0.046
    y.L730         0.0086      0.027      0.319      0.750      -0.044       0.061
                                         Roots                                     
    ===============================================================================
                        Real          Imaginary           Modulus         Frequency
    -------------------------------------------------------------------------------
    AR.1              0.6697           -0.7422j            0.9997           -0.1332
    AR.2              0.6697           +0.7422j            0.9997            0.1332
    AR.3              0.6762           -0.7372j            1.0004           -0.1319
    AR.4              0.6762           +0.7372j            1.0004            0.1319
    AR.5              0.6831           -0.7314j            1.0008           -0.1304
    AR.6              0.6831           +0.7314j            1.0008            0.1304
    AR.7              0.6913           -0.7258j            1.0023           -0.1289
    AR.8              0.6913           +0.7258j            1.0023            0.1289
    AR.9              0.6962           -0.7215j            1.0026           -0.1278
    AR.10             0.6962           +0.7215j            1.0026            0.1278
    AR.11             0.7030           -0.7129j            1.0012           -0.1261
    AR.12             0.7030           +0.7129j            1.0012            0.1261
    AR.13             0.7110           -0.7043j            1.0008           -0.1242
    AR.14             0.7110           +0.7043j            1.0008            0.1242
    AR.15             0.7200           -0.6967j            1.0019           -0.1224
    AR.16             0.7200           +0.6967j            1.0019            0.1224
    AR.17             0.7264           -0.6924j            1.0036           -0.1212
    AR.18             0.7264           +0.6924j            1.0036            0.1212
    AR.19             0.7311           -0.6859j            1.0024           -0.1199
    AR.20             0.7311           +0.6859j            1.0024            0.1199
    AR.21             0.7370           -0.6775j            1.0011           -0.1183
    AR.22             0.7370           +0.6775j            1.0011            0.1183
    AR.23             0.6636           -0.7507j            1.0019           -0.1348
    AR.24             0.6636           +0.7507j            1.0019            0.1348
    AR.25             0.7456           -0.6683j            1.0013           -0.1163
    AR.26             0.7456           +0.6683j            1.0013            0.1163
    AR.27             0.6552           -0.7570j            1.0012           -0.1365
    AR.28             0.6552           +0.7570j            1.0012            0.1365
    AR.29             0.7532           -0.6585j            1.0005           -0.1143
    AR.30             0.7532           +0.6585j            1.0005            0.1143
    AR.31             0.7587           -0.6521j            1.0004           -0.1130
    AR.32             0.7587           +0.6521j            1.0004            0.1130
    AR.33             0.7651           -0.6443j            1.0003           -0.1114
    AR.34             0.7651           +0.6443j            1.0003            0.1114
    AR.35             0.7490           -0.6717j            1.0060           -0.1164
    AR.36             0.7490           +0.6717j            1.0060            0.1164
    AR.37             0.6476           -0.7643j            1.0017           -0.1381
    AR.38             0.6476           +0.7643j            1.0017            0.1381
    AR.39             0.6422           -0.7693j            1.0021           -0.1393
    AR.40             0.6422           +0.7693j            1.0021            0.1393
    AR.41             0.7721           -0.6390j            1.0023           -0.1100
    AR.42             0.7721           +0.6390j            1.0023            0.1100
    AR.43            -0.6907           -0.7235j            1.0002           -0.3713
    AR.44            -0.6907           +0.7235j            1.0002            0.3713
    AR.45            -0.6989           -0.7181j            1.0021           -0.3729
    AR.46            -0.6989           +0.7181j            1.0021            0.3729
    AR.47            -0.7047           -0.7134j            1.0027           -0.3740
    AR.48            -0.7047           +0.7134j            1.0027            0.3740
    AR.49            -0.7108           -0.7067j            1.0024           -0.3755
    AR.50            -0.7108           +0.7067j            1.0024            0.3755
    AR.51            -0.6834           -0.7313j            1.0009           -0.3696
    AR.52            -0.6834           +0.7313j            1.0009            0.3696
    AR.53            -0.7161           -0.6995j            1.0011           -0.3769
    AR.54            -0.7161           +0.6995j            1.0011            0.3769
    AR.55            -0.7271           -0.6869j            1.0002           -0.3795
    AR.56            -0.7271           +0.6869j            1.0002            0.3795
    AR.57            -0.6767           -0.7382j            1.0014           -0.3681
    AR.58            -0.6767           +0.7382j            1.0014            0.3681
    AR.59            -0.6711           -0.7437j            1.0018           -0.3668
    AR.60            -0.6711           +0.7437j            1.0018            0.3668
    AR.61            -0.7251           -0.6957j            1.0048           -0.3783
    AR.62            -0.7251           +0.6957j            1.0048            0.3783
    AR.63            -0.7365           -0.6789j            1.0017           -0.3815
    AR.64            -0.7365           +0.6789j            1.0017            0.3815
    AR.65            -0.6636           -0.7516j            1.0026           -0.3651
    AR.66            -0.6636           +0.7516j            1.0026            0.3651
    AR.67            -0.7435           -0.6732j            1.0029           -0.3829
    AR.68            -0.7435           +0.6732j            1.0029            0.3829
    AR.69            -0.6424           -0.7668j            1.0003           -0.3610
    AR.70            -0.6424           +0.7668j            1.0003            0.3610
    AR.71            -0.6522           -0.7601j            1.0016           -0.3629
    AR.72            -0.6522           +0.7601j            1.0016            0.3629
    AR.73            -0.7482           -0.6661j            1.0017           -0.3842
    AR.74            -0.7482           +0.6661j            1.0017            0.3842
    AR.75            -0.6605           -0.7561j            1.0040           -0.3643
    AR.76            -0.6605           +0.7561j            1.0040            0.3643
    AR.77            -0.7554           -0.6587j            1.0022           -0.3859
    AR.78            -0.7554           +0.6587j            1.0022            0.3859
    AR.79            -0.7592           -0.6521j            1.0009           -0.3871
    AR.80            -0.7592           +0.6521j            1.0009            0.3871
    AR.81             0.7771           -0.6307j            1.0008           -0.1085
    AR.82             0.7771           +0.6307j            1.0008            0.1085
    AR.83            -0.7658           -0.6455j            1.0016           -0.3885
    AR.84            -0.7658           +0.6455j            1.0016            0.3885
    AR.85            -0.7721           -0.6367j            1.0008           -0.3903
    AR.86            -0.7721           +0.6367j            1.0008            0.3903
    AR.87             0.7826           -0.6237j            1.0008           -0.1071
    AR.88             0.7826           +0.6237j            1.0008            0.1071
    AR.89            -0.7800           -0.6324j            1.0042           -0.3916
    AR.90            -0.7800           +0.6324j            1.0042            0.3916
    AR.91            -0.7896           -0.6149j            1.0007           -0.3947
    AR.92            -0.7896           +0.6149j            1.0007            0.3947
    AR.93            -0.7834           -0.6229j            1.0009           -0.3931
    AR.94            -0.7834           +0.6229j            1.0009            0.3931
    AR.95            -0.6358           -0.7744j            1.0019           -0.3594
    AR.96            -0.6358           +0.7744j            1.0019            0.3594
    AR.97             0.6322           -0.7755j            1.0006           -0.1411
    AR.98             0.6322           +0.7755j            1.0006            0.1411
    AR.99            -0.7931           -0.6106j            1.0009           -0.3956
    AR.100           -0.7931           +0.6106j            1.0009            0.3956
    AR.101            0.6228           -0.7823j            0.9999           -0.1430
    AR.102            0.6228           +0.7823j            0.9999            0.1430
    AR.103            0.7932           -0.6099j            1.0006           -0.1043
    AR.104            0.7932           +0.6099j            1.0006            0.1043
    AR.105           -0.6203           -0.7866j            1.0017           -0.3563
    AR.106           -0.6203           +0.7866j            1.0017            0.3563
    AR.107           -0.6300           -0.7829j            1.0049           -0.3578
    AR.108           -0.6300           +0.7829j            1.0049            0.3578
    AR.109            0.7917           -0.6171j            1.0038           -0.1054
    AR.110            0.7917           +0.6171j            1.0038            0.1054
    AR.111            0.6075           -0.7947j            1.0003           -0.1461
    AR.112            0.6075           +0.7947j            1.0003            0.1461
    AR.113            0.7983           -0.6021j            0.9999           -0.1028
    AR.114            0.7983           +0.6021j            0.9999            0.1028
    AR.115           -0.7990           -0.6009j            0.9998           -0.3974
    AR.116           -0.7990           +0.6009j            0.9998            0.3974
    AR.117            0.6161           -0.7925j            1.0039           -0.1448
    AR.118            0.6161           +0.7925j            1.0039            0.1448
    AR.119            0.7203           -0.7174j            1.0166           -0.1247
    AR.120            0.7203           +0.7174j            1.0166            0.1247
    AR.121           -0.6085           -0.7952j            1.0014           -0.3540
    AR.122           -0.6085           +0.7952j            1.0014            0.3540
    AR.123            0.6002           -0.8016j            1.0014           -0.1477
    AR.124            0.6002           +0.8016j            1.0014            0.1477
    AR.125           -0.8068           -0.5927j            1.0011           -0.3992
    AR.126           -0.8068           +0.5927j            1.0011            0.3992
    AR.127           -0.8111           -0.5878j            1.0017           -0.4002
    AR.128           -0.8111           +0.5878j            1.0017            0.4002
    AR.129            0.6385           -0.7815j            1.0092           -0.1410
    AR.130            0.6385           +0.7815j            1.0092            0.1410
    AR.131           -0.6431           -0.7846j            1.0145           -0.3593
    AR.132           -0.6431           +0.7846j            1.0145            0.3593
    AR.133           -0.5985           -0.8015j            1.0003           -0.3521
    AR.134           -0.5985           +0.8015j            1.0003            0.3521
    AR.135            0.8050           -0.5953j            1.0012           -0.1014
    AR.136            0.8050           +0.5953j            1.0012            0.1014
    AR.137            0.5889           -0.8096j            1.0011           -0.1499
    AR.138            0.5889           +0.8096j            1.0011            0.1499
    AR.139            0.5805           -0.8146j            1.0003           -0.1515
    AR.140            0.5805           +0.8146j            1.0003            0.1515
    AR.141            0.5980           -0.8063j            1.0039           -0.1484
    AR.142            0.5980           +0.8063j            1.0039            0.1484
    AR.143           -0.5918           -0.8086j            1.0020           -0.3506
    AR.144           -0.5918           +0.8086j            1.0020            0.3506
    AR.145           -0.6169           -0.7985j            1.0090           -0.3547
    AR.146           -0.6169           +0.7985j            1.0090            0.3547
    AR.147           -0.8167           -0.5781j            1.0006           -0.4020
    AR.148           -0.8167           +0.5781j            1.0006            0.4020
    AR.149            0.8122           -0.5894j            1.0035           -0.0999
    AR.150            0.8122           +0.5894j            1.0035            0.0999
    AR.151            0.5742           -0.8220j            1.0027           -0.1530
    AR.152            0.5742           +0.8220j            1.0027            0.1530
    AR.153           -0.8216           -0.5717j            1.0009           -0.4032
    AR.154           -0.8216           +0.5717j            1.0009            0.4032
    AR.155            0.8204           -0.5739j            1.0012           -0.0971
    AR.156            0.8204           +0.5739j            1.0012            0.0971
    AR.157            0.5672           -0.8255j            1.0016           -0.1542
    AR.158            0.5672           +0.8255j            1.0016            0.1542
    AR.159           -0.8271           -0.5628j            1.0004           -0.4049
    AR.160           -0.8271           +0.5628j            1.0004            0.4049
    AR.161           -0.7709           -0.6765j            1.0256           -0.3854
    AR.162           -0.7709           +0.6765j            1.0256            0.3854
    AR.163            0.8188           -0.5854j            1.0066           -0.0988
    AR.164            0.8188           +0.5854j            1.0066            0.0988
    AR.165            0.5601           -0.8292j            1.0006           -0.1555
    AR.166            0.5601           +0.8292j            1.0006            0.1555
    AR.167           -0.5795           -0.8179j            1.0024           -0.3481
    AR.168           -0.5795           +0.8179j            1.0024            0.3481
    AR.169           -0.5747           -0.8222j            1.0032           -0.3471
    AR.170           -0.5747           +0.8222j            1.0032            0.3471
    AR.171            0.5518           -0.8354j            1.0012           -0.1571
    AR.172            0.5518           +0.8354j            1.0012            0.1571
    AR.173           -0.5657           -0.8257j            1.0009           -0.3456
    AR.174           -0.5657           +0.8257j            1.0009            0.3456
    AR.175           -0.5884           -0.8217j            1.0107           -0.3489
    AR.176           -0.5884           +0.8217j            1.0107            0.3489
    AR.177           -0.8363           -0.5504j            1.0012           -0.4074
    AR.178           -0.8363           +0.5504j            1.0012            0.4074
    AR.179           -0.8328           -0.5597j            1.0034           -0.4058
    AR.180           -0.8328           +0.5597j            1.0034            0.4058
    AR.181            0.8263           -0.5686j            1.0030           -0.0959
    AR.182            0.8263           +0.5686j            1.0030            0.0959
    AR.183            0.5450           -0.8408j            1.0020           -0.1585
    AR.184            0.5450           +0.8408j            1.0020            0.1585
    AR.185           -0.5554           -0.8344j            1.0024           -0.3435
    AR.186           -0.5554           +0.8344j            1.0024            0.3435
    AR.187            0.8339           -0.5651j            1.0074           -0.0948
    AR.188            0.8339           +0.5651j            1.0074            0.0948
    AR.189           -0.5514           -0.8360j            1.0015           -0.3428
    AR.190           -0.5514           +0.8360j            1.0015            0.3428
    AR.191           -0.8443           -0.5419j            1.0032           -0.4092
    AR.192           -0.8443           +0.5419j            1.0032            0.4092
    AR.193            0.8408           -0.5421j            1.0005           -0.0911
    AR.194            0.8408           +0.5421j            1.0005            0.0911
    AR.195            0.8368           -0.5549j            1.0041           -0.0932
    AR.196            0.8368           +0.5549j            1.0041            0.0932
    AR.197            0.5363           -0.8458j            1.0015           -0.1601
    AR.198            0.5363           +0.8458j            1.0015            0.1601
    AR.199           -0.8484           -0.5338j            1.0024           -0.4106
    AR.200           -0.8484           +0.5338j            1.0024            0.4106
    AR.201            0.8487           -0.5338j            1.0026           -0.0894
    AR.202            0.8487           +0.5338j            1.0026            0.0894
    AR.203           -0.5389           -0.8427j            1.0002           -0.3405
    AR.204           -0.5389           +0.8427j            1.0002            0.3405
    AR.205            0.8406           -0.5552j            1.0074           -0.0929
    AR.206            0.8406           +0.5552j            1.0074            0.0929
    AR.207            0.5303           -0.8511j            1.0028           -0.1613
    AR.208            0.5303           +0.8511j            1.0028            0.1613
    AR.209            0.5218           -0.8528j            0.9997           -0.1626
    AR.210            0.5218           +0.8528j            0.9997            0.1626
    AR.211            0.8511           -0.5261j            1.0006           -0.0881
    AR.212            0.8511           +0.5261j            1.0006            0.0881
    AR.213            0.8553           -0.5191j            1.0004           -0.0868
    AR.214            0.8553           +0.5191j            1.0004            0.0868
    AR.215           -0.5383           -0.8509j            1.0069           -0.3398
    AR.216           -0.5383           +0.8509j            1.0069            0.3398
    AR.217           -0.5276           -0.8517j            1.0019           -0.3383
    AR.218           -0.5276           +0.8517j            1.0019            0.3383
    AR.219           -0.8535           -0.5325j            1.0060           -0.4112
    AR.220           -0.8535           +0.5325j            1.0060            0.4112
    AR.221            0.5071           -0.8619j            1.0000           -0.1654
    AR.222            0.5071           +0.8619j            1.0000            0.1654
    AR.223           -0.5220           -0.8548j            1.0015           -0.3373
    AR.224           -0.5220           +0.8548j            1.0015            0.3373
    AR.225           -0.8587           -0.5181j            1.0029           -0.4136
    AR.226           -0.8587           +0.5181j            1.0029            0.4136
    AR.227           -0.8588           -0.5255j            1.0068           -0.4126
    AR.228           -0.8588           +0.5255j            1.0068            0.4126
    AR.229            0.8619           -0.5081j            1.0005           -0.0848
    AR.230            0.8619           +0.5081j            1.0005            0.0848
    AR.231           -0.5123           -0.8593j            1.0004           -0.3356
    AR.232           -0.5123           +0.8593j            1.0004            0.3356
    AR.233           -0.8634           -0.5053j            1.0004           -0.4157
    AR.234           -0.8634           +0.5053j            1.0004            0.4157
    AR.235            0.5015           -0.8678j            1.0023           -0.1666
    AR.236            0.5015           +0.8678j            1.0023            0.1666
    AR.237           -0.5030           -0.8642j            0.9999           -0.3339
    AR.238           -0.5030           +0.8642j            0.9999            0.3339
    AR.239            0.8663           -0.5025j            1.0015           -0.0837
    AR.240            0.8663           +0.5025j            1.0015            0.0837
    AR.241           -0.8672           -0.4992j            1.0006           -0.4169
    AR.242           -0.8672           +0.4992j            1.0006            0.4169
    AR.243            0.5224           -0.8689j            1.0139           -0.1639
    AR.244            0.5224           +0.8689j            1.0139            0.1639
    AR.245            0.4912           -0.8714j            1.0003           -0.1683
    AR.246            0.4912           +0.8714j            1.0003            0.1683
    AR.247           -0.8747           -0.4879j            1.0016           -0.4190
    AR.248           -0.8747           +0.4879j            1.0016            0.4190
    AR.249           -0.4925           -0.8726j            1.0020           -0.3318
    AR.250           -0.4925           +0.8726j            1.0020            0.3318
    AR.251           -0.4999           -0.8730j            1.0060           -0.3328
    AR.252           -0.4999           +0.8730j            1.0060            0.3328
    AR.253            0.4834           -0.8755j            1.0000           -0.1697
    AR.254            0.4834           +0.8755j            1.0000            0.1697
    AR.255           -0.4804           -0.8790j            1.0017           -0.3296
    AR.256           -0.4804           +0.8790j            1.0017            0.3296
    AR.257            0.8727           -0.4907j            1.0011           -0.0815
    AR.258            0.8727           +0.4907j            1.0011            0.0815
    AR.259            0.4739           -0.8810j            1.0004           -0.1715
    AR.260            0.4739           +0.8810j            1.0004            0.1715
    AR.261            0.8775           -0.4814j            1.0009           -0.0799
    AR.262            0.8775           +0.4814j            1.0009            0.0799
    AR.263            0.8778           -0.4968j            1.0086           -0.0820
    AR.264            0.8778           +0.4968j            1.0086            0.0820
    AR.265           -0.4670           -0.8856j            1.0012           -0.3272
    AR.266           -0.4670           +0.8856j            1.0012            0.3272
    AR.267           -0.4769           -0.8843j            1.0047           -0.3287
    AR.268           -0.4769           +0.8843j            1.0047            0.3287
    AR.269           -0.8812           -0.4782j            1.0026           -0.4209
    AR.270           -0.8812           +0.4782j            1.0026            0.4209
    AR.271            0.4715           -0.8880j            1.0054           -0.1723
    AR.272            0.4715           +0.8880j            1.0054            0.1723
    AR.273            0.8843           -0.4709j            1.0019           -0.0779
    AR.274            0.8843           +0.4709j            1.0019            0.0779
    AR.275           -0.4631           -0.8886j            1.0021           -0.3265
    AR.276           -0.4631           +0.8886j            1.0021            0.3265
    AR.277           -0.4524           -0.8919j            1.0001           -0.3247
    AR.278           -0.4524           +0.8919j            1.0001            0.3247
    AR.279           -0.8855           -0.4700j            1.0025           -0.4223
    AR.280           -0.8855           +0.4700j            1.0025            0.4223
    AR.281           -0.8895           -0.4627j            1.0026           -0.4237
    AR.282           -0.8895           +0.4627j            1.0026            0.4237
    AR.283            0.4626           -0.8906j            1.0036           -0.1738
    AR.284            0.4626           +0.8906j            1.0036            0.1738
    AR.285            0.4577           -0.8925j            1.0030           -0.1746
    AR.286            0.4577           +0.8925j            1.0030            0.1746
    AR.287            0.8869           -0.4636j            1.0007           -0.0767
    AR.288            0.8869           +0.4636j            1.0007            0.0767
    AR.289           -0.8918           -0.4855j            1.0154           -0.4207
    AR.290           -0.8918           +0.4855j            1.0154            0.4207
    AR.291           -0.4434           -0.8976j            1.0012           -0.3230
    AR.292           -0.4434           +0.8976j            1.0012            0.3230
    AR.293            0.4467           -0.8959j            1.0011           -0.1764
    AR.294            0.4467           +0.8959j            1.0011            0.1764
    AR.295            0.8936           -0.4543j            1.0025           -0.0749
    AR.296            0.8936           +0.4543j            1.0025            0.0749
    AR.297           -0.8965           -0.4491j            1.0027           -0.4261
    AR.298           -0.8965           +0.4491j            1.0027            0.4261
    AR.299           -0.4347           -0.9011j            1.0004           -0.3215
    AR.300           -0.4347           +0.9011j            1.0004            0.3215
    AR.301            0.4376           -0.9001j            1.0009           -0.1780
    AR.302            0.4376           +0.9001j            1.0009            0.1780
    AR.303            0.8954           -0.4464j            1.0005           -0.0736
    AR.304            0.8954           +0.4464j            1.0005            0.0736
    AR.305            0.4303           -0.9044j            1.0015           -0.1793
    AR.306            0.4303           +0.9044j            1.0015            0.1793
    AR.307            0.8991           -0.4392j            1.0007           -0.0723
    AR.308            0.8991           +0.4392j            1.0007            0.0723
    AR.309           -0.4292           -0.9073j            1.0037           -0.3203
    AR.310           -0.4292           +0.9073j            1.0037            0.3203
    AR.311           -0.9015           -0.4527j            1.0087           -0.4259
    AR.312           -0.9015           +0.4527j            1.0087            0.4259
    AR.313           -0.9014           -0.4361j            1.0013           -0.4283
    AR.314           -0.9014           +0.4361j            1.0013            0.4283
    AR.315           -0.4201           -0.9087j            1.0011           -0.3189
    AR.316           -0.4201           +0.9087j            1.0011            0.3189
    AR.317            0.9037           -0.4289j            1.0003           -0.0705
    AR.318            0.9037           +0.4289j            1.0003            0.0705
    AR.319           -0.9066           -0.4250j            1.0013           -0.4302
    AR.320           -0.9066           +0.4250j            1.0013            0.4302
    AR.321            0.4228           -0.9088j            1.0023           -0.1807
    AR.322            0.4228           +0.9088j            1.0023            0.1807
    AR.323           -0.4118           -0.9134j            1.0020           -0.3174
    AR.324           -0.4118           +0.9134j            1.0020            0.3174
    AR.325            0.9085           -0.4204j            1.0010           -0.0690
    AR.326            0.9085           +0.4204j            1.0010            0.0690
    AR.327            0.9118           -0.4090j            0.9993           -0.0671
    AR.328            0.9118           +0.4090j            0.9993            0.0671
    AR.329            0.4174           -0.9171j            1.0076           -0.1820
    AR.330            0.4174           +0.9171j            1.0076            0.1820
    AR.331            0.4095           -0.9152j            1.0027           -0.1830
    AR.332            0.4095           +0.9152j            1.0027            0.1830
    AR.333           -0.4052           -0.9161j            1.0017           -0.3163
    AR.334           -0.4052           +0.9161j            1.0017            0.3163
    AR.335            0.4021           -0.9163j            1.0006           -0.1842
    AR.336            0.4021           +0.9163j            1.0006            0.1842
    AR.337           -0.9115           -0.4145j            1.0013           -0.4321
    AR.338           -0.9115           +0.4145j            1.0013            0.4321
    AR.339           -0.9143           -0.4577j            1.0225           -0.4261
    AR.340           -0.9143           +0.4577j            1.0225            0.4261
    AR.341           -0.3962           -0.9190j            1.0008           -0.3148
    AR.342           -0.3962           +0.9190j            1.0008            0.3148
    AR.343            0.9204           -0.4312j            1.0164           -0.0697
    AR.344            0.9204           +0.4312j            1.0164            0.0697
    AR.345           -0.9158           -0.4164j            1.0060           -0.4321
    AR.346           -0.9158           +0.4164j            1.0060            0.4321
    AR.347            0.9167           -0.4014j            1.0008           -0.0657
    AR.348            0.9167           +0.4014j            1.0008            0.0657
    AR.349            0.3894           -0.9215j            1.0004           -0.1864
    AR.350            0.3894           +0.9215j            1.0004            0.1864
    AR.351           -0.9167           -0.4002j            1.0003           -0.4345
    AR.352           -0.9167           +0.4002j            1.0003            0.4345
    AR.353           -0.3870           -0.9219j            0.9998           -0.3133
    AR.354           -0.3870           +0.9219j            0.9998            0.3133
    AR.355            0.3831           -0.9233j            0.9996           -0.1874
    AR.356            0.3831           +0.9233j            0.9996            0.1874
    AR.357            0.9209           -0.3925j            1.0011           -0.0641
    AR.358            0.9209           +0.3925j            1.0011            0.0641
    AR.359           -0.3772           -0.9260j            0.9999           -0.3116
    AR.360           -0.3772           +0.9260j            0.9999            0.3116
    AR.361            0.9256           -0.3885j            1.0038           -0.0633
    AR.362            0.9256           +0.3885j            1.0038            0.0633
    AR.363           -0.9202           -0.3946j            1.0012           -0.4355
    AR.364           -0.9202           +0.3946j            1.0012            0.4355
    AR.365            0.3733           -0.9292j            1.0014           -0.1892
    AR.366            0.3733           +0.9292j            1.0014            0.1892
    AR.367           -0.9250           -0.3838j            1.0015           -0.4374
    AR.368           -0.9250           +0.3838j            1.0015            0.4374
    AR.369           -0.3697           -0.9314j            1.0021           -0.3101
    AR.370           -0.3697           +0.9314j            1.0021            0.3101
    AR.371            0.3652           -0.9332j            1.0021           -0.1906
    AR.372            0.3652           +0.9332j            1.0021            0.1906
    AR.373            0.9288           -0.3770j            1.0024           -0.0614
    AR.374            0.9288           +0.3770j            1.0024            0.0614
    AR.375           -0.9276           -0.3754j            1.0007           -0.4388
    AR.376           -0.9276           +0.3754j            1.0007            0.4388
    AR.377            0.9301           -0.3680j            1.0003           -0.0600
    AR.378            0.9301           +0.3680j            1.0003            0.0600
    AR.379           -0.3623           -0.9334j            1.0013           -0.3089
    AR.380           -0.3623           +0.9334j            1.0013            0.3089
    AR.381            0.3571           -0.9355j            1.0013           -0.1920
    AR.382            0.3571           +0.9355j            1.0013            0.1920
    AR.383           -0.9320           -0.3647j            1.0008           -0.4406
    AR.384           -0.9320           +0.3647j            1.0008            0.4406
    AR.385           -0.3551           -0.9371j            1.0021           -0.3076
    AR.386           -0.3551           +0.9371j            1.0021            0.3076
    AR.387           -0.9359           -0.3634j            1.0039           -0.4411
    AR.388           -0.9359           +0.3634j            1.0039            0.4411
    AR.389            0.3469           -0.9404j            1.0023           -0.1938
    AR.390            0.3469           +0.9404j            1.0023            0.1938
    AR.391            0.9374           -0.3622j            1.0049           -0.0587
    AR.392            0.9374           +0.3622j            1.0049            0.0587
    AR.393           -0.3466           -0.9402j            1.0020           -0.3062
    AR.394           -0.3466           +0.9402j            1.0020            0.3062
    AR.395           -0.3397           -0.9420j            1.0014           -0.3051
    AR.396           -0.3397           +0.9420j            1.0014            0.3051
    AR.397           -0.9405           -0.3498j            1.0034           -0.4433
    AR.398           -0.9405           +0.3498j            1.0034            0.4433
    AR.399           -0.9418           -0.3435j            1.0025           -0.4443
    AR.400           -0.9418           +0.3435j            1.0025            0.4443
    AR.401           -0.9446           -0.3339j            1.0019           -0.4459
    AR.402           -0.9446           +0.3339j            1.0019            0.4459
    AR.403            0.9369           -0.3533j            1.0012           -0.0574
    AR.404            0.9369           +0.3533j            1.0012            0.0574
    AR.405           -0.3299           -0.9444j            1.0004           -0.3035
    AR.406           -0.3299           +0.9444j            1.0004            0.3035
    AR.407            0.3444           -0.9445j            1.0054           -0.1944
    AR.408            0.3444           +0.9445j            1.0054            0.1944
    AR.409            0.9388           -0.3424j            0.9993           -0.0557
    AR.410            0.9388           +0.3424j            0.9993            0.0557
    AR.411            0.3356           -0.9453j            1.0031           -0.1957
    AR.412            0.3356           +0.9453j            1.0031            0.1957
    AR.413            0.9419           -0.3357j            0.9999           -0.0545
    AR.414            0.9419           +0.3357j            0.9999            0.0545
    AR.415           -0.3209           -0.9464j            0.9993           -0.3020
    AR.416           -0.3209           +0.9464j            0.9993            0.3020
    AR.417            0.3235           -0.9465j            1.0003           -0.1976
    AR.418            0.3235           +0.9465j            1.0003            0.1976
    AR.419            0.3117           -0.9505j            1.0003           -0.1996
    AR.420            0.3117           +0.9505j            1.0003            0.1996
    AR.421            0.9447           -0.3265j            0.9996           -0.0530
    AR.422            0.9447           +0.3265j            0.9996            0.0530
    AR.423           -0.9487           -0.3223j            1.0020           -0.4479
    AR.424           -0.9487           +0.3223j            1.0020            0.4479
    AR.425            0.9485           -0.3184j            1.0005           -0.0515
    AR.426            0.9485           +0.3184j            1.0005            0.0515
    AR.427            0.3308           -0.9592j            1.0147           -0.1971
    AR.428            0.3308           +0.9592j            1.0147            0.1971
    AR.429            0.3017           -0.9535j            1.0001           -0.2012
    AR.430            0.3017           +0.9535j            1.0001            0.2012
    AR.431           -0.3111           -0.9508j            1.0004           -0.3003
    AR.432           -0.3111           +0.9508j            1.0004            0.3003
    AR.433            0.9519           -0.3109j            1.0013           -0.0502
    AR.434            0.9519           +0.3109j            1.0013            0.0502
    AR.435           -0.3012           -0.9546j            1.0010           -0.2986
    AR.436           -0.3012           +0.9546j            1.0010            0.2986
    AR.437           -0.9518           -0.3118j            1.0016           -0.4496
    AR.438           -0.9518           +0.3118j            1.0016            0.4496
    AR.439           -0.9541           -0.3198j            1.0062           -0.4485
    AR.440           -0.9541           +0.3198j            1.0062            0.4485
    AR.441           -0.2899           -0.9592j            1.0020           -0.2967
    AR.442           -0.2899           +0.9592j            1.0020            0.2967
    AR.443           -0.2990           -0.9652j            1.0105           -0.2978
    AR.444           -0.2990           +0.9652j            1.0105            0.2978
    AR.445            0.2922           -0.9601j            1.0035           -0.2030
    AR.446            0.2922           +0.9601j            1.0035            0.2030
    AR.447            0.9558           -0.3004j            1.0019           -0.0485
    AR.448            0.9558           +0.3004j            1.0019            0.0485
    AR.449            0.2855           -0.9602j            1.0017           -0.2040
    AR.450            0.2855           +0.9602j            1.0017            0.2040
    AR.451            0.9592           -0.2938j            1.0031           -0.0473
    AR.452            0.9592           +0.2938j            1.0031            0.0473
    AR.453           -0.9571           -0.2984j            1.0025           -0.4519
    AR.454           -0.9571           +0.2984j            1.0025            0.4519
    AR.455            0.2752           -0.9624j            1.0010           -0.2057
    AR.456            0.2752           +0.9624j            1.0010            0.2057
    AR.457            0.9634           -0.2808j            1.0035           -0.0451
    AR.458            0.9634           +0.2808j            1.0035            0.0451
    AR.459           -0.9622           -0.2834j            1.0030           -0.4544
    AR.460           -0.9622           +0.2834j            1.0030            0.4544
    AR.461            0.2687           -0.9662j            1.0028           -0.2068
    AR.462            0.2687           +0.9662j            1.0028            0.2068
    AR.463            0.9635           -0.2739j            1.0017           -0.0441
    AR.464            0.9635           +0.2739j            1.0017            0.0441
    AR.465            0.2595           -0.9673j            1.0015           -0.2083
    AR.466            0.2595           +0.9673j            1.0015            0.2083
    AR.467           -0.9671           -0.2758j            1.0057           -0.4558
    AR.468           -0.9671           +0.2758j            1.0057            0.4558
    AR.469           -0.9666           -0.2921j            1.0097           -0.4533
    AR.470           -0.9666           +0.2921j            1.0097            0.4533
    AR.471           -0.2795           -0.9653j            1.0049           -0.2949
    AR.472           -0.2795           +0.9653j            1.0049            0.2949
    AR.473           -0.2712           -0.9650j            1.0024           -0.2936
    AR.474           -0.2712           +0.9650j            1.0024            0.2936
    AR.475            0.2517           -0.9691j            1.0012           -0.2096
    AR.476            0.2517           +0.9691j            1.0012            0.2096
    AR.477            0.9676           -0.2601j            1.0020           -0.0418
    AR.478            0.9676           +0.2601j            1.0020            0.0418
    AR.479           -0.9676           -0.2672j            1.0038           -0.4571
    AR.480           -0.9676           +0.2672j            1.0038            0.4571
    AR.481           -0.9672           -0.2562j            1.0006           -0.4588
    AR.482           -0.9672           +0.2562j            1.0006            0.4588
    AR.483            0.2404           -0.9708j            1.0001           -0.2114
    AR.484            0.2404           +0.9708j            1.0001            0.2114
    AR.485           -0.2657           -0.9675j            1.0033           -0.2927
    AR.486           -0.2657           +0.9675j            1.0033            0.2927
    AR.487           -0.2569           -0.9672j            1.0007           -0.2913
    AR.488           -0.2569           +0.9672j            1.0007            0.2913
    AR.489           -0.2481           -0.9705j            1.0017           -0.2898
    AR.490           -0.2481           +0.9705j            1.0017            0.2898
    AR.491            0.2256           -0.9767j            1.0024           -0.2139
    AR.492            0.2256           +0.9767j            1.0024            0.2139
    AR.493            0.2352           -0.9802j            1.0081           -0.2125
    AR.494            0.2352           +0.9802j            1.0081            0.2125
    AR.495           -0.2374           -0.9758j            1.0043           -0.2880
    AR.496           -0.2374           +0.9758j            1.0043            0.2880
    AR.497           -0.2291           -0.9730j            0.9996           -0.2868
    AR.498           -0.2291           +0.9730j            0.9996            0.2868
    AR.499            0.9731           -0.2824j            1.0132           -0.0449
    AR.500            0.9731           +0.2824j            1.0132            0.0449
    AR.501            0.9717           -0.2486j            1.0030           -0.0399
    AR.502            0.9717           +0.2486j            1.0030            0.0399
    AR.503            0.9720           -0.2364j            1.0004           -0.0380
    AR.504            0.9720           +0.2364j            1.0004            0.0380
    AR.505           -0.9697           -0.2446j            1.0001           -0.4607
    AR.506           -0.9697           +0.2446j            1.0001            0.4607
    AR.507            0.9778           -0.2304j            1.0046           -0.0368
    AR.508            0.9778           +0.2304j            1.0046            0.0368
    AR.509            0.9767           -0.2198j            1.0012           -0.0352
    AR.510            0.9767           +0.2198j            1.0012            0.0352
    AR.511           -0.9738           -0.2335j            1.0014           -0.4626
    AR.512           -0.9738           +0.2335j            1.0014            0.4626
    AR.513           -0.9866           -0.2856j            1.0271           -0.4552
    AR.514           -0.9866           +0.2856j            1.0271            0.4552
    AR.515           -0.9765           -0.2268j            1.0025           -0.4637
    AR.516           -0.9765           +0.2268j            1.0025            0.4637
    AR.517            0.9786           -0.2521j            1.0106           -0.0401
    AR.518            0.9786           +0.2521j            1.0106            0.0401
    AR.519            0.9783           -0.2111j            1.0008           -0.0338
    AR.520            0.9783           +0.2111j            1.0008            0.0338
    AR.521           -0.9788           -0.2170j            1.0026           -0.4653
    AR.522           -0.9788           +0.2170j            1.0026            0.4653
    AR.523            0.2188           -0.9782j            1.0024           -0.2150
    AR.524            0.2188           +0.9782j            1.0024            0.2150
    AR.525           -0.9818           -0.2040j            1.0028           -0.4674
    AR.526           -0.9818           +0.2040j            1.0028            0.4674
    AR.527            0.9800           -0.2007j            1.0004           -0.0322
    AR.528            0.9800           +0.2007j            1.0004            0.0322
    AR.529           -0.2191           -0.9766j            1.0008           -0.2851
    AR.530           -0.2191           +0.9766j            1.0008            0.2851
    AR.531            0.2088           -0.9807j            1.0027           -0.2166
    AR.532            0.2088           +0.9807j            1.0027            0.2166
    AR.533           -0.9813           -0.1961j            1.0007           -0.4686
    AR.534           -0.9813           +0.1961j            1.0007            0.4686
    AR.535           -0.9852           -0.1880j            1.0029           -0.4700
    AR.536           -0.9852           +0.1880j            1.0029            0.4700
    AR.537           -0.9890           -0.2239j            1.0140           -0.4646
    AR.538           -0.9890           +0.2239j            1.0140            0.4646
    AR.539            0.9848           -0.1922j            1.0033           -0.0307
    AR.540            0.9848           +0.1922j            1.0033            0.0307
    AR.541            0.9844           -0.1829j            1.0013           -0.0292
    AR.542            0.9844           +0.1829j            1.0013            0.0292
    AR.543            0.9854           -0.1740j            1.0006           -0.0278
    AR.544            0.9854           +0.1740j            1.0006            0.0278
    AR.545            0.9884           -0.1636j            1.0019           -0.0261
    AR.546            0.9884           +0.1636j            1.0019            0.0261
    AR.547           -0.2074           -0.9793j            1.0011           -0.2832
    AR.548           -0.2074           +0.9793j            1.0011            0.2832
    AR.549           -0.9857           -0.1783j            1.0017           -0.4715
    AR.550           -0.9857           +0.1783j            1.0017            0.4715
    AR.551           -0.2026           -0.9828j            1.0034           -0.2824
    AR.552           -0.2026           +0.9828j            1.0034            0.2824
    AR.553            0.1993           -0.9822j            1.0022           -0.2181
    AR.554            0.1993           +0.9822j            1.0022            0.2181
    AR.555           -0.9862           -0.1700j            1.0007           -0.4728
    AR.556           -0.9862           +0.1700j            1.0007            0.4728
    AR.557            0.1782           -0.9861j            1.0021           -0.2215
    AR.558            0.1782           +0.9861j            1.0021            0.2215
    AR.559            0.1680           -0.9868j            1.0010           -0.2232
    AR.560            0.1680           +0.9868j            1.0010            0.2232
    AR.561            0.1844           -0.9854j            1.0025           -0.2206
    AR.562            0.1844           +0.9854j            1.0025            0.2206
    AR.563           -0.9867           -0.1585j            0.9994           -0.4746
    AR.564           -0.9867           +0.1585j            0.9994            0.4746
    AR.565           -0.1909           -0.9834j            1.0017           -0.2805
    AR.566           -0.1909           +0.9834j            1.0017            0.2805
    AR.567            0.1618           -0.9895j            1.0027           -0.2242
    AR.568            0.1618           +0.9895j            1.0027            0.2242
    AR.569            0.1925           -0.9878j            1.0064           -0.2194
    AR.570            0.1925           +0.9878j            1.0064            0.2194
    AR.571           -0.1814           -0.9872j            1.0037           -0.2789
    AR.572           -0.1814           +0.9872j            1.0037            0.2789
    AR.573           -0.9895           -0.1477j            1.0004           -0.4764
    AR.574           -0.9895           +0.1477j            1.0004            0.4764
    AR.575           -0.1691           -0.9888j            1.0031           -0.2770
    AR.576           -0.1691           +0.9888j            1.0031            0.2770
    AR.577           -0.9917           -0.1366j            1.0010           -0.4782
    AR.578           -0.9917           +0.1366j            1.0010            0.4782
    AR.579            0.1486           -0.9903j            1.0013           -0.2263
    AR.580            0.1486           +0.9903j            1.0013            0.2263
    AR.581           -0.9926           -0.1249j            1.0004           -0.4801
    AR.582           -0.9926           +0.1249j            1.0004            0.4801
    AR.583            0.1399           -0.9920j            1.0018           -0.2277
    AR.584            0.1399           +0.9920j            1.0018            0.2277
    AR.585            0.1319           -0.9917j            1.0005           -0.2290
    AR.586            0.1319           +0.9917j            1.0005            0.2290
    AR.587            0.9906           -0.1528j            1.0023           -0.0243
    AR.588            0.9906           +0.1528j            1.0023            0.0243
    AR.589            0.9912           -0.1427j            1.0014           -0.0228
    AR.590            0.9912           +0.1427j            1.0014            0.0228
    AR.591            0.9931           -0.1321j            1.0018           -0.0210
    AR.592            0.9931           +0.1321j            1.0018            0.0210
    AR.593            0.9923           -0.1245j            1.0001           -0.0199
    AR.594            0.9923           +0.1245j            1.0001            0.0199
    AR.595            0.9950           -0.1122j            1.0013           -0.0179
    AR.596            0.9950           +0.1122j            1.0013            0.0179
    AR.597           -0.1594           -0.9902j            1.0029           -0.2754
    AR.598           -0.1594           +0.9902j            1.0029            0.2754
    AR.599           -0.1522           -0.9888j            1.0004           -0.2743
    AR.600           -0.1522           +0.9888j            1.0004            0.2743
    AR.601            0.1205           -0.9940j            1.0013           -0.2308
    AR.602            0.1205           +0.9940j            1.0013            0.2308
    AR.603           -0.1381           -0.9937j            1.0032           -0.2720
    AR.604           -0.1381           +0.9937j            1.0032            0.2720
    AR.605           -0.1323           -0.9932j            1.0020           -0.2711
    AR.606           -0.1323           +0.9932j            1.0020            0.2711
    AR.607            0.0408           -0.9993j            1.0001           -0.2435
    AR.608            0.0408           +0.9993j            1.0001            0.2435
    AR.609            0.0519           -1.0002j            1.0015           -0.2417
    AR.610            0.0519           +1.0002j            1.0015            0.2417
    AR.611            0.0323           -0.9998j            1.0003           -0.2449
    AR.612            0.0323           +0.9998j            1.0003            0.2449
    AR.613            0.0601           -0.9995j            1.0013           -0.2404
    AR.614            0.0601           +0.9995j            1.0013            0.2404
    AR.615            0.0226           -0.9999j            1.0002           -0.2464
    AR.616            0.0226           +0.9999j            1.0002            0.2464
    AR.617           -0.9948           -0.1170j            1.0017           -0.4814
    AR.618           -0.9948           +0.1170j            1.0017            0.4814
    AR.619           -0.9970           -0.1084j            1.0029           -0.4828
    AR.620           -0.9970           +0.1084j            1.0029            0.4828
    AR.621           -1.0002           -0.0310j            1.0007           -0.4951
    AR.622           -1.0002           +0.0310j            1.0007            0.4951
    AR.623           -1.0007           -0.0074j            1.0007           -0.4988
    AR.624           -1.0007           +0.0074j            1.0007            0.4988
    AR.625           -1.0006           -0.0194j            1.0008           -0.4969
    AR.626           -1.0006           +0.0194j            1.0008            0.4969
    AR.627           -1.0011           -0.0418j            1.0020           -0.4934
    AR.628           -1.0011           +0.0418j            1.0020            0.4934
    AR.629            0.0698           -0.9991j            1.0016           -0.2389
    AR.630            0.0698           +0.9991j            1.0016            0.2389
    AR.631           -1.0011           -0.0493j            1.0023           -0.4922
    AR.632           -1.0011           +0.0493j            1.0023            0.4922
    AR.633           -0.9964           -0.0981j            1.0012           -0.4844
    AR.634           -0.9964           +0.0981j            1.0012            0.4844
    AR.635           -1.0004           -0.0569j            1.0020           -0.4910
    AR.636           -1.0004           +0.0569j            1.0020            0.4910
    AR.637           -1.0001           -0.0674j            1.0024           -0.4893
    AR.638           -1.0001           +0.0674j            1.0024            0.4893
    AR.639           -0.1207           -0.9968j            1.0041           -0.2692
    AR.640           -0.1207           +0.9968j            1.0041            0.2692
    AR.641            0.0896           -0.9975j            1.0016           -0.2357
    AR.642            0.0896           +0.9975j            1.0016            0.2357
    AR.643            0.1129           -0.9978j            1.0041           -0.2321
    AR.644            0.1129           +0.9978j            1.0041            0.2321
    AR.645            0.0980           -0.9964j            1.0012           -0.2344
    AR.646            0.0980           +0.9964j            1.0012            0.2344
    AR.647            0.9987           -0.0700j            1.0012           -0.0111
    AR.648            0.9987           +0.0700j            1.0012            0.0111
    AR.649            0.9991           -0.0803j            1.0023           -0.0128
    AR.650            0.9991           +0.0803j            1.0023            0.0128
    AR.651            0.9968           -0.1009j            1.0019           -0.0160
    AR.652            0.9968           +0.1009j            1.0019            0.0160
    AR.653           -0.1791           -1.0037j            1.0195           -0.2781
    AR.654           -0.1791           +1.0037j            1.0195            0.2781
    AR.655            0.0117           -1.0006j            1.0007           -0.2481
    AR.656            0.0117           +1.0006j            1.0007            0.2481
    AR.657            0.1069           -0.9965j            1.0022           -0.2330
    AR.658            0.1069           +0.9965j            1.0022            0.2330
    AR.659            0.0791           -0.9978j            1.0009           -0.2374
    AR.660            0.0791           +0.9978j            1.0009            0.2374
    AR.661            0.9987           -0.0635j            1.0007           -0.0101
    AR.662            0.9987           +0.0635j            1.0007            0.0101
    AR.663            0.9981           -0.0882j            1.0020           -0.0140
    AR.664            0.9981           +0.0882j            1.0020            0.0140
    AR.665           -0.0968           -0.9967j            1.0014           -0.2654
    AR.666           -0.0968           +0.9967j            1.0014            0.2654
    AR.667           -0.0775           -0.9974j            1.0004           -0.2623
    AR.668           -0.0775           +0.9974j            1.0004            0.2623
    AR.669           -0.0611           -0.9984j            1.0003           -0.2597
    AR.670           -0.0611           +0.9984j            1.0003            0.2597
    AR.671           -0.1111           -0.9971j            1.0033           -0.2677
    AR.672           -0.1111           +0.9971j            1.0033            0.2677
    AR.673           -0.0684           -1.0005j            1.0028           -0.2609
    AR.674           -0.0684           +1.0005j            1.0028            0.2609
    AR.675            0.0027           -1.0017j            1.0017           -0.2496
    AR.676            0.0027           +1.0017j            1.0017            0.2496
    AR.677            1.0009           -0.0000j            1.0009           -0.0000
    AR.678            1.0009           -0.0512j            1.0022           -0.0081
    AR.679            1.0009           +0.0512j            1.0022            0.0081
    AR.680           -0.0873           -1.0001j            1.0039           -0.2639
    AR.681           -0.0873           +1.0001j            1.0039            0.2639
    AR.682           -1.0087           -0.0161j            1.0088           -0.4975
    AR.683           -1.0087           +0.0161j            1.0088            0.4975
    AR.684           -1.0021           -0.0780j            1.0051           -0.4876
    AR.685           -1.0021           +0.0780j            1.0051            0.4876
    AR.686           -1.0020           -0.0851j            1.0056           -0.4865
    AR.687           -1.0020           +0.0851j            1.0056            0.4865
    AR.688           -1.0081           -0.0870j            1.0118           -0.4863
    AR.689           -1.0081           +0.0870j            1.0118            0.4863
    AR.690            1.0026           -0.0120j            1.0027           -0.0019
    AR.691            1.0026           +0.0120j            1.0027            0.0019
    AR.692           -0.0334           -1.0014j            1.0020           -0.2553
    AR.693           -0.0334           +1.0014j            1.0020            0.2553
    AR.694           -0.0232           -1.0013j            1.0016           -0.2537
    AR.695           -0.0232           +1.0013j            1.0016            0.2537
    AR.696           -0.0140           -1.0012j            1.0013           -0.2522
    AR.697           -0.0140           +1.0012j            1.0013            0.2522
    AR.698           -0.0487           -1.0013j            1.0025           -0.2577
    AR.699           -0.0487           +1.0013j            1.0025            0.2577
    AR.700           -0.0430           -1.0011j            1.0020           -0.2568
    AR.701           -0.0430           +1.0011j            1.0020            0.2568
    AR.702           -0.0077           -1.0034j            1.0034           -0.2512
    AR.703           -0.0077           +1.0034j            1.0034            0.2512
    AR.704            1.0020           -0.0416j            1.0029           -0.0066
    AR.705            1.0020           +0.0416j            1.0029            0.0066
    AR.706            1.0044           -0.0175j            1.0045           -0.0028
    AR.707            1.0044           +0.0175j            1.0045            0.0028
    AR.708            1.0027           -0.0356j            1.0034           -0.0056
    AR.709            1.0027           +0.0356j            1.0034            0.0056
    AR.710           -0.1094           -0.9988j            1.0047           -0.2674
    AR.711           -0.1094           +0.9988j            1.0047            0.2674
    AR.712            1.0036           -0.0265j            1.0039           -0.0042
    AR.713            1.0036           +0.0265j            1.0039            0.0042
    AR.714            0.9994           -0.1583j            1.0119           -0.0250
    AR.715            0.9994           +0.1583j            1.0119            0.0250
    AR.716            1.0029           -0.1028j            1.0081           -0.0163
    AR.717            1.0029           +0.1028j            1.0081            0.0163
    AR.718           -1.0178           -0.1426j            1.0278           -0.4778
    AR.719           -1.0178           +0.1426j            1.0278            0.4778
    AR.720            0.0102           -1.0268j            1.0268           -0.2484
    AR.721            0.0102           +1.0268j            1.0268            0.2484
    AR.722           -0.2051           -1.0258j            1.0461           -0.2814
    AR.723           -0.2051           +1.0258j            1.0461            0.2814
    AR.724            1.2523           -0.1724j            1.2641           -0.0218
    AR.725            1.2523           +0.1724j            1.2641            0.0218
    AR.726            0.0037           -1.4371j            1.4371           -0.2496
    AR.727            0.0037           +1.4371j            1.4371            0.2496
    AR.728            1.1093           -1.1507j            1.5983           -0.1279
    AR.729            1.1093           +1.1507j            1.5983            0.1279
    AR.730           -1.6457           -0.0000j            1.6457           -0.5000
    -------------------------------------------------------------------------------
    μ=0.3325936766807077 ,ϕ=[ 2.58867966e-02  1.50169263e-02 -2.83888335e-02 -2.38492607e-02
     -9.41744148e-03 -1.89069070e-03  1.40430361e-02 -3.81961700e-02
      7.24682720e-03  1.43804748e-03  3.14759404e-03 -1.07460718e-02
     -3.34677859e-02  4.10429099e-03 -1.42104617e-02  4.98610516e-03
      4.94469436e-02 -9.85422578e-03  1.85431824e-02  5.65926682e-02
     -5.25286433e-03 -3.58457804e-02 -2.96852375e-02  1.50006957e-02
     -7.41137395e-02 -3.01201801e-02  2.75689945e-03  4.32313671e-02
     -5.30092298e-02 -3.21122479e-02  1.24582336e-02  2.68331419e-02
      4.34277321e-02 -1.73488686e-02  3.05129866e-03  1.49829616e-02
     -4.21006170e-02 -5.55918921e-02  5.67434454e-02 -3.17144702e-03
      4.52153609e-02  9.13975274e-03 -6.66910613e-03 -1.58608296e-02
      1.23062409e-02 -1.34804891e-02  3.37398751e-02 -5.29229727e-02
     -2.89423131e-02 -6.61781775e-02 -2.85544578e-02  3.93337097e-02
     -9.91344871e-03  2.53076331e-05  1.09105174e-02  1.10143571e-02
      3.11327584e-02  1.93401141e-02  3.03023707e-02  7.31947816e-03
     -1.31061811e-02 -1.82233426e-03 -3.98839254e-02 -2.41317932e-02
     -2.97227864e-02 -6.53625493e-02 -1.38359347e-02  8.15616772e-03
      4.11259867e-02 -1.75760926e-02  1.88918522e-02  9.04680949e-03
      1.92683445e-02 -2.38484215e-02 -1.83269528e-02  5.43144231e-03
      5.66863104e-02  2.75126232e-02 -3.73106815e-02 -2.57890054e-02
      2.71980559e-02  5.74767424e-03  3.31170833e-02 -6.35469277e-02
      2.54454420e-02  1.34117858e-02  1.80491265e-03 -2.87804361e-02
      2.17719535e-02 -2.74476530e-02  1.60355636e-02 -1.89514450e-03
      6.75219752e-03  1.53760326e-02  3.01134994e-02  7.83204417e-03
     -3.46541423e-02  7.67405442e-03  2.52236987e-02 -1.58474648e-02
     -2.35753967e-02 -2.00319920e-02 -4.20378317e-02  2.07544201e-03
     -2.78012453e-02  5.71467715e-02  5.55008566e-02  8.71325026e-03
      6.38459745e-03 -2.27234152e-02 -2.17149034e-02  1.36292040e-02
      9.45596317e-03  1.83839952e-02 -5.52022523e-02  4.21184086e-02
     -1.52001208e-02 -4.99964633e-02  6.78538470e-03 -2.07724882e-02
     -1.83868353e-02  7.88807814e-03  3.81950682e-02 -2.29120133e-02
     -2.19365275e-02 -3.66226514e-02  1.62720699e-02 -3.81172869e-02
     -4.25412433e-03  1.38184360e-03 -1.01337825e-02 -3.22569519e-02
     -5.27751071e-02  3.93098413e-02  3.76323510e-02  3.22797693e-02
      1.56316695e-02 -1.41432691e-02  9.62555501e-04 -4.67937630e-02
     -2.58317565e-03 -2.25333592e-02  9.29157418e-04  1.63281849e-02
      5.24390115e-03 -2.72980939e-02  1.25344038e-02 -2.70523989e-02
      3.65326303e-02  2.87217343e-02 -2.74417854e-02 -1.39792577e-02
      7.80788737e-03  5.17451113e-03 -1.78047598e-02  3.84780141e-02
      1.18626725e-02  5.49586869e-03 -9.54954706e-03 -2.35984627e-02
      1.88867340e-02 -2.58140742e-03 -2.13172256e-02  9.41595943e-04
      7.17012630e-03 -5.75406467e-03  2.47527515e-02  1.53445764e-02
     -5.10494092e-02 -1.89205801e-02  2.70334126e-02 -2.39786305e-02
      1.46107128e-03 -1.51803105e-02  5.81282512e-02 -2.21035733e-02
      3.08906897e-03  2.19161459e-03  1.81944220e-02  3.68171863e-02
      2.55945996e-02 -3.43220993e-03  2.67108066e-02  4.61171709e-03
      5.77544165e-02 -2.57502561e-03  1.75398564e-02 -2.67660097e-03
     -1.51183139e-02 -1.44424344e-02 -3.35006832e-02  1.54632747e-02
     -5.14555422e-03 -3.58619137e-02  1.25630300e-02  1.90482664e-02
      5.69831779e-03  2.82424152e-02 -5.38137986e-03 -1.24926812e-02
     -2.74240815e-02  3.41667138e-02  1.92745755e-03  1.63037389e-02
      4.36483277e-02 -1.31464979e-02  2.45880848e-02 -5.29265092e-02
      1.28881071e-02  6.44857321e-03  2.37287139e-02 -3.30711566e-02
      2.01990902e-03  9.85801692e-03 -2.48363330e-02  7.88239846e-03
     -1.00598398e-02  3.84386849e-02  1.22746314e-02 -2.98201990e-02
     -2.92445209e-02 -3.74934149e-02 -1.48543528e-02 -1.66329416e-02
      1.81421888e-02 -4.97921767e-02 -2.44781843e-02 -9.82370543e-03
      1.13254097e-02 -2.21308988e-05 -6.76947221e-03 -2.65363452e-02
      3.89281635e-02  7.74980569e-03 -3.24589994e-02 -9.44470127e-05
     -7.74716833e-03 -2.36955989e-02  6.96512166e-03  7.32132126e-04
      2.77329000e-02 -3.07634136e-02 -3.94979236e-02 -4.42500953e-02
      6.57606934e-03  2.01690319e-02  2.18487204e-02  4.14435544e-02
     -2.45294026e-02  3.18765953e-02  1.58325884e-02  5.13788652e-03
     -1.69351588e-02 -6.19210421e-03 -4.49660285e-03 -3.49185596e-02
     -5.33012879e-02  3.09891019e-02  3.46237214e-03 -3.10860969e-02
      2.39009471e-02  6.28458915e-03 -3.84634376e-02 -2.01042420e-02
      3.84876497e-03 -6.77546310e-03 -6.31218667e-03 -1.05881957e-02
      3.10068146e-02 -2.91089475e-02  2.39708446e-02  3.08412971e-02
      2.54482504e-02 -1.07678155e-02 -5.37148798e-02 -1.14751233e-02
      3.94413797e-03 -2.03937272e-02  1.17475280e-02 -4.75606398e-02
     -2.85951589e-02 -6.20155311e-03  3.70277012e-02  2.95318538e-02
      3.65049993e-02  1.47082780e-02 -1.47840359e-03  3.65314970e-02
      2.48814512e-02  1.49379768e-02 -3.39325265e-02 -3.74838104e-02
     -1.25242422e-02  1.47503327e-02  5.07840972e-02  1.71930504e-02
      4.05218534e-02  2.14100588e-02  5.53770260e-02  8.18506045e-03
     -1.49658114e-02 -2.96364559e-02  1.52504155e-02 -4.20771591e-02
     -4.09134189e-02 -3.04590408e-02 -3.93068871e-03  5.09539345e-02
      1.02512407e-02  2.85621040e-02  1.28505371e-03 -5.10915953e-03
      2.98360482e-02  2.79542880e-02  2.86836429e-02  3.60580451e-03
      2.48275338e-02 -2.13939802e-02 -4.34374939e-02  3.22154349e-02
      2.32944283e-02 -1.13794210e-02 -5.65271835e-02  7.96969135e-03
     -3.56590820e-02 -1.54029215e-02  1.11909598e-02 -3.65767944e-02
     -2.39526138e-02 -5.87148139e-02  3.44793920e-02  7.35169248e-03
     -4.67546668e-03 -3.35129818e-02  1.73861214e-03  1.03513309e-02
     -1.45618250e-02  2.28916234e-02  1.39186835e-02 -7.69045362e-03
     -2.04908682e-02 -2.23503849e-02  6.37589891e-02  2.49374024e-02
      3.71049292e-02 -1.78657585e-02 -2.11773655e-02 -3.27408451e-02
      1.98533756e-02  4.98423421e-02  6.30561169e-03 -4.79096078e-02
      8.28283039e-03  2.71806836e-02 -3.62156439e-02 -3.25609974e-03
     -2.20497798e-02  5.81364593e-02 -2.43832865e-02 -5.22465335e-02
      5.76876549e-02  1.56378572e-02  2.21045139e-02  1.09369591e-02
      6.23802413e-03 -4.69469470e-02 -9.92504329e-03 -4.97588013e-03
      2.71751945e-02  4.52418123e-02 -5.26578308e-02 -3.84188226e-02
      2.81086137e-02  6.74811011e-03  1.44547389e-02  3.20061345e-02
      4.62248989e-03 -2.87615912e-02  3.86528422e-02  2.81599243e-02
     -9.83532552e-03  3.24094249e-02 -2.14294390e-02 -3.34926792e-02
      3.63684376e-02 -3.16930318e-02  4.15567668e-03  3.17027329e-02
     -2.70919562e-02  1.27824097e-02 -1.09439746e-02  1.83509871e-02
      3.14700747e-02 -1.17925539e-02 -1.27315609e-02 -1.12330224e-02
      2.20082187e-02  4.12549447e-02  1.52738247e-03  6.26139186e-03
      1.01897254e-02  1.19578495e-02  9.53953877e-03  2.31344014e-02
      2.60174737e-02  4.30115348e-03  1.29516260e-02 -1.37474336e-02
     -8.48257470e-02 -2.57754583e-02  5.80689802e-02  2.29887963e-02
      1.10035611e-02 -2.70864019e-02  2.30286207e-03 -2.08240972e-02
      2.84121994e-02 -6.22815442e-03  2.56126781e-02  9.39193470e-03
     -5.07546162e-02 -4.14534697e-02  6.68585130e-03 -3.92823508e-02
      3.63299165e-02 -2.10575536e-02  2.76483272e-03  2.19332935e-02
     -6.12826039e-04  2.67153454e-02  7.95683181e-03  9.12745057e-03
     -3.17034892e-02 -1.38753277e-02  3.42328933e-02 -3.09336326e-02
      9.77807405e-03  4.48699916e-02 -3.96728781e-02  4.17819371e-02
      2.17408758e-02  2.06703516e-02  2.43832236e-02 -7.15928396e-02
     -3.49534646e-02  2.70615730e-02 -1.02890640e-02  9.68479297e-05
      6.97849977e-02 -1.05781219e-02  1.09236910e-02 -1.28988297e-02
      4.51695864e-02 -7.60823138e-03  3.17043089e-02  1.29968632e-02
     -6.71885633e-02  5.94228973e-03 -2.50938024e-02  5.46325954e-02
      7.58085553e-03 -2.65070734e-02  3.72582400e-03 -2.93101844e-02
     -1.57744579e-02  9.15095000e-03  4.12674455e-02 -5.97308221e-03
      2.32708805e-04 -4.98232545e-02  2.32406022e-02 -1.57983798e-03
      1.35041616e-02  2.01560802e-02 -2.55096082e-02  2.90049154e-02
      3.66689572e-02 -3.21533775e-02 -1.90627175e-02 -1.62051600e-02
      2.03911381e-02  1.49303863e-02  1.31561611e-02 -2.92717583e-02
      2.74817355e-02  9.14752316e-05 -2.27916639e-02 -1.79915876e-02
      1.54362292e-02 -2.44011997e-02  5.20922667e-02 -4.49473066e-02
      1.02630081e-02 -4.23346087e-02  4.17690096e-02  1.98697505e-02
      2.45054979e-02  1.27461836e-02  2.51410189e-03 -8.47106129e-03
      7.18489979e-02 -6.00291953e-04  2.17787202e-02  3.41623564e-02
      2.70417108e-02  6.04962437e-02  1.82270668e-02  9.33889369e-03
      1.61359452e-03 -2.25409710e-02  1.23628449e-02 -2.83398116e-02
     -2.91109873e-02  4.27929752e-03  9.55980837e-03  6.05002388e-02
      6.52459643e-03 -4.81790840e-02 -3.97342657e-02  7.52780259e-03
      1.94217478e-02 -1.07197330e-02  5.02910421e-02 -3.78296575e-02
      3.00452905e-02 -8.15361636e-03  2.99332541e-02 -3.95511608e-02
     -2.41403638e-02  5.35981436e-03  1.21459687e-02 -1.57393692e-02
      5.83441766e-02 -3.03189059e-02  1.09404528e-02  4.45913375e-03
      1.83987450e-02 -5.31486259e-02  4.53857769e-02  3.05045599e-02
     -1.63257790e-02 -1.06150493e-02  1.95484779e-03 -6.45646293e-02
     -3.35211717e-02  1.47154758e-02 -1.85269292e-02  4.12212242e-03
      6.30026298e-03 -2.98177243e-02  3.80174481e-02  4.31952351e-02
      1.50583213e-02  3.63249564e-02 -2.04630572e-02  2.01770566e-02
     -4.53591452e-02 -9.59226152e-04 -2.84296587e-02  1.97757228e-03
     -1.70229335e-02 -1.47616375e-02  2.90506219e-02  8.27121261e-03
      3.87348008e-02  1.39504145e-02  4.81426937e-02 -2.09354670e-02
      1.50719020e-02 -2.83053855e-02  3.56674617e-02 -2.67730292e-02
      4.14134617e-02 -5.73448248e-02  1.49800562e-03  5.40031540e-02
      1.75953953e-02 -3.25975588e-02  1.29583274e-02 -9.88179373e-03
      1.55722518e-02  1.18684492e-04 -4.28330002e-02 -1.55015972e-02
     -8.36818351e-03 -1.48286113e-02  1.42919363e-02  5.56829988e-03
     -4.23758829e-02  2.89311133e-02 -2.61973164e-02  1.91979604e-02
     -6.48708518e-03  7.67005170e-03 -2.31689925e-02  1.85415097e-02
     -3.14109232e-03 -5.82964244e-02 -1.98294827e-02  3.36745219e-02
      3.88374615e-02  4.58440020e-02  2.29612888e-03 -2.60616532e-02
     -6.02214997e-02  3.67181573e-02  2.62912344e-02 -2.72493525e-02
      1.50999438e-02 -4.23091974e-03  2.25445914e-02 -2.32232094e-03
      7.21406569e-02  1.72184168e-02 -1.89437820e-02  2.94908768e-03
      4.06812697e-02 -5.35080984e-02  6.28059328e-02 -3.21819211e-03
     -8.42763063e-03  1.25797136e-02  5.37299523e-02  4.99943072e-02
      6.72750493e-02  1.70435558e-03 -3.44142156e-02  3.88877482e-02
      2.34114987e-02 -2.40964279e-02 -5.40351133e-02 -3.57703979e-04
     -7.53571373e-02 -5.08118839e-02 -1.85766539e-02  6.40262927e-02
      8.23233478e-03  3.73895822e-02 -4.30586212e-02  5.08266642e-03
     -1.84310377e-02  2.57964665e-02 -2.86748774e-02 -1.28027801e-02
      3.11500875e-02  1.14692724e-02 -2.17131370e-02 -4.86611855e-02
      2.93127630e-02  1.34913183e-02  5.51735252e-03  1.58171817e-02
      1.20389801e-02  1.43874523e-02 -4.66782847e-03 -1.07482802e-02
      2.11674519e-02 -3.20771721e-02  1.38083306e-02 -8.43181109e-03
      1.88147236e-02  3.10400202e-02 -1.12650488e-02 -4.01674022e-02
     -5.40218494e-03  3.33936088e-02  2.65105007e-02 -2.10146368e-02
     -3.11466796e-02 -1.77903730e-02 -3.31686965e-02 -1.23087376e-02
     -2.06879530e-02  2.66220265e-02 -3.34184838e-02 -3.26163781e-02
      3.31377856e-02 -3.32196920e-02  1.35587196e-02  1.47738068e-03
      3.58891666e-02  9.55448457e-02  5.22115956e-04 -4.35288788e-03
     -3.60445607e-02 -4.49492214e-03  1.52795944e-02 -8.44120000e-02
     -6.82845576e-04  3.95288633e-02 -1.26592873e-02  4.29842092e-03
      1.07652061e-02 -2.78527528e-02  2.50075813e-02 -5.12171670e-02
      5.42427689e-03  7.10517186e-03  1.63890285e-02  1.99824656e-02
     -3.21187346e-02  3.92286757e-02  8.99227209e-03 -1.60244911e-02
      1.59663989e-02  9.38989178e-03 -5.60161706e-03 -1.12558428e-03
      2.36637223e-02 -4.50375536e-02 -1.68662706e-03  5.90840007e-02
     -1.10328288e-02 -1.15189810e-02  4.82446419e-02  1.43005271e-02
      2.94514047e-03  2.15269032e-02  5.80088734e-02  3.57445867e-02
      2.08422973e-02 -3.05665718e-02 -8.56329401e-02 -4.59465134e-03
      4.22250597e-02 -3.32821623e-02  2.48433945e-02 -2.44495012e-03
     -7.09817078e-03  8.57293921e-03]
    

### 4.1.3. Humidity of Montreal - AutoRegressive Model

Stages:
1. Stationarity Testing for humidity of Montreal
2. If the series is not stationary -> phải lấy sai phân (differencing)
3. Define the selected lag: Use ACF/PACF or AIC/BIC to select the lag
4. Fit the model: Estimate the parameters of the model
5. Forecast


```python
humidity_montreal_adf = adfuller(humidity["Montreal"])
print(f"p-value of Humidity of Montreal: {float(humidity_montreal_adf[1]):.10f}")
```

    p-value of Humidity of Montreal: 0.0000000000
    


```python
# Predicting humidity level of Montreal
humid = AutoReg(humidity["Montreal"].diff().iloc[1:].values, lags = 1, old_names= False)
res = humid.fit()
res.plot_predict(start=1000, end=1100, figsize=(15,5))
plt.show()
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_235_0.png)
    



```python
rmse = math.sqrt(mean_squared_error(humidity["Montreal"].diff().iloc[900:1000].values, res.predict(start=900,end=999)))
print("The root mean squared error is {}.".format(rmse))
```

    The root mean squared error is 9.442575888890405.
    

- Explain step-by-step process of time series analysis:
    - *Step 1: Create training dataset – sai phân*

        `humidity["Montreal"].diff().iloc[1:].values`
        - Mindset:
            - Mô hình AR yêu cầu dữ liệu **stationary (chuỗi dừng)** — tức là **trung bình và phương sai không đổi theo thời gian.**
            - Dữ liệu độ ẩm gốc có thể **dao động theo mùa hoặc xu hướng tăng giảm dần**, nên bạn dùng **sai phân (differencing)** để loại bỏ xu hướng đó.

        - Result: → Thu được chuỗi biến động (thay đổi) của độ ẩm qua thời gian, chứ không phải độ ẩm tuyệt đối.    
        - Why: 
            - Nếu không sai phân, mô hình AR sẽ không hội tụ và kết quả bị lệch nặng (non-stationary bias).
            - ADF test (Dickey-Fuller) có thể xác nhận điều này trước đó.

    - *Step 2: Build AutoRegression Model*

        `humid = AutoReg(humidity["Montreal"].diff().iloc[1:].values, lags=1, old_names=False)`
        `res = humid.fit()`
        - Mechanism:
            - `AutoReg()` tạo mô hình **AR(p)** — ở đây **p = 1**, tức là mô hình **chỉ dùng 1 độ trễ (lag = 1)**.
            - Nghĩa là: $y_t = c + \phi_1 y_{t-1} + \epsilon_t$ <br/>
            → Độ ẩm hôm nay được dự đoán dựa vào **độ ẩm hôm qua** và một phần nhiễu nhỏ (`ε_t`).
        - Lý do chọn `lag=1`
            - Đây là bước **thử mô hình AR(1)** – mô hình đơn giản nhất, để kiểm tra xem dữ liệu có tự tương quan ngắn hạn không.
            - Sau đó, bạn có thể thử `lags=3` hoặc `lags=7` để xem mô hình có cải thiện không.

    - *Step 3: Forecast*

        `res.plot_predict(start=1000, end=1100, figsize=(15,5))`
        - Mechanism:
            - Mô hình dùng **quan hệ trong dữ liệu quá khứ** để **dự đoán độ ẩm từ thời điểm 1000 → 1100** (tức là tương lai).
            - Biểu đồ hiển thị **giá trị dự báo (Forecast)** để quan sát:
                - Dự báo có bám sát xu hướng thật không?
                - Dự báo có ổn định không?
        - Statistical Mindset
            - Đây là bước **kiểm định trực quan (visual validation)** cho mô hình AR.   
            - Nếu dự báo dao động cực mạnh, có thể do dữ liệu chưa đủ dừng hoặc lag chọn chưa phù hợp.

    - *Step 4: Forecasting Error Evaluation*

        ```
        rmse = math.sqrt(mean_squared_error(
            humidity["Montreal"].diff().iloc[900:1000].values,
            res.predict(start=900, end=999)
        ))
        print("The root mean squared error is {}".format(rmse))
        ```
        - Mechanism:
            - **RMSE (Root Mean Squared Error)** đo **độ chênh lệch trung bình giữa giá trị thật và giá trị dự báo**.
            - Giá trị càng nhỏ → mô hình càng chính xác.

        - Mindset:
            - `sqrt(MSE)` giúp đo lỗi theo **đơn vị gốc của dữ liệu (same unit as y)**, dễ hiểu trong thực tế.
            - Ở đây, `RMSE = 9.44` → trung bình mỗi dự báo sai khoảng **9.44 đơn vị độ ẩm** so với thực tế.

## 4.2. Moving Average Model

- $Y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}$

    | Ký hiệu                                               | Ý nghĩa                                                  |
    | ----------------------------------------------------- | -------------------------------------------------------- |
    | $Y_t$                                                 | Giá trị hiện tại của chuỗi                               |
    | $c$                                                   | Hằng số                                                  |
    | $\epsilon_t$                                          | Nhiễu (white noise) tại thời điểm hiện tại               |
    | $\epsilon_{t-1}, \epsilon_{t-2}, ..., \epsilon_{t-q}$ | Nhiễu trong quá khứ                                      |
    | $\theta_1, \theta_2, ..., \theta_q$                   | **Moving Average Coefficients** (hệ số trung bình trượt) |

- Moving Average Coefficients
    -  Là các hệ số $\theta_i$ cho biết **mức độ ảnh hưởng của sai số quá khứ (error)** đến giá trị hiện tại.
    - Khác với AR (hồi quy theo **giá trị quá khứ**), MA hồi quy theo **sai số quá khứ**.

- Moving Average Model Mechanics
    | Bước | Diễn giải                                                                                         |
    | ---- | ------------------------------------------------------------------------------------------------- |
    | 1    | Mỗi giá trị $Y_t$ được hình thành từ trung bình cộng của nhiễu (sai số) ở các thời điểm trước đó. |
    | 2    | Nếu có một sai số bất thường (outlier), hiệu ứng đó sẽ **giảm dần** qua các bước tiếp theo.       |
    | 3    | Mô hình MA hoạt động tốt với **chuỗi có nhiễu tạm thời**, không có xu hướng dài hạn.              |

- Autoregressive Model vs Moving Average Model

    | Đặc điểm       | Autoregressive (AR)                               | Moving Average (MA)                                     |
    | -------------- | ------------------------------------------------- | ------------------------------------------------------- |
    | Dựa vào        | Giá trị quá khứ                                   | Sai số quá khứ                                          |
    | Hệ số          | $\phi_i$                                          | $\theta_i$                                              |
    | Dạng công thức | $Y_t = c + \phi_1Y_{t-1} + ...$                   | $Y_t = c + \epsilon_t + \theta_1\epsilon_{t-1} + ...$   |
    | Khi dùng       | Khi dữ liệu có **xu hướng kéo dài** (persistence) | Khi dữ liệu có **shock ngắn hạn** hoặc “nhiễu” mạnh     |
    | Ví dụ thực tế  | Giá cổ phiếu có quán tính (tăng → tăng)           | Giá cổ phiếu điều chỉnh ngắn hạn sau tin tức bất thường |
    | Output         | Hệ số thể hiện “đà” (momentum)                    | Hệ số thể hiện “hiệu ứng sửa sai” (correction)          |

### 4.2.1. Moving Average Model Simulation


```python
rcParams['figure.figsize'] = 16, 6
ar1 = np.array([1])
ma1 = np.array([1, -0.5])
MA1 = ArmaProcess(ar1, ma1)
sim1 = MA1.generate_sample(nsample=1000)
plt.plot(sim1)
```




    [<matplotlib.lines.Line2D at 0x2415fe67750>]




    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_240_1.png)
    



```python
# MA(1) with constant mean (μ), i.e., ARIMA(0,0,1)
model = ARIMA(sim1, order=(0, 0, 1), trend='c')
result = model.fit()
print(result.summary())

# mu = result.params['const']         # intercept (mean)
mu = result.params[0]
# theta = result.params['ma.L1']      # MA(1) coefficient
theta = result.params[1]
print("μ={} ,θ={}".format(mu, theta))
```

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 1000
    Model:                 ARIMA(0, 0, 1)   Log Likelihood               -1443.024
    Date:                Fri, 17 Oct 2025   AIC                           2892.049
    Time:                        15:55:29   BIC                           2906.772
    Sample:                             0   HQIC                          2897.644
                                   - 1000                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -0.0135      0.016     -0.819      0.413      -0.046       0.019
    ma.L1         -0.4947      0.029    -17.354      0.000      -0.551      -0.439
    sigma2         1.0490      0.049     21.327      0.000       0.953       1.145
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.02   Jarque-Bera (JB):                 3.43
    Prob(Q):                              0.90   Prob(JB):                         0.18
    Heteroskedasticity (H):               1.19   Skew:                             0.12
    Prob(H) (two-sided):                  0.12   Kurtosis:                         2.84
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    μ=-0.013465698527286334 ,θ=-0.4946840103434182
    

- Code Explaination:

    | Thành phần | Mô tả | Ý nghĩa |
    | --- | --- | --- |
    | `model = ARIMA(humidity["Montreal"].diff().iloc[1:].values, order=(0,0,3), trend='c')` | Tạo mô hình ARIMA với 3 tham số `(p,d,q)` | `p=0`: không dùng phần tự hồi quy (AR) <br> `d=0`: không lấy sai phân (vì dữ liệu đã được `.diff()` thủ công) <br> `q=3`: dùng 3 độ trễ của nhiễu (MA(3)) |
    | `.fit()` | Ước lượng các hệ số MA (ma.L1, ma.L2, ma.L3) và hằng số | Tìm hệ số sao cho mô hình mô phỏng tốt nhất chuỗi sai phân |
    | `.summary()` | Xuất kết quả thống kê của mô hình | Hiển thị giá trị ước lượng, kiểm định p-value, AIC/BIC,... |
    | `print("μ={} ,θ={}".format(...))` | In ra giá trị trung bình và hệ số MA đầu tiên | Thể hiện bản chất mô hình MA dùng nhiễu trễ để dự đoán |

    - Moving Average Model:

        | Đặc điểm mô hình  | Biểu hiện trong code | Giải thích |
        | --- | --- | --- |
        | **Không có phần tự hồi quy (AR)** | `p = 0` | Mô hình không dựa vào giá trị quá khứ của chuỗi (y_{t-1}, y_{t-2}, ...) |
        | **Không có phần tích phân (I)** | `d = 0` và dữ liệu đã `.diff()` | Tức là đã loại bỏ xu hướng bằng tay, nên không cần “Integrated” trong mô hình |
        | **Chỉ có phần trung bình trượt (MA)** | `q = 3` | Mô hình dựa vào 3 sai số dự đoán trong quá khứ: ϵₜ₋₁, ϵₜ₋₂, ϵₜ₋₃ để dự đoán giá trị hiện tại |
        | **Phương trình cơ bản:** | | yₜ = μ + θ₁ϵₜ₋₁ + θ₂ϵₜ₋₂ + θ₃ϵₜ₋₃ + ϵₜ  |

        → Do đó, mô hình chỉ học cách các sai số trước đây ảnh hưởng đến sai số hiện tại, không học trực tiếp mối quan hệ giữa các giá trị độ ẩm quá khứ.

    - Mindset:

        | Mục đích tư duy | Mô tả chi tiết | Lý do thực hiện |
        | --- | --- | --- |
        | **a. Tách nhiễu (noise-driven model)** | MA model giả định rằng sai số dự báo (residuals) mang thông tin có tính lặp lại và có thể dự đoán được. | Dữ liệu độ ẩm (humidity) có thể chịu ảnh hưởng bởi các yếu tố ngẫu nhiên như thời tiết cục bộ, gió, hoặc đo sai. Những nhiễu này có cấu trúc và có thể được dự đoán ở mức độ ngắn hạn. |
        | **b. Kiểm tra xem chuỗi có cấu trúc ngẫu nhiên hay không** | Nếu mô hình MA cho kết quả p-value nhỏ → các hệ số θ có ý nghĩa thống kê → chứng minh rằng nhiễu không hoàn toàn ngẫu nhiên. | Đây là bước kiểm tra bản chất stochastic của dữ liệu sau khi đã loại bỏ xu hướng (bằng `.diff()`). |
        | **c. So sánh với AR Model** | AR dựa trên giá trị quá khứ của chính chuỗi; MA dựa trên sai số quá khứ. | Mục tiêu của bạn có thể là kiểm tra xem mô hình nào (AR hay MA) phù hợp hơn với dữ liệu độ ẩm của Montreal. |
        | **d. Là bước trung gian để tiến tới ARMA hoặc ARIMA** | Khi hiểu được hành vi của phần “nhiễu”, ta có thể kết hợp AR và MA sau này. | MA giúp phát hiện pattern trong sai số; AR giúp phát hiện pattern trong chính dữ liệu. Cả hai kết hợp → ARMA. |



- Model Comparison:

    | Mô hình | Thành phần chính | Khi nào sử dụng | Ví dụ |
    | --- | --- | --- | --- |
    | **AR(p)** | Giá trị trong quá khứ của chính biến | Khi biến có tính tự tương quan mạnh | Dự báo giá cổ phiếu dựa trên chính giá trước đó |
    | **MA(q)** | Sai số dự báo trong quá khứ | Khi nhiễu có cấu trúc, không phải ngẫu nhiên | Dự báo độ ẩm hoặc nhiệt độ có tính ngẫu nhiên lặp lại |
    | **ARMA(p,q)** | Kết hợp AR và MA | Khi dữ liệu tĩnh (stationary) nhưng có cả pattern và noise | Chuỗi nhiệt độ trung bình hàng ngày |
    | **ARIMA(p,d,q)** | ARMA + sai phân để loại xu hướng | Khi dữ liệu có xu hướng hoặc mùa vụ nhẹ | Dự báo GDP hoặc nhu cầu sản phẩm theo quý |

- `ARIMA` is a statistical model for time series forecasting.
    - It combines autoregressive, differencing, and moving average components to model non-stationary data.
    - ARIMA refers to the Autoregressive Integrated Moving Average model, a foundational method in time series analysis used to understand and predict future points in a series. It is implemented via the ARIMA class from the statsmodels.tsa.arima_model module in Python.
    - The model is specified by three parameters: p, d, and q, forming an ARIMA(p,d,q) model:
        - `p` (autoregressive order): Number of lagged observations included in the model.
        - `d` (degree of differencing): Number of times the data have been differenced to achieve stationarity.
        - `q` (moving average order): Size of the moving average window.
    - Params:
        - `endog`: Time series data (1D array-like)
        - `order`: Tuple (p, d, q) defining model structure
        - `trend`: Optional trend component ('c' for constant, 't' for linear trend)
    - Side effects: Fits model parameters via optimization; produces residuals, AIC, BIC.
    - Returns: A results object containing fitted parameters, diagnostics, and forecasting methods.

### 4.2.2. Humidity of Montreal - Moving Average Model


```python
humidity["Montreal"].iloc[1000]
```




    np.float64(100.0)




```python
predicted.predicted_mean
```




    array([ 3.12319810e-01, -5.94954851e-03, -5.23523980e-02, -1.97800032e-02,
           -1.72362060e-03,  1.51302358e-03,  2.06467371e-04, -7.53736434e-04,
           -9.69124595e-04, -9.21275256e-04, -8.72017363e-04, -8.58514811e-04,
            6.67946847e+00, -5.42667706e-01, -1.75945490e+00, -4.94727758e-01,
            7.15785589e-03,  1.87261489e+00, -1.16394777e-01, -4.71323146e-01,
           -1.37110053e-01, -3.97514392e-04,  2.07486354e-02,  7.30769133e-03,
           -6.35486570e+00,  5.13521798e-01,  1.67148423e+00,  1.44646859e+00,
           -8.77373166e-02, -2.29064496e+00,  1.19761377e+00,  4.21224440e-01,
           -1.41119774e-01, -8.15595513e-02, -2.28952982e-02,  3.56115723e-03,
            3.49862561e-03,  4.26586985e-04, -9.77883676e-04, -1.08242649e-03,
           -9.38364859e-04, -8.62617258e-04, -8.52421886e-04,  2.93196839e+00,
           -2.38728493e-01, -7.72928147e-01, -2.17681841e-01,  2.65834180e-03,
            3.47834575e-02,  1.22333848e-02, -2.67177885e-04, -2.50793406e-03,
           -1.60339755e-03, -9.38641702e-04, -2.93361662e+00,  2.37043107e-01,
            7.71207904e-01,  2.15952503e-01, -4.38650339e-03, -3.65098714e-02,
            6.82930397e+00, -5.56478813e-01, -1.80070224e+00, -5.06032683e-01,
            7.42916574e-03, -1.05830533e+00,  1.22158205e-01,  3.00768368e-01,
            7.96202713e-02,  3.09180405e+00, -5.47989692e+00, -3.97869396e-01,
            4.90955279e-01,  1.90823614e+00, -7.31252873e-01,  3.70702906e-03,
            5.52949379e-02, -3.53922789e-02, -8.87857468e-03, -2.72005025e-03,
            3.36137690e-04, -1.95553722e+00,  9.72492922e-01,  1.91414083e+00,
            4.61996577e-01, -5.02336047e-01, -3.03626680e-01, -6.97851133e-01,
           -1.39257066e+00,  3.05371497e-01,  4.36596051e-01,  1.05918452e-01,
           -1.13508452e-02, -2.18053590e-02, -7.51661563e-03, -7.55063095e-04,
           -2.76975561e+00,  2.24174540e-01,  1.86887258e+00,  1.11360342e-01,
            1.48782554e+00])




```python
# Forecasting and predicting montreal humidity
model = ARIMA(humidity["Montreal"].diff().iloc[1:].values, order=(0,0,3), trend = 'c')
result = model.fit()
print(result.summary())
print("μ={} ,θ={}".format(result.params[0],result.params[1]))
predicted = result.get_prediction(start=1000, end=1100)
forecast_original_scale = humidity["Montreal"].diff().iloc[1000]+ np.cumsum(predicted.predicted_mean)

plt.plot(humidity["Montreal"].diff().iloc[1000:1101].values, label='Actual Humidity', color=palette[1])
plt.plot(forecast_original_scale, label="Predicted Humidity", color = palette[2])
plt.legend()
plt.show()
```

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                44574
    Model:                 ARIMA(0, 0, 3)   Log Likelihood             -151553.136
    Date:                Fri, 17 Oct 2025   AIC                         303116.271
    Time:                        16:08:58   BIC                         303159.796
    Sample:                             0   HQIC                        303129.978
                                  - 44574                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -0.0008      0.031     -0.025      0.980      -0.062       0.061
    ma.L1         -0.1629      0.003    -55.826      0.000      -0.169      -0.157
    ma.L2          0.0398      0.003     13.032      0.000       0.034       0.046
    ma.L3          0.0343      0.003     10.415      0.000       0.028       0.041
    sigma2        52.5802      0.172    305.886      0.000      52.243      52.917
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.02   Jarque-Bera (JB):            117068.52
    Prob(Q):                              0.89   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.77   Skew:                            -0.00
    Prob(H) (two-sided):                  0.00   Kurtosis:                        10.94
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    μ=-0.0007862678580034305 ,θ=-0.16293483865962324
    


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_248_1.png)
    



```python
fig, ax = plt.subplots(figsize=(10, 8))
fig = plot_predict(result, start=1000, end=1100, ax=ax)
plt.show()
# legend = ax.legend(loc="upper left")

```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_249_0.png)
    


- `pandas.DataFrame.iloc[]` is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array.
    - Allowed inputs are:
        - An integer, e.g. 5.
        - A list or array of integers, e.g. [4, 3, 0].
        - A slice object with ints, e.g. 1:7.
        - A boolean array.
        - A callable function with one argument (the calling Series or DataFrame) and that returns valid output for indexing (one of the above). This is useful in method chains, when you don’t have a reference to the calling object, but would like to base your selection on some value.
        - A tuple of row and column indexes. The tuple elements consist of one of the above inputs, e.g. (0, 1).


```python
rmse = math.sqrt(mean_squared_error(humidity["Montreal"].diff().iloc[1000:1101].values, result.predict(start=1000,end=1100)))
print("The root mean squared error is {}.".format(rmse))
```

    The root mean squared error is 10.599810655093645.
    

## 4.3. Autoregressive Moving Average Model

Autoregressive–moving-average (ARMA) models provide a parsimonious description of a (weakly) stationary stochastic process in terms of two polynomials, one for the autoregression and the second for the moving average. It's the fusion of AR and MA models.

- ARMA(1,1) model

    $R_t = μ + \theta R_{t-1} + \epsilon_t + \phi\epsilon_{t-1}$ <br/>
    Basically, Today's return = mean + Yesterday's return + noise + yesterday's noise.

**(A) Autoregressive (AR)**

| Đặc điểm               | Mô tả                                                            |
| ---------------------- | ---------------------------------------------------------------- |
| **Công thức**          | $y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \epsilon_t$ |
| **Cơ chế**             | Dự đoán giá trị hiện tại dựa vào *các giá trị trong quá khứ*     |
| **Giả định chính**     | Quan hệ tuyến tính, dữ liệu **stationary**                       |
| **Ưu điểm**            | Đơn giản, dễ diễn giải, nhanh                                    |
| **Nhược điểm**         | Không mô tả được nhiễu hoặc phi tuyến tính                       |
| **Ứng dụng điển hình** | Modeling stock returns, demand time series without noise shock   |


**(B) Moving Average (MA)**

| Đặc điểm               | Mô tả                                                      |
| ---------------------- | ---------------------------------------------------------- |
| **Công thức**          | $y_t = c + \theta_1 \epsilon_{t-1} + ... + \epsilon_t$   |
| **Cơ chế**             | Dựa vào các *nhiễu trong quá khứ* (past errors) để dự đoán |
| **Giả định**           | Nhiễu có ảnh hưởng ngắn hạn và tuyến tính                  |
| **Ưu điểm**            | Xử lý tốt các biến động ngắn hạn                           |
| **Nhược điểm**         | Không nắm bắt được xu hướng dài hạn                        |
| **Ứng dụng điển hình** | Correction noise in measurement data                       |


**(C) ARMA (Autoregressive Moving Average)**

| Đặc điểm               | Mô tả                                                                                           |
| ---------------------- | ----------------------------------------------------------------------------------------------- |
| **Công thức**          | $y_t = c + \sum_{i=1}^{p}\phi_i y_{t-i} + \sum_{j=1}^{q}\theta_j \epsilon_{t-j} + \epsilon_t$ |
| **Cơ chế**             | Kết hợp *tác động giá trị quá khứ* và *nhiễu quá khứ*                                           |
| **Giả định**           | Dữ liệu phải **stationary**                                                                     |
| **Ưu điểm**            | Linh hoạt hơn AR hay MA đơn lẻ                                                                  |
| **Nhược điểm**         | Không xử lý được xu hướng hoặc seasonality                                                      |
| **Ứng dụng điển hình** | Dự báo biến động kinh tế ngắn hạn, tín hiệu kỹ thuật số ổn định                                 |



```python
# Forecasting and predicting microsoft stocks volume
model = ARIMA(microsoft_stock["Volume"].diff().iloc[1:].values, order=(3,0,3))
result = model.fit()
print(result.summary())
print("μ={}, ϕ={}, θ={}".format(result.params[0],result.params[1],result.params[2]))
predicted = result.get_prediction(start=1000, end=1100)
forecast_original_scale = microsoft_stock["Volume"].diff().iloc[1000]+ np.cumsum(predicted.predicted_mean)

plt.plot(microsoft_stock["Volume"].diff().iloc[1000:1101].values, label='Actual Humidity', color=palette[1])
plt.plot(forecast_original_scale, label="Predicted Humidity", color = palette[2])
plt.legend()
plt.show()
```

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 3018
    Model:                 ARIMA(3, 0, 3)   Log Likelihood              -55411.375
    Date:                Fri, 17 Oct 2025   AIC                         110838.751
    Time:                        16:10:15   BIC                         110886.849
    Sample:                             0   HQIC                        110856.046
                                   - 3018                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       -2.03e+04   1.12e+04     -1.814      0.070   -4.22e+04    1634.809
    ar.L1          0.0936      0.687      0.136      0.892      -1.253       1.440
    ar.L2          0.8902      0.744      1.197      0.231      -0.567       2.348
    ar.L3         -0.1882      0.132     -1.422      0.155      -0.448       0.071
    ma.L1         -0.7019      0.688     -1.021      0.307      -2.050       0.646
    ma.L2         -0.9866      1.163     -0.848      0.396      -3.266       1.293
    ma.L3          0.6927      0.482      1.437      0.151      -0.252       1.637
    sigma2      5.424e+14   2.45e-05   2.22e+19      0.000    5.42e+14    5.42e+14
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):           1467189.37
    Prob(Q):                              0.99   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.21   Skew:                             6.59
    Prob(H) (two-sided):                  0.00   Kurtosis:                       110.21
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    [2] Covariance matrix is singular or near-singular, with condition number 7.73e+32. Standard errors may be unstable.
    μ=-20297.220640330408, ϕ=0.09358020304259693, θ=0.8902405158593464
    


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_255_1.png)
    



```python
fig, ax = plt.subplots(figsize=(10, 8))
fig = plot_predict(result, start=1000, end=1100, ax=ax)
plt.show()
# legend = ax.legend(loc="upper left")

```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_256_0.png)
    



```python
rmse = math.sqrt(mean_squared_error(microsoft_stock["Volume"].diff().iloc[1000:1101].values, result.predict(start=1000,end=1100)))
print("The root mean squared error is {}.".format(rmse))
```

    The root mean squared error is 38022545.84186862.
    

## 4.4. Autoregressive Integrated Moving Average Model

- An autoregressive integrated moving average (ARIMA) model is a generalization of an autoregressive moving average (ARMA) model. Both of these models are fitted to time series data either to better understand the data or to predict future points in the series (forecasting). ARIMA models are applied in some cases where data show evidence of non-stationarity, where an initial differencing step (corresponding to the "integrated" part of the model) can be applied one or more times to eliminate the non-stationarity. ARIMA model is of the form: ARIMA(p,d,q): p is AR parameter, d is differential parameter, q is MA parameter

- ARIMA(1,0,0) <br/>
    $yt = a_1 y_{t-1} + \epsilon_t$

- ARIMA(1,0,1) <br/>
    $yt = a_1 y_{t-1} + \epsilon_t + b_1 \epsilon_{t-1}$

- ARIMA(1,1,1) <br/>
    $\delta y_t = a_1 \delta y_{t-1} + \epsilon_t + b_1 \epsilon_{t-1}$  where $\delta y_t = y_t - y_{t-1}$


```python
# Predicting the microsoft stocks volume
rcParams['figure.figsize'] = 16, 6
model = ARIMA(microsoft_stock["Volume"].diff().iloc[1:].values, order=(2,1,0))
result = model.fit()
print(result.summary())
predicted = result.get_prediction(start=700, end=1000)
forecast_original_scale = microsoft_stock["Volume"].diff().iloc[700] + np.cumsum(predicted.predicted_mean)

plt.plot(microsoft_stock["Volume"].diff().iloc[700:1000].values, label = 'Original', color = palette[1])
plt.plot(forecast_original_scale, label = 'Forecast', color = palette[2])
plt.legend()
plt.show()
```

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 3018
    Model:                 ARIMA(2, 1, 0)   Log Likelihood              -56385.152
    Date:                Fri, 17 Oct 2025   AIC                         112776.303
    Time:                        16:20:09   BIC                         112794.339
    Sample:                             0   HQIC                        112782.789
                                   - 3018                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1         -0.8715      0.003   -258.467      0.000      -0.878      -0.865
    ar.L2         -0.4549      0.007    -61.690      0.000      -0.469      -0.440
    sigma2      1.001e+15        nan        nan        nan         nan         nan
    ===================================================================================
    Ljung-Box (L1) (Q):                  73.46   Jarque-Bera (JB):            287181.39
    Prob(Q):                              0.00   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.22   Skew:                            -0.58
    Prob(H) (two-sided):                  0.00   Kurtosis:                        50.78
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    [2] Covariance matrix is singular or near-singular, with condition number 8.68e+45. Standard errors may be unstable.
    


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_259_1.png)
    


- `cumsum` is a NumPy function that computes the cumulative sum of array elements along a specified axis.
    - It returns an array of partial sums, where each element is the sum of all previous values up to that position.
    - Params:
        - `a`: Input array or array-like object.
        - `axis`: Optional axis for summation; None flattens the array.
        - `dtype`: Optional output data type; influences precision and overflow behavior.
        - `out`: Optional pre-allocated array for result storage.
    - Side effects: None; function is pure unless out is provided (in-place write).
    - Returns: ndarray containing cumulative sums; shape matches input unless flattened.



```python
rmse = math.sqrt(mean_squared_error(microsoft_stock["Volume"].diff().iloc[700:1001].values, result.predict(start=700,end=1000)))
print("The root mean squared error is {}.".format(rmse))
```

    The root mean squared error is 33814021.136790834.
    

## 4.5. Vector Autoregression Model

- Vector AutoRegression Model (VAR) is a stochastic process model used to capture the linear interdependencies among multiple time series. VAR models generalize the univariate autoregressive model (AR model) by allowing for more than one evolving variable. All variables in a VAR enter the model in the same way: each variable has an equation explaining its evolution based on its own lagged values, the lagged values of the other model variables, and an error term. VAR modeling does not require as much knowledge about the forces influencing a variable as do structural models with simultaneous equations: The only prior knowledge required is a list of variables which can be hypothesized to affect each other intertemporally.


```python
# Predicting closing price of Google and microsoft
train_sample = pd.concat([google["Close"].diff().iloc[1:],microsoft["Close"].diff().iloc[1:]],axis=1)
model = sm.tsa.VARMAX(train_sample,order=(2,1),trend='c')
result = model.fit(maxiter=1000,disp=False)
print(result.summary())
predicted_result = result.predict(start=0, end=1000)
result.plot_diagnostics()
# calculating error
rmse = math.sqrt(mean_squared_error(train_sample.iloc[1:1002].values, predicted_result.values))
print("The root mean squared error is {}.".format(rmse))
```

    C:\Users\yenpth8\AppData\Roaming\Python\Python313\site-packages\statsmodels\tsa\statespace\varmax.py:160: EstimationWarning:
    
    Estimation of VARMA(p,q) models is not generically robust, due especially to identification issues.
    
    C:\Users\yenpth8\AppData\Roaming\Python\Python313\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning:
    
    A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
    
    

                               Statespace Model Results                           
    ==============================================================================
    Dep. Variable:     ['Close', 'Close']   No. Observations:                 3018
    Model:                     VARMA(2,1)   Log Likelihood              -12185.090
                              + intercept   AIC                          24404.179
    Date:                Fri, 17 Oct 2025   BIC                          24506.389
    Time:                        16:29:07   HQIC                         24440.933
    Sample:                             0                                         
                                   - 3018                                         
    Covariance Type:                  opg                                         
    ===================================================================================
    Ljung-Box (L1) (Q):             0.00, 0.01   Jarque-Bera (JB):   48240.08, 14923.91
    Prob(Q):                        0.96, 0.94   Prob(JB):                   0.00, 0.00
    Heteroskedasticity (H):         3.32, 1.62   Skew:                      1.15, -0.03
    Prob(H) (two-sided):            0.00, 0.00   Kurtosis:                 22.45, 13.89
                               Results for equation Close                          
    ===============================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------
    intercept       0.3554      0.279      1.274      0.203      -0.191       0.902
    L1.Close       -0.1964      0.534     -0.368      0.713      -1.242       0.849
    L1.Close       -0.1784      4.963     -0.036      0.971      -9.905       9.548
    L2.Close        0.0049      0.036      0.139      0.889      -0.065       0.075
    L2.Close        0.3781      0.436      0.868      0.385      -0.476       1.232
    L1.e(Close)     0.2358      0.534      0.442      0.659      -0.811       1.283
    L1.e(Close)    -0.0856      4.985     -0.017      0.986      -9.856       9.685
                               Results for equation Close                          
    ===============================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------
    intercept       0.0179      0.029      0.626      0.531      -0.038       0.074
    L1.Close        0.0407      0.055      0.744      0.457      -0.067       0.148
    L1.Close       -0.4482      0.530     -0.846      0.397      -1.486       0.590
    L2.Close        0.0014      0.004      0.384      0.701      -0.006       0.008
    L2.Close       -0.0433      0.043     -1.017      0.309      -0.127       0.040
    L1.e(Close)    -0.0385      0.055     -0.703      0.482      -0.146       0.069
    L1.e(Close)     0.4013      0.530      0.757      0.449      -0.638       1.441
                                    Error covariance matrix                                 
    ========================================================================================
                               coef    std err          z      P>|z|      [0.025      0.975]
    ----------------------------------------------------------------------------------------
    sqrt.var.Close           6.8950      0.041    167.415      0.000       6.814       6.976
    sqrt.cov.Close.Close     0.2923      0.005     57.677      0.000       0.282       0.302
    sqrt.var.Close           0.4807      0.003    163.189      0.000       0.475       0.486
    ========================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    The root mean squared error is 3.6751131473631515.
    


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_264_2.png)
    


## 4.6. [State Space Models](https://www.statsmodels.org/dev/statespace.html)

- Definition:
    | Khái niệm | Giải thích dễ hiểu | Business Mindset |
    | --- | --- | --- |
    | **State Space Method** | Là **khung toán học tổng quát** mô tả hệ thống động (dynamic system) bằng **2 phương trình**: (1) *phương trình trạng thái ẩn (state equation)* và (2) *phương trình quan sát (observation equation)*. | |
    | **State (trạng thái)** | Là biến ẩn (unobserved) đại diện cho “tình hình nội tại” của hệ thống qua thời gian. | Giống như bạn chỉ nhìn thấy doanh số, nhưng không thấy được “mức độ nhu cầu tiềm ẩn”, “xu hướng mùa vụ” hay “cú sốc thị trường” — những thứ đó chính là state. State space model giúp tái dựng các yếu tố ẩn này dựa trên dữ liệu quan sát được. |
    | **Observation (quan sát)** | Là dữ liệu thực tế bạn nhìn thấy (doanh thu, độ ẩm, giá cổ phiếu, v.v.). | |

- Mathematical Formulation:
    - **State equation (ẩn):** <br/>
        $s_t = F s_{t-1} + v_t$ <br/>
        → mô tả cách trạng thái thay đổi theo thời gian (có thể chịu ảnh hưởng bởi shock ngẫu nhiên (v_t)).

    - **Observation equation (quan sát):** <br/>
        $y_t = H s_t + w_t$ <br/>
        → mô tả cách trạng thái ẩn sinh ra dữ liệu quan sát.

- Models:
    | Mô hình | Liên hệ với State Space |
    | --- | --- |
    | **ARIMA** | Là **trường hợp đặc biệt** của state-space model. ARIMA có thể được viết lại hoàn toàn dưới dạng hai phương trình state–observation. Khi viết như vậy, ta có thể dùng **Kalman Filter** để ước lượng. |
    | **Unobserved Components Model (UCM)** | Chính là **phiên bản mở rộng trực tiếp** của state space, trong đó mỗi thành phần (trend, seasonal, cyclical, irregular) được coi là một **state** riêng. |
    | **Dynamic Factor Model (DFM)** | Cũng thuộc họ **state space**, nhưng ở đây có **nhiều chuỗi quan sát cùng lúc** (đa biến). Các chuỗi này chia sẻ **một vài yếu tố ẩn (common factors)** — chính là state. |
    | **Kalman Filter / Smoother** | Là **công cụ tính toán cốt lõi** dùng trong mọi mô hình state space, giúp ước lượng trạng thái ẩn qua thời gian. |


### 4.6.1. Seasonal AutoRegressive Integrated Moving Average (SARIMA) Model 
- SARIMA models are useful for modeling seasonal time series, in which the mean and other statistics for a given season are not stationary across the years. The SARIMA model defined constitutes a straightforward extension of the nonseasonal autoregressive-moving average (ARMA) and autoregressive integrated moving average (ARIMA) models presented    
- SARIMA Model = ARIMA + yếu tố mùa vụ (Seasonality)
    - Nó phù hợp cho các chuỗi thời gian có cả trend và seasonality (như doanh thu, lượng khách hàng, nhiệt độ,…).


```python
# Predicting closing price of Google'
train_sample = google_stock["Close"].diff().iloc[1:].values
model = sm.tsa.SARIMAX(train_sample,order=(4,0,4),trend='c')
result = model.fit(maxiter=1000,disp=False)
print(result.summary())
predicted_result = result.predict(start=0, end=500)
result.plot_diagnostics()
# calculating error
rmse = math.sqrt(mean_squared_error(train_sample[1:502], predicted_result))
print("The root mean squared error is {}.".format(rmse))
```

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 3018
    Model:               SARIMAX(4, 0, 4)   Log Likelihood              -10098.031
    Date:                Fri, 17 Oct 2025   AIC                          20216.061
    Time:                        17:23:41   BIC                          20276.185
    Sample:                             0   HQIC                         20237.681
                                   - 3018                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    intercept      0.1072      0.048      2.245      0.025       0.014       0.201
    ar.L1          0.2272      0.007     32.899      0.000       0.214       0.241
    ar.L2          1.1200      0.006    196.705      0.000       1.109       1.131
    ar.L3          0.2521      0.006     44.975      0.000       0.241       0.263
    ar.L4         -0.9814      0.007   -135.959      0.000      -0.996      -0.967
    ma.L1         -0.2295      0.008    -27.721      0.000      -0.246      -0.213
    ma.L2         -1.1215      0.007   -154.683      0.000      -1.136      -1.107
    ma.L3         -0.2609      0.007    -35.274      0.000      -0.275      -0.246
    ma.L4          0.9745      0.008    117.703      0.000       0.958       0.991
    sigma2        46.9035      0.401    116.880      0.000      46.117      47.690
    ===================================================================================
    Ljung-Box (L1) (Q):                   1.70   Jarque-Bera (JB):             53947.17
    Prob(Q):                              0.19   Prob(JB):                         0.00
    Heteroskedasticity (H):               3.27   Skew:                             1.22
    Prob(H) (two-sided):                  0.00   Kurtosis:                        23.57
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    The root mean squared error is 4.380316723172399.
    


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_267_1.png)
    


### 4.6.2. Unobserved components Model

- A UCM decomposes the response series into components such as trend, seasons, cycles, and the regression effects due to predictor series. The following model shows a possible scenario:
    $$y_t = \mu_t + \gamma_t \psi_t + \sum_{j=1}^m \beta_j X_{jt} + \epsilon_t$$
    
    $$\epsilon_t ~ i.i.d .N(0, \sigma_{\epsilon}^2)$$

    Where 
    - $\mu_t$ is the trend component
    - $\gamma_t$ is the seasonal component
    - $\psi_t$ is the cyclical component
    - $\sum_{j=1}^m \beta_j X_{jt}$ includes contribution of regression variables with fixed regression coefficients

- [Reference from Statsmodel](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_cycles.html#:~:text=%3D129600-,Unobserved%20components,-and%20ARIMA%20model)


```python
# Predicting closing price of Google'
train_sample = google_stock["Close"].diff().iloc[1:].values
model = sm.tsa.UnobservedComponents(train_sample,'local level')
result = model.fit(maxiter=1000,disp=False)
print(result.summary())
predicted_result = result.predict(start=0, end=500)
result.plot_diagnostics()
# calculating error
rmse = math.sqrt(mean_squared_error(train_sample[1:502], predicted_result))
print("The root mean squared error is {}.".format(rmse))
```

                            Unobserved Components Results                         
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 3018
    Model:                    local level   Log Likelihood              -10116.511
    Date:                Fri, 17 Oct 2025   AIC                          20237.023
    Time:                        17:39:36   BIC                          20249.047
    Sample:                             0   HQIC                         20241.346
                                   - 3018                                         
    Covariance Type:                  opg                                         
    ====================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    sigma2.irregular    47.7219      0.384    124.248      0.000      46.969      48.475
    sigma2.level      5.033e-05      0.000      0.458      0.647      -0.000       0.000
    ===================================================================================
    Ljung-Box (L1) (Q):                   2.22   Jarque-Bera (JB):             48879.50
    Prob(Q):                              0.14   Prob(JB):                         0.00
    Heteroskedasticity (H):               3.36   Skew:                             1.11
    Prob(H) (two-sided):                  0.00   Kurtosis:                        22.59
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    The root mean squared error is 4.412418985322813.
    


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_270_1.png)
    



```python
plt.plot(train_sample[1:502],color= palette[1])
plt.plot(predicted_result,color= palette[0])
plt.legend(['Actual','Predicted'])
plt.title('Google Closing prices')
plt.show()
```


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_271_0.png)
    


### 4.6.3. Dynamic Factor Model


- Dynamic-factor models are flexible models for multivariate time series in which the observed endogenous variables are linear functions of exogenous covariates and unobserved factors, which have a vector autoregressive structure. The unobserved factors may also be a function of exogenous covariates. The disturbances in the equations for the dependent variables may be autocorrelated.




```python
# Predicting closing price of Google and microsoft
train_sample = pd.concat([google_stock["Close"].diff().iloc[1:],microsoft_stock["Close"].diff().iloc[1:]],axis=1)
model = sm.tsa.DynamicFactor(train_sample, k_factors=1, factor_order=2)
result = model.fit(maxiter=1000,disp=False)
print(result.summary())
predicted_result = result.predict(start=0, end=1000)
result.plot_diagnostics()
# calculating error
rmse = math.sqrt(mean_squared_error(train_sample.iloc[1:1002].values, predicted_result.values))
print("The root mean squared error is {}.".format(rmse))
```

    C:\Users\yenpth8\AppData\Roaming\Python\Python313\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning:
    
    A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
    
    

                                       Statespace Model Results                                  
    =============================================================================================
    Dep. Variable:                    ['Close', 'Close']   No. Observations:                 3018
    Model:             DynamicFactor(factors=1, order=2)   Log Likelihood              -12197.974
    Date:                               Fri, 17 Oct 2025   AIC                          24407.948
    Time:                                       17:41:58   BIC                          24444.022
    Sample:                                            0   HQIC                         24420.919
                                                  - 3018                                         
    Covariance Type:                                 opg                                         
    ===================================================================================
    Ljung-Box (L1) (Q):             4.01, 0.57   Jarque-Bera (JB):   49450.41, 15108.67
    Prob(Q):                        0.05, 0.45   Prob(JB):                   0.00, 0.00
    Heteroskedasticity (H):         3.38, 1.63   Skew:                      1.14, -0.03
    Prob(H) (two-sided):            0.00, 0.00   Kurtosis:                 22.70, 13.96
                              Results for equation Close                          
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    loading.f1    -3.5813      0.813     -4.407      0.000      -5.174      -1.989
                              Results for equation Close                          
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    loading.f1    -0.5636      0.129     -4.386      0.000      -0.815      -0.312
                            Results for factor equation f1                        
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    L1.f1         -0.0305      0.019     -1.629      0.103      -0.067       0.006
    L2.f1         -0.0240      0.018     -1.314      0.189      -0.060       0.012
                                Error covariance matrix                             
    ================================================================================
                       coef    std err          z      P>|z|      [0.025      0.975]
    --------------------------------------------------------------------------------
    sigma2.Close    34.9678      5.794      6.035      0.000      23.612      46.323
    sigma2.Close   2.12e-10      0.145   1.46e-09      1.000      -0.284       0.284
    ================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    The root mean squared error is 3.682329681843541.
    


    
![png](251004_Time_Series_Analysis_files/251004_Time_Series_Analysis_274_2.png)
    


- Summary 

    | Model  | Input                  | Handles Trend? | Handles Seasonality? | Handles Multiple Series? | Main Idea                            |
    | ------ | ---------------------- | -------------- | -------------------- | ------------------------ | ------------------------------------ |
    | AR     | Past values            | No             | No                   | No                       | Data predicts itself                 |
    | MA     | Past errors            | No             | No                   | No                       | Past shocks matter                   |
    | ARMA   | Past values + errors   | No             | No                   | No                       | Combines AR + MA                     |
    | ARIMA  | Differenced data       | Yes            | No                   | No                       | Handles trends                       |
    | VAR    | Multiple time series   | Maybe          | No                   | Yes                      | Variables influence each other       |
    | SARIMA | Differenced + seasonal | Yes            | Yes                  | No                       | Trend + seasonality                  |
    | UCM    | Decomposition approach | Yes            | Yes                  | No                       | Trend + season + noise components    |
    | DFM    | Many related series    | Yes            | Maybe                | Yes                      | Few hidden factors drive many series |

1. **Autoregressive Model (AR)**

    **Definition:**
    An autoregressive model predicts the current value based on its own past values.
    It assumes that the past pattern influences the present.

    **Example:**
    Today’s stock price depends partly on yesterday’s price.

    **Formula:**
    $y_t = c + \phi_1 y_{t-1} + \epsilon_t$

2. **Moving Average Model (MA)**

    **Definition:**
    A moving average model predicts the current value based on **past errors (shocks or surprises)** — not the values themselves.
    It assumes random events have lingering effects for a short time.

    **Example:**
    A sudden news impact today may still influence tomorrow’s stock price slightly.

    **Formula:**
    $y_t = c + \theta_1 \epsilon_{t-1} + \epsilon_t$

3. **Autoregressive Moving Average Model (ARMA)**

    **Definition:**
    Combines both AR and MA.
    It uses both **past values and past errors** to explain the present value.

    **Example:**
    Today’s stock price depends on both yesterday’s price (AR part) and yesterday’s market shock (MA part).

    **Formula:**
    $y_t = c + \phi_1 y_{t-1} + \theta_1 \epsilon_{t-1} + \epsilon_t$

4. **Autoregressive Integrated Moving Average Model (ARIMA)**

    **Definition:**
    Adds the “Integrated” part (I), which means **differencing the data** to make it stationary (remove trend).
    It’s used when the series has a **trend or non-stationarity**.

    **Vietnamese:**
    Mô hình ARIMA thêm phần “I” (tích phân) — tức là **lấy sai phân dữ liệu** để loại bỏ xu hướng và làm dữ liệu ổn định hơn.
    Thường dùng cho dữ liệu có **xu hướng tăng hoặc giảm theo thời gian**.

    **Formula:**
    $(1 - B)^d y_t = c + \phi(B)\theta(B)\epsilon_t$

5. **Vector Autoregression Model (VAR)**

    **Definition:**
    An extension of AR but for **multiple time series** that influence each other.
    Each variable depends on **its own past** and the **past of others**.

    **Example:**
    GDP, inflation, and interest rate all affect each other over time.

    **Formula:**
    $y_t = A_1 y_{t-1} + A_2 y_{t-2} + \dots + \epsilon_t$

6. **Seasonal Autoregressive Integrated Moving Average Model (SARIMA)**

    **Definition:**
    An ARIMA model that also captures **seasonal patterns** (repeating cycles).
    It adds seasonal AR, MA, and differencing terms.

    **Example:**
    Sales that rise every December due to Christmas shopping.

7. **Unobserved Components Model (UCM)**

    **Definition:**
    This model decomposes a time series into **unseen (latent) components** such as trend, seasonality, and noise.
    Each part is modeled separately and then combined.

    **Example:**
    Sales = trend (steady growth) + seasonality (monthly pattern) + random shocks.

8. **Dynamic Factor Model (DFM)**

    **Definition:**
    Used when you have **many related time series** (like 100 economic indicators).
    It assumes they share a few **common hidden factors** driving their behavior.

    **Example:**
    Many macroeconomic variables move together because of underlying “economic growth” or “inflation pressure” factors.
