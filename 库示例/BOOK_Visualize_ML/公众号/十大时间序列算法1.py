#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:27:34 2024

@author: jack
https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247486710&idx=1&sn=3482a260213272aaaf7bfa77cd8a8787&chksm=c1b99c7bdca46156f1ba5ddbe776cb164d6cab7e668a0e0089ddec5720e0643b44f3c5b448c0&mpshare=1&scene=1&srcid=09128x2XRlEPK2Vgoev5S4vI&sharer_shareinfo=1e2d0a588b1d3fe7db68370e23477627&sharer_shareinfo_first=1e2d0a588b1d3fe7db68370e23477627&exportkey=n_ChQIAhIQH2BzNocWoLIxrR4nWoOz8BKfAgIE97dBBAEAAAAAACg6LDkJGb4AAAAOpnltbLcz9gKNyK89dVj0SuTc2GXsmwIECRtmqACeKlSCziP0SFSsnyHBqfLBthCmWPqTRl1HMH1535apiazknRTGiScMV%2BTSrYxml%2B%2FkmSqSJ4oRIRGm32jrVV3OHELZH3jc%2FkQ5jax5TuDJ4KgHq9nUlNV10cuURTlhKt66%2FbItAJAqtDmRsjJYLemQCQS%2BE07Zw%2FpYo4YkdtDYdb%2FD%2FBG%2BS33qTWd1t8QwgOAnAKEgh%2B%2FETJkYMQaSw1IsPGRT7x160lBls5A1Em9j7MUVEYxpFGU626Klof6QgyZmo6QCwLSM8xhi%2BE4RWVldAOaN2Xt5xZlR5uSgKXb4YqYAVencX1ULnqz9&acctmode=0&pass_ticket=ptgfzzs26E64MnOFp4Iu4%2FCD7B8UxedgaX4QEisYu5xJ1QQ26lMwRl0Gba4DpAEQ&wx_header=0#rd


"""



#%%>>>>>>>>>>>>>>>>>>>>>>> 1. AR, 自回归模型
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# 设置随机种子以便复现
np.random.seed(42)

# 生成虚拟时间序列数据，假设是基于AR(3)模型
n = 2000
coef = [0.6, -0.2, 0.1]  # AR模型的系数
lags = len(coef)
noise = np.random.normal(scale=0.5, size=n)
data = [0, 0, 0]  # 初始化前几个值
for i in range(lags, n):
    value = sum(coef[j] * data[i - 1 - j] for j in range(lags)) + noise[i]
    data.append(value)

# 拆分为训练集和测试集
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# 拟合AR模型
model = AutoReg(train, lags=lags)
model_fit = model.fit()

# 预测
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# 计算均方误差
error = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {error}')

# 图形绘制
plt.figure(figsize = (14, 7))

# 原始数据
plt.plot(range(len(data)), data, label = 'Original Data', color = 'cyan', linewidth = 2)

# 训练数据拟合值
fitted_values = model_fit.fittedvalues
plt.plot(range(lags, len(train)), fitted_values, label = 'Fitted Values (Train)', color = 'magenta', linestyle = '--', linewidth = 2)

# 测试数据预测值
plt.plot(range(len(train), len(train)+len(test)), predictions, label = 'Predictions (Test)', color = 'red', linewidth = 2)

# 显示垂直分割线
plt.axvline(x = len(train)-1, color = 'blue', linestyle = ':', label = 'Train/Test Split')

# 设置图形样式
plt.title('AutoRegressive (AR) Model - Time Series Analysis', fontsize=16)
plt.xlabel('Time Steps', fontsize=14)
plt.ylabel('Value', fontsize = 14)
plt.legend(loc='best', fontsize=12)

# 显示图形
plt.grid(True)
plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 2. MA, 移动平均模型

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置随机种子以保证可重复性
np.random.seed(42)

# 生成虚拟时间序列数据
time = np.arange(0, 100, 0.1)
original_data = np.sin(time) + np.cos(time / 2)

# 添加噪声
noise = np.random.normal(0, 0.5, len(time))
noisy_data = original_data + noise

# 计算移动平均 (MA)
window_size = 10
moving_average = np.convolve(noisy_data, np.ones(window_size) / window_size, mode = 'valid')

# 计算误差
error = original_data[(window_size-1):] - moving_average

# 创建图形
plt.figure(figsize=(14, 10))

# 原始数据和噪声数据图
plt.subplot(3, 1, 1)
plt.plot(time, original_data, label='Original Data', color='blue', linewidth=2)
plt.plot(time, noisy_data, label='Noisy Data', color='red', linestyle='--', alpha=0.7)
plt.title('Original and Noisy Data')
plt.legend(loc='upper right')

# 噪声数据与移动平均图
plt.subplot(3, 1, 2)
plt.plot(time[(window_size-1):], moving_average, label='Moving Average', color='green', linewidth=2)
plt.plot(time, noisy_data, label='Noisy Data', color='red', linestyle='--', alpha=0.7)
plt.title('Noisy Data and Moving Average')
plt.legend(loc='upper right')

# 移动平均误差图
plt.subplot(3, 1, 3)
plt.plot(time[(window_size-1):], error, label='Error (Original - MA)', color='purple', linewidth=2)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Error between Original Data and Moving Average')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 3. ARMA, 自回归移动平均模型
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# 生成虚拟的ARMA(2,2)模型数据
np.random.seed(42)

# 定义AR参数和MA参数
ar_params = np.array([1, -0.5, 0.25])  # AR(2): 1 - 0.5L + 0.25L^2
ma_params = np.array([1, 0.4, 0.3])    # MA(2): 1 + 0.4L + 0.3L^2

# 创建ARMA进程
arma_process = ArmaProcess(ar_params, ma_params)

# 生成1000个样本
n_samples = 1000
y = arma_process.generate_sample(nsample=n_samples)

# 转换为DataFrame格式
data = pd.DataFrame(y, columns=['ARMA(2,2)'])

# 创建ARMA模型并进行拟合
model = ARIMA(data, order=(2, 0, 2))
model_fitted = model.fit()

# 预测 (仅返回预测值)
forecast = model_fitted.forecast(steps=50)

# 获取预测的置信区间
conf_int = model_fitted.get_forecast(steps=50).conf_int(alpha=0.05)

# 设置图形大小和颜色
fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

# 原始ARMA时间序列图
axs[0, 0].plot(data, color='blue', label='ARMA(2,2)', linewidth=2)
axs[0, 0].set_title('ARMA(2,2) Time Series', fontsize=14)
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Value')
axs[0, 0].grid(True)
axs[0, 0].legend()

# 自相关图（ACF）
plot_acf(data, ax=axs[0, 1], lags=40, color='red')
axs[0, 1].set_title('ACF of ARMA(2,2)', fontsize=14)

# 偏自相关图（PACF）
plot_pacf(data, ax=axs[1, 0], lags=40, color='green')
axs[1, 0].set_title('PACF of ARMA(2,2)', fontsize=14)

# 预测结果图
axs[1, 1].plot(np.arange(n_samples, n_samples + 50), forecast, color='orange', label='Forecast', linewidth=2)
axs[1, 1].fill_between(np.arange(n_samples, n_samples + 50), conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3)
axs[1, 1].set_title('ARMA(2,2) Forecast (50 steps)', fontsize=14)
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Forecasted Value')
axs[1, 1].grid(True)
axs[1, 1].legend()

# 显示图形
plt.suptitle('ARMA(2,2) Model Analysis', fontsize=16, color='purple')
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>> 4. ARIMA, 自回归积分滑动平均模型
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters

# 注册 matplotlib converters，用于时间序列绘图
register_matplotlib_converters()

# 生成虚拟时间序列数据
np.random.seed(42)
time_points = 100
time = pd.date_range(start='2020-01-01', periods=time_points, freq='M')
data = np.cumsum(np.random.normal(loc=0.5, scale=2, size=time_points))

# 创建 DataFrame
df = pd.DataFrame({'Date': time, 'Value': data})
df.set_index('Date', inplace=True)

# ARIMA 模型的参数 (p, d, q)
p, d, q = 2, 1, 2

# 拟合 ARIMA 模型
model = ARIMA(df['Value'], order=(p, d, q))
fitted_model = model.fit()

# 生成预测值
forecast_steps = 12
forecast = fitted_model.forecast(steps=forecast_steps)
forecast_index = pd.date_range(df.index[-1], periods=forecast_steps + 1, freq='M')[1:]

# 残差
residuals = fitted_model.resid

# 创建图形
plt.figure(figsize=(12, 8))

# 子图 1：时间序列数据与预测值
plt.subplot(2, 2, 1)
plt.plot(df.index, df['Value'], label='Original Data', color='cyan', linewidth=2)
plt.plot(forecast_index, forecast, label='Forecast', color='magenta', linestyle='--', linewidth=2)
plt.title('Time Series Data and Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()

# 子图 2：模型拟合值 vs 原始数据
plt.subplot(2, 2, 2)
plt.plot(df.index, df['Value'], label='Original Data', color='blue', linewidth=2)
plt.plot(df.index, fitted_model.fittedvalues, label='Fitted Values', color='orange', linestyle='--', linewidth=2)
plt.title('Fitted Values vs Original Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()

# 子图 3：残差图
plt.subplot(2, 2, 3)
plt.plot(df.index, residuals, label='Residuals', color='red', linewidth=2)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend()

# 子图 4：残差的直方图
plt.subplot(2, 2, 4)
plt.hist(residuals, bins=20, color='green', edgecolor='black')
plt.title('Residuals Histogram')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')

# 调整图形布局
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 5. SARIMA, 季节性ARIMA模型
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# 设置图像样式
sns.set(style="whitegrid")

# 模拟一个简单的时间序列
np.random.seed(42)
time_points = 200
trend = np.linspace(0, 10, time_points)
seasonal_pattern = 10 + np.sin(np.linspace(0, 2 * np.pi, time_points)) * 5
noise = np.random.normal(0, 1, time_points)
data = trend + seasonal_pattern + noise

# 创建DataFrame
dates = pd.date_range(start='2020-01-01', periods=time_points, freq='M')
df = pd.DataFrame({'Date': dates, 'Value': data})
df.set_index('Date', inplace=True)

# 案例1：SARIMA模型分析
model1 = SARIMAX(df['Value'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
result1 = model1.fit()

# 获取置信区间，并将其与原始数据对齐
conf_int1 = result1.conf_int()
conf_int1 = conf_int1.iloc[-len(df):]

# 案例2：复杂季节性序列 + 分解
seasonal_complex = 5 * np.cos(np.linspace(0, 4 * np.pi, time_points))
complex_data = trend + seasonal_pattern + seasonal_complex + noise
df_complex = pd.DataFrame({'Date': dates, 'Value': complex_data})
df_complex.set_index('Date', inplace=True)

# SARIMA分析
model2 = SARIMAX(df_complex['Value'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
result2 = model2.fit()

# 获取置信区间，并将其与原始数据对齐
conf_int2 = result2.conf_int()
conf_int2 = conf_int2.iloc[-len(df_complex):]

# 分解复杂时间序列
decomposition = seasonal_decompose(df_complex['Value'], model='additive', period=12)

# 绘制图像
plt.figure(figsize=(14, 10))

# 案例1: 简单时间序列及其预测
plt.subplot(2, 2, 1)
plt.plot(df.index, df['Value'], label='Observed', color='blue', linewidth=2)
plt.plot(df.index, result1.fittedvalues, label='SARIMA Fitted', color='orange', linestyle='--', linewidth=2)
plt.fill_between(df.index[-len(conf_int1):], conf_int1.iloc[:, 0], conf_int1.iloc[:, 1], color='gray', alpha=0.3)
plt.title('Case 1: Simple Time Series with SARIMA Model')
plt.legend()

# 案例2: 复杂季节性序列及其预测
plt.subplot(2, 2, 2)
plt.plot(df_complex.index, df_complex['Value'], label='Observed', color='green', linewidth=2)
plt.plot(df_complex.index, result2.fittedvalues, label='SARIMA Fitted', color='red', linestyle='--', linewidth=2)
plt.fill_between(df_complex.index[-len(conf_int2):], conf_int2.iloc[:, 0], conf_int2.iloc[:, 1], color='gray', alpha=0.3)
plt.title('Case 2: Complex Seasonal Time Series with SARIMA Model')
plt.legend()

# 案例2: 季节性分解 - 趋势
plt.subplot(2, 2, 3)
plt.plot(decomposition.trend.index, decomposition.trend, color='purple', linewidth=2)
plt.title('Decomposition: Trend Component')

# 案例2: 季节性分解 - 季节性
plt.subplot(2, 2, 4)
plt.plot(decomposition.seasonal.index, decomposition.seasonal, color='cyan', linewidth=2)
plt.title('Decomposition: Seasonal Component')

# 调整图像布局
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 6. VAR, 向量自回归模型
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import seaborn as sns

# 设置随机种子
np.random.seed(42)

# 创建虚拟数据集
n_obs = 100  # 样本数量
time = np.arange(n_obs)
data = {
    'Variable1': np.sin(0.1 * time) + np.random.normal(scale=0.5, size=n_obs),
    'Variable2': np.cos(0.1 * time) + np.random.normal(scale=0.5, size=n_obs)
}
df = pd.DataFrame(data)

# 构建VAR模型
model = VAR(df)
results = model.fit(4)  # 使用4阶的VAR模型

# 预测未来10个点
lag_order = results.k_ar
forecast_input = df.values[-lag_order:]
forecast = results.forecast(y=forecast_input, steps=10)
forecast_df = pd.DataFrame(forecast, columns=['Variable1_Forecast', 'Variable2_Forecast'])

# 创建时间序列图
plt.figure(figsize=(12, 6))

# 时间序列实际值
plt.subplot(1, 2, 1)
sns.lineplot(x=time, y=df['Variable1'], label='Variable 1', color='orange', linewidth=2.5)
sns.lineplot(x=time, y=df['Variable2'], label='Variable 2', color='blue', linewidth=2.5)
plt.title('Actual Time Series')
plt.xlabel('Time')
plt.ylabel('Values')
plt.grid(True)
plt.legend()

# 预测值与实际值对比
plt.subplot(1, 2, 2)
extended_time = np.arange(n_obs + 10)
sns.lineplot(x=time, y=df['Variable1'], label='Variable 1 Actual', color='orange', linestyle='--', linewidth=2.5)
sns.lineplot(x=time, y=df['Variable2'], label='Variable 2 Actual', color='blue', linestyle='--', linewidth=2.5)
sns.lineplot(x=extended_time[-10:], y=forecast_df['Variable1_Forecast'], label='Variable 1 Forecast', color='red', linewidth=2.5)
sns.lineplot(x=extended_time[-10:], y=forecast_df['Variable2_Forecast'], label='Variable 2 Forecast', color='green', linewidth=2.5)
plt.title('Forecast vs Actual')
plt.xlabel('Time')
plt.ylabel('Values')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 7. GARCH
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# 1. 生成虚拟数据集：AR(1) + GARCH(1, 1)
np.random.seed(42)
n = 1000
omega = 0.1
alpha = 0.3
beta = 0.5

# 生成正态分布的随机噪声
eps = np.random.normal(0, 1, n)

# 创建波动率的初始值
variance = np.zeros(n)
y = np.zeros(n)
variance[0] = omega / (1 - alpha - beta)

# 使用GARCH(1,1)生成数据
for t in range(1, n):
    variance[t] = omega + alpha * (eps[t-1]**2) + beta * variance[t-1]
    y[t] = np.sqrt(variance[t]) * eps[t]

# 转换为时间序列数据
dates = pd.date_range('2020-01-01', periods=n, freq='D')
returns = pd.Series(y, index=dates)

# 2. 拟合 GARCH 模型
garch_model = arch_model(returns, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(disp="off")

# 3. 绘制多个数据分析图
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 图1: 时间序列数据
axes[0].plot(returns, color='blue', label='Simulated Returns')
axes[0].set_title('Simulated Time Series Data', fontsize=15)
axes[0].set_ylabel('Returns', fontsize=12)
axes[0].legend(loc='upper right')

# 图2: 拟合的波动率
axes[1].plot(returns.index, garch_fit.conditional_volatility, color='red', label='Fitted Volatility')
axes[1].set_title('Fitted Conditional Volatility (GARCH(1,1))', fontsize=15)
axes[1].set_ylabel('Volatility', fontsize=12)
axes[1].legend(loc='upper right')

# 图3: 残差的自相关性
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(garch_fit.resid, ax=axes[2], lags=30, alpha=0.05, color='green')
axes[2].set_title('ACF of Residuals', fontsize=15)
axes[2].set_ylabel('Autocorrelation', fontsize=12)

# 全局布局
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 8. Holt-Winters 指数平滑法
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from matplotlib.dates import DateFormatter

# 生成虚拟时间序列数据
np.random.seed(42)
periods = 200
time = pd.date_range(start='2024-01-01', periods=periods, freq='D')
trend = np.linspace(20, 100, periods)
seasonality = 10 * np.sin(np.linspace(0, 3 * np.pi, periods))
noise = np.random.normal(0, 3, periods)
data = trend + seasonality + noise

# 创建一个DataFrame
df = pd.DataFrame({'Date': time, 'Value': data})
df.set_index('Date', inplace=True)

# 应用 Holt-Winters 指数平滑法（加性模型）
model = ExponentialSmoothing(df['Value'], trend='add', seasonal='add', seasonal_periods=30)
fit_model = model.fit()

# 预测未来50天
forecast_periods = 50
forecast = fit_model.forecast(forecast_periods)

# 计算残差
df['Fitted'] = fit_model.fittedvalues
df['Residual'] = df['Value'] - df['Fitted']

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [2, 1]})
fig.suptitle('Holt-Winters Time Series Analysis', fontsize=16)

# 图 1: 原始数据和预测数据
axes[0, 0].plot(df.index, df['Value'], label='Original Data', color='blue', linewidth=2)
axes[0, 0].plot(df.index, df['Fitted'], label='Fitted Data', color='red', linestyle='--')
axes[0, 0].plot(pd.date_range(df.index[-1], periods=forecast_periods, freq='D'), forecast, label='Forecast', color='green', linestyle='--')
axes[0, 0].set_title('Original vs Fitted Data with Forecast', fontsize=14)
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Value')
axes[0, 0].legend()

# 图 2: 残差分布
axes[0, 1].hist(df['Residual'], bins=20, color='purple', edgecolor='black')
axes[0, 1].set_title('Residuals Distribution', fontsize=14)
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Frequency')

# 图 3: 残差随时间变化
axes[1, 0].plot(df.index, df['Residual'], label='Residuals', color='orange', linewidth=1.5)
axes[1, 0].axhline(0, linestyle='--', color='black', linewidth=1)
axes[1, 0].set_title('Residuals Over Time', fontsize=14)
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Residual')
axes[1, 0].legend()

# 图 4: 季节性成分提取可视化
seasonal_periods = 30  # 设置季节周期
seasonal_effects = df['Fitted'] - (fit_model.level + fit_model.trend)  # 提取季节性成分
axes[1, 1].plot(seasonal_effects[:seasonal_periods], label='Seasonal Component (First Cycle)', color='green', linewidth=2)
axes[1, 1].set_title('Extracted Seasonal Component', fontsize=14)
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Seasonal Effect')
axes[1, 1].legend()

# 美化图表
fig.autofmt_xdate()
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>> 9. LSTM


#%%>>>>>>>>>>>>>>>>>>>>>>> 10. XGBoost 时间序列模型

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 设置随机种子
np.random.seed(42)

# 生成虚拟时间序列数据
n_points = 200
X = np.arange(n_points)
y = 10 * np.sin(0.1 * X) + np.random.normal(0, 0.5, n_points)  # 基于正弦波生成带噪声的数据

# 构造特征，采用滑动窗口法
def create_features(X, y, window_size=10):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(y[i:i+window_size])
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

# 创建特征
window_size = 10
X_features, y_target = create_features(X, y, window_size)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, shuffle=False)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建 XGBoost 模型并训练
model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train_scaled, y_train)

# 进行预测
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# 计算均方误差
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"训练集RMSE: {train_rmse:.4f}")
print(f"测试集RMSE: {test_rmse:.4f}")

# 可视化
plt.figure(figsize=(14, 8))

# 子图1: 原始时间序列和噪声
plt.subplot(2, 1, 1)
plt.plot(X, y, color='blue', label='Original Data with Noise', linewidth=2)
plt.title("Original Time Series Data", fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.grid(True)
plt.legend()

# 子图2: 训练集、测试集和预测值
plt.subplot(2, 1, 2)
plt.plot(np.arange(window_size, len(y_train_pred) + window_size), y_train_pred, color='green', label='Train Predictions', linewidth=2)
plt.plot(np.arange(len(y) - len(y_test), len(y)), y_test_pred, color='red', label='Test Predictions', linewidth=2)
plt.plot(np.arange(len(y)), y, color='blue', alpha=0.5, label='Actual Values', linewidth=2)
plt.title("Train and Test Predictions vs Actual Values", fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.grid(True)
plt.legend()

# 显示图像
plt.tight_layout()
plt.show()






















