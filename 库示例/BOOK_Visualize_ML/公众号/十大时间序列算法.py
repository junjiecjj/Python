
"""



最强总结，十大时间序列算法 ！
https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247484899&idx=1&sn=ddaf00f9556d0dc94c6969def32217c3&chksm=c0e5df25f792563383b34e963472b0c5d70fb7742bc293c789d63fecc3b8e39b41ccccff51be&mpshare=1&scene=1&srcid=07260UeJJ8hygwJU7TXrHtV8&sharer_shareinfo=5bf80316723883c20fc67001dff23aa7&sharer_shareinfo_first=5bf80316723883c20fc67001dff23aa7&exportkey=n_ChQIAhIQ8edcNcWPPzFcT3ddvlUC%2BBKfAgIE97dBBAEAAAAAABAhGiqbB7UAAAAOpnltbLcz9gKNyK89dVj00YOKmC2OLnRLgmP1RjvrDqWaI7vJ7%2B%2BV1vmfeiDesvv1BBMxOXubllth5kHZD5EmYp%2FcnQRhFIdbTFOsS7fGTxidNtu4EK7rIDd7%2F47h5Y1h7%2Bkt2DsCZfVY2U%2BMMHpDmqDsIfIXdsFZKBqJYQGEzWZYXsvHe5PngPSZtLGln5lbW%2FxCPyD3BmbN%2FjFA9If1Bp0JuQSRSdCeAnmyNgDNU2niisBx29DmyPYdY8FsUjMO9avoFP%2FH5iF7WC3giIn%2BZnnXBBSos5HeyW8MbduolDQLJQ2ecFWC7bC5Npaafmo74b2lHc4Kr6FFP57oYR5uo4Pf6Apa4UAY&acctmode=0&pass_ticket=QqF4Sde8zOYNGqSFByHH7x8%2FHME6xJS67gSsVLRDOf8p%2ByVyruW2UvQdkXFwnbIb&wx_header=0#rd













"""
#%% 1. 自回归 (AR, Autoregressive Model)


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 下载数据
ticker = 'AAPL'  # 以苹果公司股票为例
start_date = '2020-01-01'
end_date = '2023-01-01'
data = yf.download(ticker, start=start_date, end=end_date)
close_prices = data['Close']

# 绘制时间序列图
plt.figure(figsize=(14, 7))
plt.plot(close_prices, label='Close Price')
plt.title('Apple Stock Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# ACF和PACF图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
plot_acf(close_prices, ax=ax1, lags=50)
plot_pacf(close_prices, ax=ax2, lags=50)
plt.show()

# 拟合自回归模型
lags = 30
model = AutoReg(close_prices, lags=lags)
model_fit = model.fit()

# 模型预测
pred_start = len(close_prices)
pred_end = pred_start + 50
predictions = model_fit.predict(start=pred_start, end=pred_end)

# 绘制预测结果
plt.figure(figsize=(14, 7))
plt.plot(close_prices, label='Observed')
plt.plot(predictions, label='Forecast', linestyle='--')
plt.title('Apple Stock Close Price Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()









#%% 2. 移动平均 (MA, Moving Average Model)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 生成模拟的时间序列数据
np.random.seed(42)
n_periods = 120
date_range = pd.date_range(start='2010-01', periods=n_periods, freq='M')
seasonal_pattern = np.sin(2 * np.pi * date_range.month / 12)
random_noise = np.random.normal(scale=0.5, size=n_periods)
sales = 10 + seasonal_pattern + random_noise

# 创建数据框
data = pd.DataFrame({'Date': date_range, 'Sales': sales})
data.set_index('Date', inplace=True)

# 绘制原始数据
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['Sales'], label='Original Sales Data')
plt.title('Monthly Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# 使用移动平均模型进行平滑
window_size = 12
data['Sales_MA'] = data['Sales'].rolling(window=window_size).mean()

# 绘制平滑后的数据
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['Sales'], label='Original Sales Data')
plt.plot(data.index, data['Sales_MA'], label=f'{window_size}-month Moving Average', color='red')
plt.title('Monthly Sales Data with Moving Average')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# 计算残差
data['Residual'] = data['Sales'] - data['Sales_MA']

# 绘制残差
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['Residual'], label='Residuals', color='green')
plt.title('Residuals from Moving Average Model')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.legend()
plt.show()

# 绘制自相关图和偏自相关图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_acf(data['Residual'].dropna(), ax=axes[0], lags=40)
plot_pacf(data['Residual'].dropna(), ax=axes[1], lags=40)
axes[0].set_title('ACF of Residuals')
axes[1].set_title('PACF of Residuals')
plt.show()


#%% 3. 自回归滑动平均 (ARMA, Autoregressive Moving Average Model)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 生成示例时间序列数据
np.random.seed(42)
n = 200
ar_params = np.array([0.75, -0.25])
ma_params = np.array([0.65, 0.35])
ar = np.r_[1, -ar_params]  # add zero-lag and negate
ma = np.r_[1, ma_params]   # add zero-lag
y = np.random.normal(size=n)
x = np.convolve(y, ma)[:n] + np.random.normal(size=n)
time_series = pd.Series(x)

# 绘制原始时间序列数据
plt.figure(figsize=(12, 6))
plt.plot(time_series, label='Original Time Series')
plt.title('Original Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# 绘制自相关图 (ACF) 和偏自相关图 (PACF)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_acf(time_series, ax=axes[0], title='Autocorrelation Function (ACF)')
plot_pacf(time_series, ax=axes[1], title='Partial Autocorrelation Function (PACF)')
plt.show()

# 建立并拟合ARMA模型
model = ARIMA(time_series, order=(2, 0, 2))
arma_result = model.fit()

# 打印模型摘要
print(arma_result.summary())

# 绘制拟合后的时间序列和残差图
plt.figure(figsize=(12, 6))
plt.plot(time_series, label='Original Time Series')
plt.plot(arma_result.fittedvalues, color='red', label='Fitted Values')
plt.title('Original and Fitted Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# 绘制残差图
residuals = arma_result.resid
plt.figure(figsize=(12, 6))
plt.plot(residuals, label='Residuals')
plt.title('Residuals of ARMA Model')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.legend()
plt.show()


#%% 4. 自回归积分滑动平均 (ARIMA, Autoregressive Integrated Moving Average Model)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. 创建模拟时间序列数据
np.random.seed(42)
n = 200
time = np.arange(n)
data = np.sin(0.1 * time) + 0.5 * np.random.randn(n)

# 转换为 Pandas DataFrame
df = pd.DataFrame(data, columns=['Value'])
df.index = pd.date_range(start='2020-01-01', periods=n, freq='D')

# 2. 绘制原始时间序列图
plt.figure(figsize=(14, 7))
plt.subplot(3, 1, 1)
plt.plot(df.index, df['Value'], label='Original Data')
plt.title('Original Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()

# 3. 绘制自相关函数 (ACF) 和偏自相关函数 (PACF) 图
plt.subplot(3, 1, 2)
plot_acf(df['Value'], ax=plt.gca(), lags=30)
plt.title('ACF of Time Series')

plt.subplot(3, 1, 3)
plot_pacf(df['Value'], ax=plt.gca(), lags=30)
plt.title('PACF of Time Series')

plt.tight_layout()
plt.show()

# 4. 应用 ARIMA 模型
from statsmodels.tsa.arima.model import ARIMA

# 拟合 ARIMA 模型
model = ARIMA(df['Value'], order=(5, 0, 0))  # (p, d, q) 这里 d=0 是因为数据没有差分
model_fit = model.fit()

# 打印模型摘要
print(model_fit.summary())

# 5. 预测未来 20 个时间点
forecast = model_fit.forecast(steps=20)

# 创建预测数据的时间序列
forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=20, freq='D')
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])

# 6. 绘制预测结果图
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Value'], label='Original Data')
plt.plot(forecast_df.index, forecast_df['Forecast'], color='red', linestyle='--', label='Forecast')
plt.title('Forecast using ARIMA Model')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()



#%% 5. 季节性自回归积分滑动平均 (SARIMA, Seasonal ARIMA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf

# 加载航空乘客数据集
file = 'airline-passengers.csv'
data = pd.read_csv(file, index_col='Month', parse_dates=True)
data.index.freq = 'MS'

# 绘制原始数据
plt.figure(figsize=(10, 6))
plt.plot(data, label='Monthly Airline Passengers')
plt.title('Monthly Airline Passengers from 1949 to 1960')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()

# 进行ADF检验
adf_result = adfuller(data['Passengers'])
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')

# 绘制ACF和PACF图
lag_acf = acf(data['Passengers'], nlags=40)
lag_pacf = pacf(data['Passengers'], nlags=40, method='ols')

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.stem(range(len(lag_acf)), lag_acf, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.stem(range(len(lag_pacf)), lag_pacf, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

# 拟合SARIMA模型
model = SARIMAX(data['Passengers'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# 打印模型总结
print(results.summary())

# 绘制预测结果
data['forecast'] = results.predict(start=120, end=144, dynamic=True)
plt.figure(figsize=(10, 6))
plt.plot(data['Passengers'], label='Actual Passengers')
plt.plot(data['forecast'], label='Forecasted Passengers', color='red')
plt.title('Actual vs Forecasted Passengers')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()


#%% 6. 向量自回归 (VAR, Vector Autoregression)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

# 加载数据集
from statsmodels.datasets.macrodata import load_pandas
data = load_pandas().data

# 选择感兴趣的变量
df = data[['realgdp', 'realcons', 'realinv']]

# 设置时间索引
dates = pd.date_range(start='1959Q1', periods=len(df), freq='Q')
df.index = dates

# 绘制原始数据的时间序列图
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
df.plot(subplots=True, ax=axes)
plt.tight_layout()
plt.show()

# 计算一阶差分
df_diff = df.diff().dropna()

# 绘制差分后的数据
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
df_diff.plot(subplots=True, ax=axes)
plt.tight_layout()
plt.show()

# 构建并训练VAR模型
model = VAR(df_diff)
results = model.fit(maxlags=15, ic='aic')

# 打印模型摘要
print(results.summary())

# 绘制模型残差的时间序列图
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
for i, column in enumerate(df_diff.columns):
    axes[i].plot(results.resid[:, i])
    axes[i].set_title(f'Residuals of {column}')
plt.tight_layout()
plt.show()

# 预测未来的时间序列
lag_order = results.k_ar
forecast = results.forecast(df_diff.values[-lag_order:], steps=10)
forecast_index = pd.date_range(start=df.index[-1], periods=10, freq='Q')
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=df.columns)

# 绘制预测结果
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
for i, column in enumerate(df.columns):
    axes[i].plot(df.index, df[column], label='Original')
    axes[i].plot(forecast_df.index, forecast_df[column], label='Forecast')
    axes[i].set_title(f'{column} - Original vs Forecast')
    axes[i].legend()
plt.tight_layout()
plt.show()



#%% 7. 向量自回归滑动平均 (VARMA, Vector Autoregressive Moving Average)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.vector_ar.var_model import VAR

# 生成模拟数据
np.random.seed(42)
n_obs = 100
sales = np.random.normal(loc=100, scale=15, size=n_obs)
advertising = np.random.normal(loc=50, scale=10, size=n_obs)

# 创建DataFrame
data = pd.DataFrame({'Sales': sales, 'Advertising': advertising})

# 拆分数据为训练集和测试集
train = data.iloc[:80]
test = data.iloc[80:]

# 拟合VARMA模型
model = VARMAX(train, order=(1, 1))
results = model.fit(maxiter=1000, disp=False)
print(results.summary())

# 预测未来值
forecast = results.forecast(steps=len(test))

# 绘制销售量和广告支出的时间序列及预测结果
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(train['Sales'], label='Actual Sales (Train)')
plt.plot(test.index, forecast['Sales'], label='Forecasted Sales')
plt.title('Sales Forecast using VARMA(1,1)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(train['Advertising'], label='Actual Advertising (Train)')
plt.plot(test.index, forecast['Advertising'], label='Forecasted Advertising')
plt.title('Advertising Forecast using VARMA(1,1)')
plt.legend()

plt.tight_layout()
plt.show()



#%% 8. 长短期记忆网络 (LSTM, Long Short-Term Memory)




#%% 9. Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# 生成示例数据
dates = pd.date_range(start='2020-01-01', periods=730, freq='D')
data = np.random.poisson(lam=200, size=730) + np.linspace(0, 100, 730)
df = pd.DataFrame({'ds': dates, 'y': data})

# 初始化并训练Prophet模型
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(df)

# 创建未来的数据框架并进行预测
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# 绘制原始数据及预测结果
fig1 = model.plot(forecast)
plt.title('Original Data and Forecast')
plt.xlabel('Date')
plt.ylabel('Website Traffic')

# 绘制趋势和季节性成分
fig2 = model.plot_components(forecast)
plt.show()

# 绘制实际数据与预测值的对比
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['y'], label='Actual')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
plt.title('Actual vs Forecast')
plt.xlabel('Date')
plt.ylabel('Website Traffic')
plt.legend()
plt.show()

# 绘制残差图
residuals = df['y'] - forecast['yhat'][:len(df)]
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], residuals)
plt.axhline(0, linestyle='--', color='red')
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.show()



#%% 10. 变分自编码器 (VAE, Variational Autoencoders)


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 生成正弦波时间序列数据
def generate_sine_wave(seq_length, num_samples):
    x = np.linspace(0, np.pi * 2 * num_samples, seq_length * num_samples)
    y = np.sin(x)
    data = y.reshape(num_samples, seq_length, 1)
    return data

seq_length = 50
num_samples = 1000
data = generate_sine_wave(seq_length, num_samples)

# 数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = TimeSeriesDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义VAE模型：
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # 输出mean和logvar
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mean_logvar = self.encoder(x)
        mean, logvar = mean_logvar[:, :latent_dim], mean_logvar[:, latent_dim:]
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar

# 训练VAE模型：
input_dim = seq_length
hidden_dim = 128
latent_dim = 16
num_epochs = 50
learning_rate = 0.001

vae = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

def loss_function(recon_x, x, mean, logvar):
    recon_loss = criterion(recon_x, x)
    kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kld_loss

for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        recon_batch, mean, logvar = vae(batch)
        loss = loss_function(recon_batch, batch.view(-1, input_dim), mean, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {train_loss / len(dataloader.dataset)}")

# 绘制原始时间序列与重构时间序列的对比图
vae.eval()
with torch.no_grad():
    for batch in dataloader:
        recon_batch, mean, logvar = vae(batch)
        break

recon_batch = recon_batch.view(-1, seq_length).numpy()
original_batch = batch.numpy()

plt.figure(figsize=(12, 6))
plt.plot(original_batch[0], label="Original")
plt.plot(recon_batch[0], label="Reconstructed")
plt.legend()
plt.title("Original vs Reconstructed Time Series")
plt.show()


# 绘制VAE潜在空间的可视化图
latent_vectors = []
vae.eval()
with torch.no_grad():
    for batch in dataloader:
        _, mean, _ = vae(batch)
        latent_vectors.append(mean.numpy())

latent_vectors = np.concatenate(latent_vectors, axis=0)

plt.figure(figsize=(10, 8))
plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], alpha=0.5)
plt.title("Latent Space Visualization")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.show()



