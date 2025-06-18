# Black-Scholes Option Pricing Model
![GitHub top language](https://img.shields.io/github/languages/top/T-KALV/Black-Scholes-Model?style=plastic)
![GitHub repo size](https://img.shields.io/github/repo-size/T-KALV/Black-Scholes-Model?style=plastic)
![GitHub last commit](https://img.shields.io/github/last-commit/T-KALV/Black-Scholes-Model?style=plastic)
![GitHub commits since latest release (by date including pre-releases)](https://img.shields.io/github/commits-since/T-KALV/Black-Scholes-Model/latest?include_prereleases&style=plastic)
![GitHub issues](https://img.shields.io/github/issues/T-KALV/Black-Scholes-Model?style=plastic)
![GitHub all releases](https://img.shields.io/github/downloads/T-KALV/Black-Scholes-Model/total?style=plastic)


![image](https://github.com/user-attachments/assets/cf4c797e-c477-4ee9-a5a9-4346052da3eb)
![image](https://github.com/user-attachments/assets/763318df-982b-4887-9c19-6d76895ddd22)

<!---
![Screenshot From 2025-04-27 12-48-55](https://github.com/user-attachments/assets/9c020d1d-db49-4313-ad25-a907d2720bff)
--->
## About:

### Basic Black-Scholes Algorithm
- provides theoretical estimate of the price of Europoean call/put option

### Formula:
- $ S $ = current price of stock
- $ K $ = strike price of the option
- $ T $ = time to maturity in years
- $ r $ = risk-free interest rate
- $ \sigma $ = volatility of stock (the standard deviation of returns)

### European Call Option:
$$C = S \cdot N(d_1) - K e^{-rT} \cdot N(d_2)$$

### European Put Option:
$$P = K e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)$$

Where:
$$d_1 = \frac{\ln(S/K) + \left(r + \frac{\sigma^2}{2}\right) T}{\sigma \sqrt{T}}$$
$$d_2 = d_1 - \sigma \sqrt{T}$$
$$N(d) = \text{cumulative distribution function of the standard normal distribution}$$

### Newton Raphson (Implied Volatility):
- $ f(\sigma) = \text{BlackScholes}(\sigma) - \text{MarketPrice} = 0 $
- $$\sigma_{\text{new}} = \sigma - \frac{f(\sigma)}{f'(\sigma)} = \sigma - \frac{\text{BlackScholes}(\sigma) - \text{MarketPrice}}{\text{Vega}(\sigma)}$$
- $ \text{Vega}(\sigma) = \text{rate of change of BlackScholes option price wrt changes in volatility } \sigma $

### Option Greeks:
- Delta ($ \Delta $) = measures price sensitivity to underlying:
  - Call option: $ \Delta $ ranges from 0 to 1
  - Put option: $ \Delta $ ranges from -1 to 0
- Gamma ($ \Gamma $) = measures rate of change of delta
- Vega ($ \nu $) = measures sensitivity to volatility
- Theta ($ \Theta $) = measures sensitivity to time decay
- Rho ($ \rho $) = measures sensitivity to interest rates

### Formula for Option Greeks:
- For Calls: 
  $$\Delta = N(d_1)$$
- For Puts: 
  $$\Delta = N(d_1) - 1$$
- where $ N() $ is the standard normal cumulative distribution function
- 
- $$\Gamma = \frac{N'(d_1)}{S \sigma \sqrt{T}}$$
- where $ N'(d_1) $ is the standard normal probability density function
- 
- $$\nu = S \sqrt{T} N'(d_1)$$


## Tasks:
- Implement basic Black Scholes Algotithm
- Read options list from a csv file and ouput the call and put option price using the black scholes algorithm
- Implement implied volatility where we reverse the black sholes algo given a market option price find out what the implied volatility value is
- Visualise Option Prices using matplotlib such as option price vs stock price, option price vs strike price, option price vs volatility, option price vs time to maturity
- Implement simple Steamlit app that shows these results
- Add real time market data integration using Yahoo Finance API to retrieve real-world stock data 
- Add a SQL database backend to store and export user stock option queries
![image](https://github.com/user-attachments/assets/73aded7a-ba87-4cd7-8edc-9c3499a6425a)
![image](https://github.com/user-attachments/assets/ce9aab0f-b049-4449-9b6b-4f9d692dca9d)
![image](https://github.com/user-attachments/assets/444c0a2f-5a10-4e62-aaed-e04f3512d56e)
![image](https://github.com/user-attachments/assets/e8fcc01a-e663-4385-bc83-9ee4b2ed45c7)
