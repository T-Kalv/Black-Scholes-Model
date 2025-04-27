# Black-Scholes-Model
![Screenshot From 2025-04-27 12-48-55](https://github.com/user-attachments/assets/9c020d1d-db49-4313-ad25-a907d2720bff)

## About:

### Basic Black Scholes Algorithm
- provides theoretical estimate of the price of Europoean call/put option

### Formula:
- S = current price of stock
- K = strike price of the option
- T = time to maturity in years
- r = risk-free interest rate
- σ = volatility of stock which is the standard deviation of returns

C = S*N(d1) - (Ke^(-rT)) * N(d2) which is the European Call Option
P = (Ke^(-rT)) * N(-d2) - S*N(-d1) which is the European Put Option

- d1 = (ln(S/K) + (r+(σ^2)/2)T)/(σ*sqrt(T))
- d2 = d1 - σ*sqrt(T)
- N(d) = cumulative distribution function of the standard normal distribution which gives the probability that a normally distributed random varaible is < d

Newton Raphson (Implied Volatility):
- f(σ) = BlackScholes(σ) - MarketPrice = 0
- σnew = σ - (f(σ)/f'(σ)) = σ - (BlackScholes(σ) - MarketPrice/Vega(σ))
- Vega(σ) = rate of change of BlackScholes option price wrt chaanges in volatility σ


## Tasks:
- Implement basic Black Scholes Algotithm
- Read options list from a csv file and ouput the call and put option price using the black scholes algorithm
- Implement implied volatility where we reverse the black sholes algo given a market option price find out what the implied volatility value is
- Visualise Option Prices using matplotlib such as option price vs stock price, option price vs strike price, option price vs volatility, option price vs time to maturity

![image](https://github.com/user-attachments/assets/b7669cb1-c615-4eaa-aab8-4618b4b9e26f)
![image](https://github.com/user-attachments/assets/fa744f71-dcf3-454c-81c4-1b3919e30ee7)
![image](https://github.com/user-attachments/assets/6be9c358-7d8d-4f5f-a178-af7fea0c255a)
![image](https://github.com/user-attachments/assets/41a3796f-2988-4ace-98aa-dc7ec40dd881)
![image](https://github.com/user-attachments/assets/11723d0b-f0b7-4616-8845-f9280bced1bb)
![image](https://github.com/user-attachments/assets/61f4695c-b5f6-42f9-b0f1-170a7601052c)
![image](https://github.com/user-attachments/assets/205cc716-8075-4dea-b2eb-f78b0ca89d7b)
![image](https://github.com/user-attachments/assets/b6154a1c-1be1-4d70-a5ea-67f81e2bc76c)

