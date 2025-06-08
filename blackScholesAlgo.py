# Program: blackScholeAlgo.py
# Author: T-Kalv
# Module: 
# Email: 
# Student Number:
"""
About:

Basic Black Scholes Algorithm
- provides theoretical estimate of the price of Europoean call/put option

Formula:
- S = current price of stock
- K = strike price of the option
- T = time to maturity in years
- r = risk-free interest rate
- σ = volatility of stock which is the standard deviation of returns

C = S*N(d1) - (Ke^(-rT)) * N(d2) which is the European Call Option
P = (Ke^(-rT)) * N(-d2) - S*N(-d1) which is the European Put Option

d1 = (ln(S/K) + (r+(σ^2)/2)T)/(σ*sqrt(T))
d2 = d1 - σ*sqrt(T)
N(d) = cumulative distribution function of the standard normal distribution which gives the probability that a normally distributed random varaible is < d

Newton Raphson (Implied Volatility):
f(σ) = BlackScholes(σ) - MarketPrice = 0
σnew = σ - (f(σ)/f'(σ)) = σ - (BlackScholes(σ) - MarketPrice/Vega(σ))
- Vega(σ) = rate of change of BlackScholes option price wrt chaanges in volatility σ


Tasks:
- Implement basic Black Scholes Algotithm
- Read options list from a csv file and ouput the call and put option price using the black scholes algorithm
- Implement implied volatility where we reverse the black sholes algo given a market option price find out what the implied volatility value is
- Visualise Option Prices using matplotlib such as option price vs stock price, option price vs strike price, option price vs volatility, option price vs time to maturity
- Implement simple Steamlit app that shows these results
"""

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Code:
import math
from scipy.stats import norm
import csv
import streamlit as st
import matplotlib.pyplot as plt

def europeanCallOption(S, K, T, r, σ):
    d1 = (math.log(S/K) + (r+0.5*σ**2) * T) / (σ * math.sqrt(T))
    d2 = d1 - σ * math.sqrt(T)

    callPrice = S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
    return callPrice

def europeanPutOption(S, K, T, r, σ):
    d1 = (math.log(S/K) + (r + 0.5 * σ ** 2) * T) / (σ * math.sqrt(T))
    d2 = d1 - σ * math.sqrt(T)

    putPrice = K * math.exp(-r*T) * norm.cdf(-d2) - S*norm.cdf(-d1)
    return putPrice

def optionListData(inputFileName, ouputFileName = "results.csv"):
    with open(inputFileName, newline='') as csvfile, open(ouputFileName, mode = 'w', newline='') as outfile:
        readData = csv.DictReader(csvfile)
        columnNames = ['S', 'K', 'T', 'r', 'σ', 'Call Option Price', 'Put Option Price']
        writeData = csv.DictWriter(outfile, fieldnames=columnNames)
        writeData.writeheader()
        for i in readData:# read through each row in the csv file and get the relevant values and pass that into the black schole algo
            S = i['S']
            K = i['K']
            T = i['T']
            r = i['r']
            σ = i['σ']

            callPrice = europeanCallOption(float(S), float(K), float(T), float(r), float(σ))
            putPrice = europeanPutOption(float(S), float(K), float(T), float(r), float(σ))

            print(f"S = {S}, K = {K}, T = {T}, r = {r}, σ = {σ}")
            print(f"Call Option Price: {callPrice}")
            print(f"Put Option Price: {putPrice}")

            writeData.writerow({'S': S, 'K':K, 'T':T, 'r':r, 'σ':σ, 'Call Option Price': callPrice, 'Put Option Price' : putPrice})

def Vega(S, K, T, r, σ):
    d1 = (math.log(S/K) + (r+0.5*σ**2) * T) / (σ * math.sqrt(T))
    return S * norm.pdf(d1) * math.sqrt(T)

def impliedVolatility(optionType, marketValue, S, K, T, r, tolerance = 1e-6, maxIterationsNum = 100):
    σ = 0.1
    iteration = 0

    while iteration < maxIterationsNum:
        if optionType == "call":
            value = europeanCallOption(S, K, T, r, σ)
        else:
            value = europeanPutOption(S, K, T, r, σ)
        vega = optionsVega(S, K, T, r, σ)
        difference = value - marketValue
        if abs(difference) < tolerance:
            return σ
        if vega == 0:
            break
        σ -= difference / vega
        iteration =  iteration + 1

    return None #when there is no converges

def optionsVega(S, K, T, r, σ):
    d1 = (math.log(S/K) + (r + 0.5 * σ ** 2) * T) / (σ * math.sqrt(T))
    return S * norm.pdf(d1) * math.sqrt(T)

def visualiseOptionStock(S, K, T, r, σ, optionType):
    x = list(range(50, 150))#testing
    y = []
    for s in x:
        if optionType == "call":
            value = europeanCallOption(s, K, T, r, σ)
        else:
            value = europeanCallOption(s, K, T, r, σ)
        y.append(value)
    plt.plot(x, y, color="grey")
    plt.title(f"{optionType} Option Price Vs Stock Price")
    plt.xlabel("Stock Price")
    plt.ylabel("Option Price")
    plt.grid(True)
    plt.show()

def visualiseOptionStrike(S, K, T, r, σ, optionType):
    x = list(range(50, 150))#testing
    y = []
    for k in x:
        if optionType == "call":
            value = europeanCallOption(S, k, T, r, σ)
        else:
            value = europeanPutOption(S, k, T, r, σ)
        y.append(value)
    plt.plot(x, y, color="grey")
    plt.title(f"{optionType} Option Price Vs Strike Price")
    plt.xlabel("Strike Price")
    plt.ylabel("Option Price")
    plt.grid(True)
    plt.show()

def visualiseOptionMaturity(S, K, r, σ, optionType):
    x = [i/10 for i in range(1, 50)]#testing
    y = []
    for t in x:
        if optionType == "call":
            value = europeanCallOption(S, K, t, r, σ)
        else:
            value = europeanPutOption(S, K, t, r, σ)
        y.append(value)
    plt.plot(x, y, color="grey")
    plt.title(f"{optionType} Option Price Vs Time To Maturity")
    plt.xlabel("Time to Maturity")
    plt.ylabel("Option Price")
    plt.grid(True)
    plt.show()

def visualiseOptionVolatility(S, K, T, r, σ, optionType):
    x = [i/10 for i in range(1, 50)]#testing
    y = []
    for σ in x:
        if optionType == "call":
            value = europeanCallOption(S, K, T, r, σ)
        else:
            value = europeanPutOption(S, K, T, r, σ)
        y.append(value)
    plt.plot(x, y, color="grey")
    plt.title(f"{optionType} Option Price Vs Volatility")
    plt.xlabel("Volatility")
    plt.ylabel("Option Price")
    plt.grid(True)
    plt.show()    



def Main2():
    optionType = input("Enter Option Type: ")
    S = float(input("Enter Current Stock Price (S): "))
    K = float(input("Enter Strike Price (K): "))
    T = float(input("Enter Time To Maturity (T): "))
    r = float(input("Enter Risk-Free Rate (r): "))
    marketPrice = float(input("Enter Market Option Price: "))

    impliedVol = impliedVolatility(optionType, marketPrice, S, K, T, r)
    if impliedVol is not None:
        print(f"Implied Volatility: {impliedVol}")
    else:
        print("No Convergence On Implied Volatility!")


def Main():
    S = input("Enter Current Stock Price (S): ")
    K = input("Enter Strike Price (K): ")
    T = input("Enter Time To Maturity In Year (T): ")
    r = input("Enter Risk-Free Interest Rate (r): ")
    σ = input("Enter Volatility (σ): ")

    callPrice = europeanCallOption(float(S), float(K), float(T), float(r), float(σ))
    putPrice = europeanPutOption(float(S), float(K), float(T), float(r), float(σ))

    print(f"Call Option Price: {callPrice}")
    print(f"Put Option Price: {putPrice}")
    visualiseOptionStock(float(S), float(K), float(T), float(r), float(σ), "call")
    visualiseOptionStock(float(S), float(K), float(T), float(r), float(σ), "put")
    visualiseOptionStrike(float(S), float(K), float(T), float(r), float(σ), "call")
    visualiseOptionStrike(float(S), float(K), float(T), float(r), float(σ), "put")
    visualiseOptionMaturity(float(S), float(K), float(r), float(σ), "call")
    visualiseOptionMaturity(float(S), float(K), float(r), float(σ), "put")
    visualiseOptionVolatility(float(S), float(K), float(T), float(r), float(σ), "call")
    visualiseOptionVolatility(float(S), float(K), float(T), float(r), float(σ), "put")


def optionCurveGraph(x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    ax.plot(x, y, color="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    st.pyplot(fig)

def StreamlitInterface():
    plt.style.use('dark_background')
    st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📈",
    initial_sidebar_state="expanded")
    st.title("Black-Scholes Option Pricing Model")
    #st.caption("Black-Scholes Algorithm Visualiser")
    S = st.number_input("Enter Current Stock Price(S): ", value=0.0)
    K = st.number_input("Enter Strike Price(K): ", value=0.0)
    T = st.number_input("Enter Time To Maturity In Year(T): ", value=0.0)
    r = st.number_input("Enter Risk-Free Interest Rate(r): ", value=0.0)
    σ  = st.number_input("Enter Volatility(σ): ", value=0.0)

    if st.button("Calculate & Visualise"):
        callPrice = europeanCallOption(S, K, T, r, σ)
        putPrice  = europeanPutOption(S, K, T, r, σ)
        st.success(f"Call Price: {callPrice:.4f}")
        st.success(f"Put  Price: {putPrice:.4f}")
        stockPrices  = list(range(50, 151))
        strikePrices = list(range(50, 151))
        maturities   = [i/10 for i in range(1, 51)]
        volatilities = [i/10 for i in range(1, 51)]

        y1 = [europeanCallOption(s, K, T, r, σ) for s in stockPrices]
        optionCurveGraph(stockPrices, y1, "Stock Price",         "Call Price", "Call Price vs Stock Price")
        y2 = [europeanCallOption(S, k, T, r, σ) for k in strikePrices]
        optionCurveGraph(strikePrices, y2, "Strike Price (K)",   "Call Price", "Call Price vs Strike Price")
        y3 = [europeanCallOption(S, K, t, r, σ) for t in maturities]
        optionCurveGraph(maturities,  y3, "Time to Maturity (T)","Call Price", "Call Price vs Maturity")
        y4 = [europeanCallOption(S, K, T, r, v) for v in volatilities]
        optionCurveGraph(volatilities, y4, "Volatility (σ)",     "Call Price", "Call Price vs Volatility")

    st.subheader("Implied Volatility Estimation")
    optionType = st.selectbox("Option Type", ["call", "put"])
    marketPrice = st.number_input("Market Price Option", value=0.0)

    if st.button("Estimate Implied Volatility"):
        result = impliedVolatility(optionType, marketPrice, S, K, T, r)
        if result:
            st.success(f"Estimated Implied Volatility:  {result}")
        else:
            st.error("No convergence On Implied Volatility!")

if __name__ == "__main__":
    #Main()
    #optionListData("optionsList.csv")
    #Main2()
    StreamlitInterface()