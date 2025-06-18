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

Option Greeks:
- Delta(Δ) = measures price sensitiviy to underlying, call option => Δ ranges from 0 to 1, put option => Δ ranges from -1 to 0
- Gamma(Γ) = measure rate of change of delta
- Vega(ν) = measures sensitivity to volatilty
- Theta(Θ) = measures sensitivity to time decay
- Rho(ρ) = measures sensitivity to interest rates

Formala for Option Greeks:
- Δ = N(d1) for calls, Δ = N(d1)-1 for puts, where N() is the standard normal cumulative distribution function
- Γ = (N'(d1))/(Sσ srt(T)) where N'(d1) is the standard normal probability density function
- ν = S srt(T) N'(d1)


Tasks:
- Implement basic Black Scholes Algotithm
- Read options list from a csv file and ouput the call and put option price using the black scholes algorithm
- Implement implied volatility where we reverse the black sholes algo given a market option price find out what the implied volatility value is
- Visualise Option Prices using matplotlib such as option price vs stock price, option price vs strike price, option price vs volatility, option price vs time to maturity
- Implement simple Steamlit app that shows these results
- Add real time market data integration using Yahoo Finance to retrieve real-world stock data 
- Add stock option prices heatmap visualisation
"""

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Code:
import math
from scipy.stats import norm
import csv
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import plotly.graph_objs as go
import pandas as pd
import sqlite3
from datetime import datetime

st.set_page_config(
    page_title="Black-Scholes Option Pricing Model Algorithm",
    page_icon="📈",
    initial_sidebar_state="expanded"
)
st.title("Black-Scholes Option Pricing Model Algorithm 📈")
st.caption("*Using Yahoo Finance Stock Market Data API")

def initialiseDB():
    connection = sqlite3.connect("stockOptions.db")
    c = connection.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS queriedOptions (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            [timestamp] TEXT,
            stockTicker TEXT,
            OptionType TEXT,
            S REAL,
            K REAL,
            T REAL,
            r REAL,
            σ, REAL,
            optionPrice REAL
        )
        """)
    connection.commit()
    connection.close()

def europeanCallOption(S, K, T, r, σ):
    if T <= 0:
        return max(S - K, 0)
    if σ <= 0:
        return max(S - K * math.exp(-r * T), 0)
    epsilon = 1e-8
    denom = σ * math.sqrt(T) + epsilon 
    d1 = (math.log(S / K) + (r + 0.5 * σ ** 2) * T) / denom
    d2 = d1 - denom
    callPrice = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return callPrice

def europeanPutOption(S, K, T, r, σ):
    if T <= 0:
        return max(K - S, 0)
    if σ <= 0:
        return max(K * math.exp(-r * T) - S, 0)
    epsilon = 1e-8
    denom = σ * math.sqrt(T) + epsilon 
    d1 = (math.log(S / K) + (r + 0.5 * σ ** 2) * T) / denom
    d2 = d1 - denom
    putPrice = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
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
    epsilon = 1e-8
    denom = σ * math.sqrt(T) + epsilon
    d1 = (math.log(S/K) + (r+0.5*σ**2) * T) / denom
    return S * norm.pdf(d1) * math.sqrt(T)

def impliedVolatility(optionType, marketValue, S, K, T, r, tolerance=1e-6, maxIterationsNum=100):
    σ = 0.1  
    iteration = 0
    while iteration < maxIterationsNum:
        if optionType == "call":
            value = europeanCallOption(S, K, T, r, σ)
        else:
            value = europeanPutOption(S, K, T, r, σ)
        vega = Vega(S, K, T, r, σ)
        difference = value - marketValue
        if abs(difference) < tolerance:
            return σ
        if vega == 0:
            break
        σ -= difference / vega
        iteration += 1
    return None #when there is no converges

def optionsVega(S, K, T, r, σ):
    d1 = (math.log(S/K) + (r + 0.5 * σ ** 2) * T) / (σ * math.sqrt(T))
    return S * norm.pdf(d1) * math.sqrt(T)

def visualiseOptionStock(S, K, T, r, σ, optionType):
    x = list(range(50, 150))
    y = []
    for s in x:
        if optionType == "call":
            value = europeanCallOption(s, K, T, r, σ)
        else:
            value = europeanPutOption(s, K, T, r, σ)
        y.append(value)
    plt.plot(x, y, color="grey")
    plt.title(f"{optionType} Option Price vs Stock Price")
    plt.xlabel("Stock Price")
    plt.ylabel("Option Price")
    plt.grid(True)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close()

def visualiseOptionStrike(S, K, T, r, σ, optionType):
    x = list(range(50, 150))
    y = []
    for k in x:
        if optionType == "call":
            value = europeanCallOption(S, k, T, r, σ)
        else:
            value = europeanPutOption(S, k, T, r, σ)
        y.append(value)
    plt.plot(x, y, color="grey")
    plt.title(f"{optionType} Option Price vs Strike")
    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.grid(True)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close()

def visualiseOptionMaturity(S, K, r, σ, optionType):
    x = [i/10 for i in range(1, 50)]
    y = []
    for t in x:
        if optionType == "call":
            value = europeanCallOption(S, K, t, r, σ)
        else:
            value = europeanPutOption(S, K, t, r, σ)
        y.append(value)
    plt.plot(x, y, color="grey")
    plt.title(f"{optionType} Option Price vs Time To Maturity")
    plt.xlabel("Time To Maturity")
    plt.ylabel("Option Price")
    plt.grid(True)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close()


def visualiseOptionVolatility(S, K, T, r, σ, optionType):
    x = [i/10 for i in range(1, 50)]
    y = []
    for σ in x:
        if optionType == "call":
            value = europeanCallOption(S, K, T, r, σ)
        else:
            value = europeanPutOption(S, K, T, r, σ)
        y.append(value)
    plt.plot(x, y, color="grey")
    plt.title(f"{optionType} Option Price vs Volatility")
    plt.xlabel("Volatility")
    plt.ylabel("Option Price")
    plt.grid(True)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close()

def optionCurveGraph(x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    ax.plot(x, y, color="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    st.pyplot(fig)

def previousVolatilities(stockTicker, period='1y'):
    stockData = yf.Ticker(stockTicker).history(period=period)['Close'].pct_change()
    numberOfDays = np.sqrt(252)
    volatility = np.std(stockData) * numberOfDays
    return volatility

def riskFreeInterestYield():
    try:
        tnx = yf.Ticker("^TNX")
        riskFreeData = tnx.history(period='1d')
        tenYearYield = riskFreeData['Close'].iloc[-1]/100
        return tenYearYield
    except Exception as e:
        st.error(f"Unable To Retrieve Risk-Free Rate!")
        return 0.05
    
def Δ(optionType, S, K, T, r, σ):
    if σ <= 0 or T <= 0:
        return 0.0
    d1 = (math.log(S/K) + (r+0.5*σ**2)*T)/(σ*(math.sqrt(T)))
    if optionType == "call":
        return norm.cdf(d1)
    else:
        return (norm.cdf(d1) - 1)
    
def Γ(S, K, T, r, σ):
    if σ <= 0 or T <= 0:
        return 0.0
    d1 = (math.log(S/K) + (r*0.5+σ**2)*T)/(σ*(math.sqrt(T)))
    return ((norm.pdf(d1))/(S*σ*math.sqrt(T)))

def ν(S, K, T, r, σ):
    if σ <= 0 or T <= 0:
        return 0.0
    d1 = (math.log(S/K) + (r+0.5*σ**2)*T)/(σ*math.sqrt(T))
    return (norm.pdf(d1))/(S*σ*math.sqrt(T))

def Θ(optionType, S, K, T, r, σ):
    if σ <= 0 or T <= 0:
        return 0.0
    d1 = (math.log(S/K) + (r+0.5*σ**2)*T)/(σ*math.sqrt(T))
    d2 = d1 - σ*math.sqrt(T)
    firstTerm = -(S*norm.pdf(d1)*σ) / (2*math.sqrt(T))
    if optionType == "call":
        secondTerm = r*K*math.exp(-r*T)*norm.cdf(d2)
        return (firstTerm - secondTerm)
    else:
        secondTerm = r*K*math.exp(-r*T)*norm.cdf(-d2)
        return (firstTerm + secondTerm)

def ρ(optionType, S, K, T, r, σ):
    if σ <= 0 or T <= 0:
        return 0.0
    d1 = (math.log(S/K) + (r+0.5*σ**2)*T)/(σ*math.sqrt(T))
    d2 = d1 - σ*math.sqrt(T)
    if optionType == "call":
        return (K*T*math.exp(-r*T)*norm.cdf(d2))
    else:
        return (-K*T*math.exp(-r*T)*norm.cdf(-d2))
    
def StreamlitInterface():
    initialiseDB()
    plt.style.use('dark_background')
    stockTicker = st.text_input("Enter Stock Ticker Symbol: ", key='stockTicker')
    if 'currentS' not in st.session_state:
        st.session_state['currentS'] = 100.0
    if 'currentSigma' not in st.session_state:
        st.session_state['currentSigma'] = 0.2
    if 'currentR' not in st.session_state:
        st.session_state['currentR'] = riskFreeInterestYield()
    if st.button("Load Current Stock Market Data", key='loadData'):
        if stockTicker:
            try:
                stockData = yf.Ticker(stockTicker)
                stockPrevious = stockData.history(period='1d')
                if len(stockPrevious) > 0:
                    st.session_state['currentS'] = stockPrevious['Close'].iloc[-1]
                    st.session_state['currentSigma'] = previousVolatilities(stockTicker)
                    st.success(f"Current Market Price: {st.session_state['currentS']:.2f}")
                    st.success(f"Risk-free Interest Rate (10Y Yield): {st.session_state['currentR']*100:.2f}%")
                    st.success(f"Previous Volatility Estimation: {st.session_state['currentSigma']:.4f}")
                else:
                    st.error("No Market Data Found!")
            except Exception as e:
                st.error(f"Error Loading Data: {e}")
        else:
            st.error("Enter Stock Ticker!")
    
    S = st.number_input("Enter Current Stock Price (S): ", min_value=0.01, value=st.session_state['currentS'], key='currentS')
    σ = st.number_input("Enter Volatility (σ): ", min_value=0.001, value=st.session_state['currentSigma'], key='currentSigma')
    K = st.number_input("Enter Strike Price (K): ", min_value=0.01, value=S, key='currentK')
    T = st.number_input("Enter Time To Maturity (yrs): ", min_value=0.001, value=0.5, key='currentT')
    r = st.number_input("Enter Risk-free Interest-Rate (r): ", min_value=0.0, value=st.session_state['currentR'], format="%.6f", key='currentR')

    callPrice = europeanCallOption(S, K, T, r, σ)
    putPrice = europeanPutOption(S, K, T, r, σ)
    st.success(f"Call Option Price: {callPrice:.4f}")
    st.success(f"Put Option Price: {putPrice:.4f}")

    if st.button("Calculate & Visualise"):
        callPrice = europeanCallOption(st.session_state['currentS'], K, T, r, σ)
        putPrice = europeanPutOption(st.session_state['currentS'], K, T, r, σ)
        st.success(f"Call Option Price: {callPrice:.4f}")
        st.success(f"Put Option Price: {putPrice:.4f}")
        timeStamp = datetime.now().isoformat()
        S = st.session_state['currentS']
        connection = sqlite3.connect("stockOptions.db")
        c = connection.cursor()
        c.execute(
            """
            INSERT INTO queriedOptions (timeStamp, stockTicker, optionType, S, K, T, r, σ, optionPrice)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
        (timeStamp, stockTicker, "call", S, K, T, r, σ, callPrice))
        c.execute(
            """
            INSERT INTO queriedOptions (timeStamp, stockTicker, optionType, S, K, T, r, σ, optionPrice)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
        (timeStamp, stockTicker, "put", S, K, T, r, σ, putPrice))
        connection.commit()
        connection.close()
        optionType = st.selectbox("Option Type", ["call", "put"], key="optionTypevisualise")
        visualiseOptionStock(S, K, T, r, σ, optionType)
        visualiseOptionStrike(S, K, T, r, σ, optionType)
        visualiseOptionMaturity(S, K, r, σ, optionType)
        visualiseOptionVolatility(S, K, T, r, σ, optionType)

    st.subheader("Option Greeks")
    greeksOptionType = st.selectbox("Select Option Type For Greeks: ", ["call", "put"])
    greeksOptionData = {"Option Greeks": ["Delta(Δ)", "Gamma(Γ)", "Vega(ν)", "Theta(Θ)", "Rho(ρ)"],"Value": [Δ(greeksOptionType, S, K, T, r, σ),Γ(S, K, T, r, σ),ν(S, K, T, r, σ),Θ(greeksOptionType, S, K, T, r, σ),ρ(greeksOptionType, S, K, T, r, σ),]}
    dfGreeks = pd.DataFrame(greeksOptionData)
    dfGreeks["Value"] = dfGreeks["Value"].apply(lambda x: f"{x:.6f}")
    st.table(dfGreeks)

    st.subheader("Stock Option Prices Heatmap")
    if st.button("Visualise Heatmap"):
        sValues = np.linspace(S*0.5,S*1.5,50)
        sigmaValues = np.linspace(0.05, 0.5, 50)
        sGrid, sigmaGrid = np.meshgrid(sValues, sigmaValues)
        stockPrices = np.vectorize(europeanCallOption)(sGrid, K, T, r, sigmaGrid)
        fig = go.Figure(data=go.Heatmap(z=stockPrices,x=sValues,y=sigmaValues,colorbar=dict(title='Option Price'),colorscale='RdYlGn', text=np.round(stockPrices,2), texttemplate='%{text}', hovertemplate='Stock: %{x}<br>Volatility: %{y}<br>Price: %{z}<extra></extra>',showscale=True))
        fig.update_layout(
        title=f'Stock Option Prices Heatmap (centered at Current Stock ${S})', xaxis_title='Stock price (S)', yaxis_title='Volatility (σ)')
        st.plotly_chart(fig)

    st.subheader("Implied Volatility Estimation")
    optionType = st.selectbox("Option Type", ["call", "put"], key="optionTypeImpliedVolatility")
    marketPrice = st.number_input("Current Market Option Price", min_value=0.0, value=0.0)

    if st.button("Estimate Implied Volatility"):
        result = impliedVolatility(optionType, marketPrice, S, K, T, r)
        if result:
            st.success(f"Estimated Implied Volatility: {result:.6f}")
        else:
            st.error("No Convergence On Implied Volatility!")

    if st.button("Export Queries"):
        connection = sqlite3.connect("stockOptions.db")
        df = pd.read_sql_query("SELECT * FROM queriedOptions", connection)
        connection.close()
        csvFile = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Queries As CSV Format", data=csvFile, file_name="queriedOptions.csv", mime='text/csv') 

if __name__ == "__main__":
    #Main()
    #optionListData("optionsList.csv")
    #Main2()
    StreamlitInterface()