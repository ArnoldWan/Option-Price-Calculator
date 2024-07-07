import streamlit as st
from scipy.stats import norm
import numpy as np
from datetime import datetime, timedelta

# Black-Scholes option pricing model
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
    elif option_type == "put":
        price = (K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - S * norm.cdf(-d1, 0.0, 1.0))
    return price

# Calculate the number of days to expiry
def days_to_expiry(expiry_date, selected_date):
    return (expiry_date - selected_date).days / 365

# Streamlit app
st.title('Black Schole European Option Price Calculator')

# Inputs
option_type = st.selectbox('Option Type', ('Call', 'Put'))
last_stock_price = st.number_input('Last Trading Day\'s Stock Price', value=129.900)
last_option_price = st.number_input('Last Trading Day\'s Option Price', value=2.5924)
premarket_change_percent = st.number_input('Premarket Change in Stock Price (%)', value=0.0) / 100
K = st.number_input('Strike Price (K)', value=130.0)

# Expiry date input
expiry_date = st.date_input('Select Expiry Date', min_value=datetime.now().date(), value=datetime.now().date() + timedelta(days=7))

if expiry_date > datetime.now().date():
    # Slider to select date between now and expiry date
    selected_date = st.slider('Select Date', min_value=datetime.now().date(), max_value=expiry_date, value=datetime.now().date())
else:
    selected_date = datetime.now().date()

# Calculate the time to maturity based on the selected date
T = days_to_expiry(expiry_date, selected_date)

r = st.number_input('Risk-free Interest Rate (r) in %', value=5.261) / 100
sigma = st.number_input('Volatility (sigma) in %', value=39.53) / 100

# Calculate expected stock price at market open
expected_stock_price = last_stock_price * (1 + premarket_change_percent)

# Calculate option price at market open
if st.button('Calculate'):
    option_price_open = black_scholes(expected_stock_price, K, T, r, sigma, option_type.lower())
    st.write(f"The expected {option_type} option price at market open is: {option_price_open:.4f}")
    st.write(f"Last Trading Day's Stock Price: {last_stock_price}")
    st.write(f"Last Trading Day's Option Price: {last_option_price}")
    st.write(f"Expected Stock Price at Market Open: {expected_stock_price:.4f}")
    st.write(f"Time to Maturity (T): {T:.4f} years")

# Run the app with: streamlit run app.py
