import streamlit as st
import numpy as np
from datetime import datetime, timedelta

# Binomial option pricing model
def binomial_option_pricing(S, K, T, r, sigma, N, option_type="call"):
    # Calculate the necessary parameters
    delta_t = T / N
    u = np.exp(sigma * np.sqrt(delta_t))
    d = 1 / u
    p = (np.exp(r * delta_t) - d) / (u - d)
    
    # Initialize the asset prices at maturity
    ST = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            ST[j, i] = S * (u ** (i - j)) * (d ** j)
    
    # Initialize option values at maturity
    if option_type == "call":
        option_values = np.maximum(ST[:, N] - K, 0)
    elif option_type == "put":
        option_values = np.maximum(K - ST[:, N], 0)
    
    # Backward induction to get the option value at the initial node
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j] = np.exp(-r * delta_t) * (p * option_values[j] + (1 - p) * option_values[j + 1])
            if option_type == "call":
                option_values[j] = np.maximum(option_values[j], ST[j, i] - K)
            elif option_type == "put":
                option_values[j] = np.maximum(option_values[j], K - ST[j, i])
    
    return option_values[0]

# Calculate the number of days to expiry
def days_to_expiry(expiry_date, selected_date):
    return (expiry_date - selected_date).days / 365

# Streamlit app
st.title('Option Price Calculator Using Binomial Model')

# Inputs
option_type = st.selectbox('Option Type', ('Call', 'Put'))
last_stock_price = st.number_input('Last Trading Day\'s Stock Price', value=129.900)
premarket_change_percent = st.number_input('Premarket Change in Stock Price (%)', value=0.0, format="%.2f") / 100
K = st.number_input('Strike Price (K)', value=130.0)

# Expiry date input
expiry_date = st.date_input('Select Expiry Date', min_value=datetime.now().date(), value=datetime.now().date() + timedelta(days=7))

if expiry_date == datetime.now().date():
    selected_date = datetime.now().date()
    st.write("Today is the expiry date. No selection needed.")
else:
    selected_date = st.slider('Select Date', min_value=datetime.now().date(), max_value=expiry_date, value=datetime.now().date())

# Calculate the time to maturity based on the selected date
T = (expiry_date - selected_date).days / 365

r = st.number_input('Risk-free Interest Rate (r) in %', value=5.261, format="%.3f") / 100
sigma = st.number_input('Volatility (sigma) in %', value=39.53, format="%.2f") / 100

# Calculate changes
expected_stock_price = last_stock_price * (1 + premarket_change_percent)

# Current option price input
current_option_price = st.number_input('Current Option Price', value=0.0, format="%.4f")

# Calculate option price at market open using the Binomial Options Pricing Model
if st.button('Calculate'):
    # Number of steps in the binomial model
    N = 100  # You can adjust this value for more precision
    new_option_price = binomial_option_pricing(
        expected_stock_price, K, T, r, sigma, N, option_type.lower()
    )
    
    # Calculate the percentage change
    if current_option_price > 0:
        option_price_change_percent = ((new_option_price - current_option_price) / current_option_price) * 100
    else:
        option_price_change_percent = float('inf')
    
    st.write(f"Current Option Price: {current_option_price:.4f}")
    st.write(f"The estimated {option_type} option price at market open is: {new_option_price:.4f}")
    st.write(f"Percentage Change in Option Price: {option_price_change_percent:.2f}%")
    
    
    
    
    
    
