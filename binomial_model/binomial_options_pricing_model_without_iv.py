import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import minimize

# Binomial option pricing model
def binomial_option_pricing(S, K, T, r, sigma, N, option_type="call"):
    delta_t = T / N
    u = np.exp(sigma * np.sqrt(delta_t))
    d = 1 / u
    p = (np.exp(r * delta_t) - d) / (u - d)

    ST = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            ST[j, i] = S * (u ** (i - j)) * (d ** j)

    if option_type == "call":
        option_values = np.maximum(ST[:, N] - K, 0)
    elif option_type == "put":
        option_values = np.maximum(K - ST[:, N], 0)

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

# Objective function to minimize: difference between market price and model price
def objective_function(sigma, S, K, T, r, market_price, N, option_type):
    model_price = binomial_option_pricing(S, K, T, r, sigma, N, option_type)
    return (model_price - market_price) ** 2

# Streamlit app
st.title('Option Price Calculator Using Binomial Model with IV Estimation')

# Inputs
option_type = st.selectbox('Option Type', ('Call', 'Put'))
last_stock_price = st.number_input('Last Trading Day\'s Stock Price', value=129.900)
last_option_price = st.number_input('Last Trading Day\'s Option Price', value=2.5924)
premarket_change_percent = st.number_input('Premarket Change in Stock Price (%)', value=0.0, format="%.2f") / 100
K = st.number_input('Strike Price (K)', value=130.0)

# Expiry date input
expiry_date = st.date_input('Select Expiry Date', min_value=datetime.now().date(), value=datetime.now().date() + timedelta(days=7))

if expiry_date == datetime.now().date() and datetime.now().weekday() == 4:
    selected_date = datetime.now().date()
    st.write("Today is the expiry date and it's a Friday. No selection needed.")
else:
    selected_date = st.slider('Select Date', min_value=datetime.now().date(), max_value=expiry_date, value=datetime.now().date())

T = (expiry_date - selected_date).days / 365

r = st.number_input('Risk-free Interest Rate (r) in %', value=5.261, format="%.3f") / 100

expected_stock_price = last_stock_price * (1 + premarket_change_percent)

# Use a reasonable initial guess for IV
initial_guess = 0.3

if st.button('Calculate'):
    N = 100

    # Optimize to find the implied volatility
    result = minimize(
        objective_function, initial_guess, args=(expected_stock_price, K, T, r, last_option_price, N, option_type.lower()),
        bounds=[(0.01, 5.0)], method='L-BFGS-B'
    )
    implied_volatility = result.x[0]

    new_option_price = binomial_option_pricing(expected_stock_price, K, T, r, implied_volatility, N, option_type.lower())
    
    delta_S = 0.01
    price_up = binomial_option_pricing(expected_stock_price + delta_S, K, T, r, implied_volatility, N, option_type.lower())
    price_down = binomial_option_pricing(expected_stock_price - delta_S, K, T, r, implied_volatility, N, option_type.lower())
    delta = (price_up - price_down) / (2 * delta_S)
    
    st.write(f"The estimated {option_type} option price at market open is: {new_option_price:.4f}")
    st.write(f"Implied Volatility: {implied_volatility:.4f}")
    st.write(f"Calculated Delta: {delta:.4f}")
    st.write(f"Last Trading Day's Stock Price: {last_stock_price}")
    st.write(f"Expected Stock Price at Market Open: {expected_stock_price:.4f}")
    st.write(f"Change in Stock Price: {expected_stock_price - last_stock_price:.4f}")
    st.write(f"Time to Maturity (T): {T:.4f} years")
