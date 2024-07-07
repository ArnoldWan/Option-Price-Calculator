import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

# Binomial option pricing model
def binomial_option_pricing(S, K, T, r, sigma, N, option_type="call"):

    '''
    S: current stock price
    K: strike price of option
    T: time to maturity in years
    r: risk-free rate
    sigma: volatility of underlying asset
    N: number of time steps
    delta_t: time interval per step
    u: up factor of stock price (in %)
    d: down factor of stock price (in %)
    p: risk neutral probability of up move
    ST: matrix of stock prices at each node
    option_values: array of option values at each node
    '''
    
    # Calculate the necessary parameters
    delta_t = T / N
    u = np.exp(sigma * np.sqrt(delta_t)) #
    d = 1 / u
    p = (np.exp(r * delta_t) - d) / (u - d)
    
    # create a (N+1) x (N+1) matrix to store price at each node
    ST = np.zeros((N + 1, N + 1)) #numpy array of dimension (N+1, N+1)
    for i in range(N + 1):
        for j in range(i + 1):
            ST[j, i] = S * (u ** (i - j)) * (d ** j)
    
    # Initialize option values at each node as a (N+1) x (N+1) matrix
    option_values = np.zeros((N + 1, N + 1))
    if option_type == "call":
        option_values[:, N] = np.maximum(ST[:, N] - K, 0)
    elif option_type == "put":
        option_values[:, N] = np.maximum(K - ST[:, N], 0)
    
    # Backward induction to get the option value at each node
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j, i] = np.exp(-r * delta_t) * (p * option_values[j, i+1] + (1-p) * option_values[j+1, i+1])
            if option_type == "call":
                option_values[j, i] = np.maximum(option_values[j, i], ST[j, i] - K)
            elif option_type == "put":
                option_values[j, i] = np.maximum(option_values[j, i], K - ST[j, i])
    
    return ST, option_values


def plot_binomial_tree(ST, option_values, filename='binomial_tree.png'):
    
    # Number of steps in the binomial tree
    N = ST.shape[0] - 1
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(N + 1): #each time step
        for j in range(i + 1): #each state/node at each time step
            x = i
            y = i - 2*j 
            ax.scatter(x, y, color='grey')
            #add stock price at the current node in blue
            ax.text(x, y, f'{ST[j, i]:.2f}', ha='center', va='bottom', fontsize=10, color='blue')
            ax.text(x, y-0.3, f'{option_values[j, i]:.2f}', ha='center', va='top', fontsize=10, color='red')
            
            if i < N:
                ax.plot([x, x+1], [y, y+1], 'k-', lw=0.5)
                ax.plot([x, x+1], [y, y-1], 'k-', lw=0.5)

    ax.set_xticks(range(N + 1))
    ax.set_xticklabels([f'Time {i}' for i in range(N + 1)])
    ax.set_yticks([])
    ax.set_ylabel('Stock Price / Option Value')
    ax.set_xlabel('Time Steps')
    plt.title('Binomial Tree for Option Pricing')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Calculate the number of days to expiry
def days_to_expiry(expiry_date, selected_date):
    return (expiry_date - selected_date).days / 365

# Streamlit app
st.title('Option Price Calculator Using Binomial Model')

# Inputs
option_type = st.selectbox('Option Type', ('Call', 'Put'))
last_stock_price = st.number_input('Last Trading Day\'s Stock Price', value=100.0)
premarket_change_percent = st.number_input('Premarket Change in Stock Price (%)', value=5.0, format="%.2f") / 100
K = st.number_input('Strike Price (K)', value=105.0)

# Expiry date input
expiry_date = st.date_input('Select Expiry Date', min_value=datetime.now().date(), value=datetime.now().date() + timedelta(days=7))

if expiry_date == datetime.now().date():
    selected_date = datetime.now().date()
    st.write("Today is the expiry date. No selection needed.")
else:
    selected_date = st.slider('Select Date', min_value=datetime.now().date(), max_value=expiry_date, value=datetime.now().date())

# Calculate the time to maturity based on the selected date
T = (expiry_date - selected_date).days / 365

r = st.number_input('Risk-free Interest Rate (r) in %', value=4.0, format="%.3f") / 100
sigma = st.number_input('Volatility (sigma) in %', value=40.0, format="%.2f") / 100

# Calculate changes
expected_stock_price = last_stock_price * (1 + premarket_change_percent)

# Current option price input
current_option_price = st.number_input('Current Option Price', value=1.0, format="%.4f")

# Calculate option price at market open using the Binomial Options Pricing Model
if st.button('Calculate'):
    # Number of steps in the binomial model
    N = 10  # You can adjust this value for more precision
    ST, option_prices = binomial_option_pricing(
        expected_stock_price, K, T, r, sigma, N, option_type.lower()
    )
    
    # Calculate the percentage change
    if current_option_price > 0:
        option_price_change_percent = ((option_prices[0][0] - current_option_price) / current_option_price) * 100
    else:
        option_price_change_percent = float('inf')
    
    st.write(f"Current Option Price: {current_option_price:.4f}")
    st.write(f"The estimated {option_type} option price at market open is: {option_prices[0][0]:.4f}")
    st.write(f"Percentage Change in Option Price: {option_price_change_percent:.2f}%")
    
    st.image('binomial_tree.png', caption='Binomial Tree for Option Pricing')
    
    
    
    
