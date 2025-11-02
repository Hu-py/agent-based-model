# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.ndimage import distance_transform_edt, uniform_filter, convolve

# =============================
# City setup (same as your Jupyter code)
# =============================
CLASSES = {0:'Residential', 1:'Commercial', 2:'Industrial'}
PALETTE = {0:(0.75,0.85,1.0), 1:(0.95,0.5,0.2), 2:(0.8,0.8,0.8)}

@dataclass
class City:
    grid: np.ndarray
    center: tuple
    roads: np.ndarray
    access: np.ndarray
    heat: np.ndarray
    ind_protect: np.ndarray

# ... include make_city, render_grid, neigh_share, DevParams, Developer, cell_profit, run_round, run_sim ...
# (全部函数可以直接复用，去掉 ipywidgets 部分)

# =============================
# Streamlit UI
# =============================

st.title("ABM: Three Developers on a Grid City (Streamlit Version)")

# City parameters
H = st.slider("Rows", min_value=20, max_value=120, value=50)
W = st.slider("Cols", min_value=20, max_value=120, value=50)
seed = st.slider("Random seed", 0, 9999, 0)
rounds = st.slider("Rounds", 1, 60, 10)
ppd = st.slider("Parcels/dev/round", 10, 200, 50)

# Feature toggles
st.subheader("Features & Policies")
enable_coop = st.checkbox("Enable JV cooperation", True)
endogenous_price = st.checkbox("Enable endogenous prices", True)
protect_industrial = st.checkbox("Protect industrial zones", True)
commercial_height_limit = st.checkbox("Commercial height limit", False)
tod_incentive = st.checkbox("TOD incentives on roads", True)

# Developer parameters (example for Large, repeat for Medium/Small)
st.subheader("Large developer parameters")
L_budget = st.slider("Budget", 20, 300, 180)
L_temp   = st.slider("Risk Temp", 0.05, 1.0, 0.20)
L_aggr   = st.slider("Aggressiveness", 0.8, 2.0, 1.35)
L_coop   = st.slider("Cooperation propensity", 0.0, 1.0, 0.65)
L_scale  = st.slider("Scale sensitivity", 0.0, 1.5, 0.6)
L_conv   = st.slider("Conversion aversion", 0.0, 1.0, 0.3)
L_prefR  = st.slider("Pref Residential", 0.5, 1.5, 0.9)
L_prefC  = st.slider("Pref Commercial", 0.5, 1.5, 1.1)
L_prefI  = st.slider("Pref Industrial", 0.5, 1.5, 1.0)

# Build city and developers
if 'city' not in st.session_state:
    st.session_state.city = make_city(H, W, seed)

if st.button("Reset city"):
    st.session_state.city = make_city(H, W, seed)

city = st.session_state.city

# Display initial city
st.subheader("Initial city")
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(render_grid(city.grid))
ax.axis('off')
st.pyplot(fig)

# Run simulation
if st.button("Run simulation"):
    devs = [
        Developer("Large", DevParams(L_budget, L_temp, L_aggr, L_coop, L_scale, L_conv, L_prefR, L_prefC, L_prefI)),
        # Add Medium and Small similarly
    ]
    shares_ts, coop_sum, comp_sum = run_sim(city, devs, rounds=rounds, seed=seed, parcels_per_dev=ppd)
    
    # Plot city after simulation
    st.subheader("City after simulation")
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.imshow(render_grid(city.grid))
    ax2.axis('off')
    st.pyplot(fig2)
    
    # Plot time series
    st.subheader("Land-use class shares over rounds")
    fig3, ax3 = plt.subplots(figsize=(7,4))
    t = np.arange(shares_ts.shape[0])
    for k,name in CLASSES.items():
        ax3.plot(t, shares_ts[:,k], label=name)
    ax3.set_ylim(0,1); ax3.set_xlabel("Round"); ax3.set_ylabel("Class share")
    ax3.legend(); plt.tight_layout()
    st.pyplot(fig3)
    
    # Developer KPIs
    st.subheader("Developer KPIs")
    rows=[]
    for d in devs:
        rows.append(dict(Developer=d.name, Profit=round(d.profit,2),
                         Built_R=d.built.get(0,0), Built_C=d.built.get(1,0), Built_I=d.built.get(2,0)))
    df = pd.DataFrame(rows)
    st.dataframe(df)

    # Cooperation/Competition
    st.subheader("Cooperation counts")
    st.write(coop_sum)
    st.subheader("Competition counts")
    st.write(comp_sum)
