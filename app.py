import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from scipy.ndimage import distance_transform_edt, uniform_filter, convolve

# =============================
# City & layers
# =============================
CLASSES = {0:'Residential', 1:'Commercial', 2:'Industrial'}
PALETTE = {0:(0.75,0.85,1.0), 1:(0.95,0.5,0.2), 2:(0.8,0.8,0.8)}

@dataclass
class City:
    grid: np.ndarray
    center: Tuple[int,int]
    roads: np.ndarray
    access: np.ndarray
    heat: np.ndarray
    ind_protect: np.ndarray

def make_city(H=50, W=50, seed=0) -> City:
    rng = np.random.default_rng(seed)
    grid = rng.integers(0,3,size=(H,W))
    roads = np.zeros((H,W), dtype=np.uint8)
    cx, cy = H//2, W//2
    roads[cx,:] = 1; roads[:,cy] = 1
    rr = min(H,W)//3
    for t in np.linspace(0,2*np.pi,400,endpoint=False):
        i = int(cx + rr*np.sin(t)); j=int(cy+rr*np.cos(t))
        if 0<=i<H and 0<=j<W: roads[i,j]=1
    dist = distance_transform_edt(1-roads)
    access = 1.0/(1.0+dist)
    access = (access - access.min())/(access.max()-access.min()+1e-9)
    heat = access.copy()
    ind = (grid==2).astype(float)
    ind_smooth = uniform_filter(ind, size=7)
    thr = np.quantile(ind_smooth, 0.75)
    ind_protect = (ind_smooth>=thr).astype(np.uint8)
    return City(grid=grid, center=(cx,cy), roads=roads, access=access, heat=heat, ind_protect=ind_protect)

def render_grid(grid: np.ndarray):
    H,W = grid.shape
    rgb = np.zeros((H,W,3), dtype=float)
    for k,c in PALETTE.items():
        rgb[grid==k]=c
    return rgb

# =============================
# Neighborhood helpers
# =============================
def neigh_share(grid: np.ndarray, target: int, r: int=1) -> np.ndarray:
    k = 2*r+1
    mask = (grid==target).astype(float)
    ker = np.ones((k,k),dtype=float)
    num = convolve(mask, ker, mode='nearest')
    den = convolve(np.ones_like(mask), ker, mode='nearest')
    num = num - mask; den = den - 1
    den = np.maximum(den,1.0)
    return num/den

# =============================
# Developer agents
# =============================
@dataclass
class DevParams:
    budget_per_round: float
    risk_temp: float
    aggressiveness: float
    coop_propensity: float
    scale_sensitivity: float
    conversion_aversion: float
    pref_R: float
    pref_C: float
    pref_I: float


@dataclass
class Developer:
    name: str
    params: DevParams
    profit: float = 0.0
    built: Dict[int,int] = field(default_factory=lambda: {0:0,1:0,2:0})
    coop_with: Dict[str,int] = field(default_factory=dict)
    comp_with: Dict[str,int] = field(default_factory=dict)

    def reset_stats(self):
        self.profit = 0.0
        self.built = {0:0,1:0,2:0}
        self.coop_with = {}
        self.comp_with = {}


DEFAULTS = {
    'Large': DevParams(180.0,0.2,1.35,0.65,0.6,0.3,0.9,1.1,1.0),
    'Medium':DevParams(110.0,0.35,1.15,0.55,0.45,0.35,1.0,1.0,1.0),
    'Small': DevParams(60.0,0.55,1.05,0.4,0.3,0.45,1.1,0.9,1.0),
}

BASE_PRICE = {0:1.0,1:1.6,2:1.2}
CONV_COST = {(0,0):0.1,(0,1):0.3,(0,2):0.25,(1,0):0.35,(1,1):0.1,(1,2):0.3,(2,0):0.3,(2,1):0.35,(2,2):0.1}
DIST_COST_WEIGHT = 0.05
NEIGH_WEIGHT = 0.6
DIVERSITY_WEIGHT = 0.25

# =============================
# Profit & policy
# =============================
POLICY = {'protect_industrial': True,'commercial_height_limit': False,'tod_incentive': True}
FEATURES = {'enable_coop': True,'endogenous_price': True}

def endogenous_price(city: City, to_k: int, alpha_demand=0.8, beta_heat=0.5):
    H,W = city.grid.shape
    counts = np.bincount(city.grid.ravel(), minlength=3)
    share = counts/(H*W)
    target = {0:0.34,1:0.33,2:0.33}[to_k]
    gap = target - share[to_k]
    mean_heat = city.heat.mean()
    return BASE_PRICE[to_k] * (1 + alpha_demand*gap) * (1 + beta_heat*mean_heat)

def policy_penalty(city: City, i:int, j:int, from_k:int, to_k:int):
    pen = 0.0
    if POLICY['protect_industrial'] and from_k==2 and to_k!=2 and city.ind_protect[i,j]==1:
        pen += 0.5
    if POLICY['commercial_height_limit'] and to_k==1:
        di = abs(i-city.center[0]); dj=abs(j-city.center[1]); d = np.hypot(di,dj)
        ring = min(city.grid.shape)//3
        if d>ring: pen += 0.3
    if POLICY['tod_incentive'] and to_k in (0,1):
        if city.roads[i,j]==1: pen -= 0.2
    return pen

def cell_profit(city: City, i:int, j:int, to_k:int, params: DevParams):
    from_k = city.grid[i,j]
    access = city.access[i,j]
    neigh_k = neigh_share(city.grid, to_k)[i,j]
    arr = np.array([neigh_share(city.grid,k)[i,j] for k in [0,1,2]])
    arr = np.clip(arr,1e-6,1.0)
    ent = -np.sum(arr*np.log(arr))/np.log(3)
    if FEATURES['endogenous_price']:
        p = endogenous_price(city, to_k)*(1+0.25*city.heat[i,j])
    else:
        p = BASE_PRICE[to_k]
    price_pref = {0: params.pref_R*p, 1: params.pref_C*p, 2: params.pref_I*p}[to_k]
    base_rev = price_pref * (0.6*access+0.4)
    conv_pen = params.conversion_aversion * CONV_COST[(from_k,to_k)]
    di = abs(i-city.center[0]); dj = abs(j-city.center[1]); d = np.hypot(di,dj)
    dist_pen = DIST_COST_WEIGHT*d
    pol_pen = policy_penalty(city,i,j,from_k,to_k)
    aggl = NEIGH_WEIGHT * neigh_k
    jacobs = DIVERSITY_WEIGHT * ent
    return base_rev + aggl + jacobs - conv_pen - dist_pen - pol_pen

# =============================
# Simulation
# =============================
def softmax_choice(values: np.ndarray, temp: float, k: int):
    x = values/ (temp if temp>1e-6 else 1e-6)
    x = x - x.max()
    p = np.exp(x)
    p = p / (p.sum()+1e-12)
    idx = np.arange(len(values))
    return np.random.choice(idx, size=min(k,len(values)), replace=False, p=p)

def adjacent(a:Tuple[int,int], b:Tuple[int,int]):
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))==1

@dataclass
class MarketOutcome:
    coop_edges: Dict[Tuple[str,str], int]
    comp_edges: Dict[Tuple[str,str], int]

def run_round(city: City, devs: List[Developer], rseed=0, parcels_per_dev=50):
    rng = np.random.default_rng(rseed)
    H,W = city.grid.shape
    candidates = [(i,j) for i in range(H) for j in range(W)]
    N = len(candidates)
    access_flat = city.access.ravel()
    access_probs = access_flat/(access_flat.sum()+1e-12)
    proposals = []
    for d_idx, dev in enumerate(devs):
        budget = dev.params.budget_per_round
        kcand = min(parcels_per_dev*4, N)
        cand_ids = rng.choice(np.arange(N), size=kcand, replace=False, p=access_probs)
        cand = [candidates[cid] for cid in cand_ids]
        vals, choices = [], []
        for (i,j) in cand:
            per_k = [cell_profit(city,i,j,k,dev.params) for k in [0,1,2]]
            k_best = int(np.argmax(per_k)); v = float(per_k[k_best])
            bid = min(v*dev.params.aggressiveness, budget)
            if bid>0:
                vals.append(v)
                choices.append((i,j,k_best,bid))
        if len(vals)==0: continue
        sel_ids = softmax_choice(np.array(vals), dev.params.risk_temp, k=parcels_per_dev)
        for sid in sel_ids:
            (i,j,k_best,bid) = choices[sid]
            if budget<=0: break
            spend = min(bid, budget)
            budget -= spend
            proposals.append((d_idx,(i,j),k_best,spend))
    per_cell = {}
    for d_idx,(i,j),k,b in proposals:
        per_cell.setdefault((i,j), []).append((d_idx,k,b))
    coop_edges = {('Large','Medium'):0,('Large','Small'):0,('Medium','Small'):0}
    comp_edges = {('Large','Medium'):0,('Large','Small'):0,('Medium','Small'):0}
    heat_add = np.zeros_like(city.heat)
    for (i,j), lst in per_cell.items():
        if len(lst)==1:
            d_idx,k,bid = lst[0]
            city.grid[i,j] = k
            devs[d_idx].profit += bid
            devs[d_idx].built[k] = devs[d_idx].built.get(k,0)+1
            heat_add[i,j]=0.05
        else:
            lst.sort(key=lambda x:x[2], reverse=True)
            d_idx,k,bid = lst[0]
            city.grid[i,j]=k
            devs[d_idx].profit+=bid
            devs[d_idx].built[k]=devs[d_idx].built.get(k,0)+1
            heat_add[i,j]=0.05
            for o_idx, ok, ob in lst[1:]:
                pair = tuple(sorted((devs[d_idx].name, devs[o_idx].name)))
                if FEATURES['enable_coop']:
                    coop_edges[pair]=coop_edges.get(pair,0)+1
                else:
                    comp_edges[pair]=comp_edges.get(pair,0)+1
    city.heat = np.clip(city.heat + heat_add, 0.0, 1.0)
    return MarketOutcome(coop_edges, comp_edges)

def run_sim(H,W,rounds,parcels_per_dev,seed=0):
    city = make_city(H,W,seed)
    devs = [Developer(name, DEFAULTS[name]) for name in ['Large','Medium','Small']]
    hist_shares=[]
    for r in range(rounds):
        run_round(city, devs, rseed=seed+r, parcels_per_dev=parcels_per_dev)
        counts = np.bincount(city.grid.ravel(), minlength=3)
        shares = counts / city.grid.size
        hist_shares.append(shares)
    hist_shares = np.array(hist_shares)
    return city, devs, hist_shares

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="ABM City Simulation", layout="wide")
st.title("Agent-Based City Simulation")

# -------- Sidebar Controls --------
with st.sidebar:
    st.header("City & Simulation Parameters")
    H = st.slider("Rows (H)", 20, 120, 50)
    W = st.slider("Cols (W)", 20, 120, 50)
    seed = st.number_input("Random Seed", 0, 9999, 0)
    rounds = st.slider("Rounds", 1, 60, 10)
    parcels_per_dev = st.slider("Parcels per developer per round", 10, 200, 50)

    st.header("Developer Settings")
    enable_coop = st.checkbox("Enable JV cooperation", value=True)
    enable_endo = st.checkbox("Enable endogenous prices", value=True)
    POLICY['protect_industrial'] = st.checkbox("Protect industrial zones", value=True)
    POLICY['commercial_height_limit'] = st.checkbox("Commercial height limit", value=False)
    POLICY['tod_incentive'] = st.checkbox("TOD incentives on roads", value=True)

    if st.button("Run Simulation"):
        st.session_state.run_sim = True

# -------- Main Panel --------
if 'run_sim' in st.session_state and st.session_state.run_sim:
    city, devs, hist_shares = run_sim(H,W,rounds,parcels_per_dev,seed)
    
    st.subheader("City Map")
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(render_grid(city.grid))
    ax.set_xticks([]); ax.set_yticks([])
    st.pyplot(fig, use_container_width=True)

    st.subheader("Class Shares Over Time")
    fig2, ax2 = plt.subplots(figsize=(12,6))
    for k,name in CLASSES.items():
        ax2.plot(hist_shares[:,k], label=name)
    ax2.set_xlabel("Step"); ax2.set_ylabel("Share"); ax2.set_ylim(0,1)
    ax2.legend()
    st.pyplot(fig2, use_container_width=True)

    st.subheader("Developer Profit")
    prof_df = pd.DataFrame({d.name: d.profit for d in devs}, index=["Profit"])
    st.table(prof_df)
