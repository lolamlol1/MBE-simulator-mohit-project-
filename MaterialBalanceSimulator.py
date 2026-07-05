import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
import datetime

# --- Fluid Property Functions ---

def calculate_Bw(p, t):
    """Calculate water formation volume factor using McCain's correlation."""
    delta_VwP = (-1.95301e-9 * p * t - 1.72834e-13 * (p**2) * t - 3.58922e-7 * p - 2.25341e-10 * (p**2))
    delta_VwT = (-1.0001e-2 + 1.33391e-4 * t + 5.50654e-7 * (t**2))
    return (1 + delta_VwP) * (1 + delta_VwT)

def compute_z(Sg, p_val, t):
    """Compute Z-factor using Newton-Raphson method."""
    T_pc = 120.1 + 425 * Sg - 62.9 * (Sg ** 2)
    P_pc = 671.1 + 14 * Sg - 34.3 * (Sg ** 2)
    T_pr = (t + 460) / T_pc
    P_pr = p_val / P_pc
    Z = 1.0
    tol = 1e-6
    max_iter = 100
    for _ in range(max_iter):
        rho_r = 0.27 * P_pr / (Z * T_pr)
        term1 = (0.3265 - 1.07/T_pr - 0.5339/(T_pr**3) + 0.01569/(T_pr**4) - 0.05165/(T_pr**5)) * rho_r
        term2 = (0.5475 - 0.7361/T_pr + 0.1844/(T_pr**2)) * (rho_r**2)
        term3 = -0.1056 * ((-0.7361/T_pr) + (0.1844/(T_pr**2))) * (rho_r**5)
        term4 = 0.6134 * (1 + 0.7210*(rho_r**2)) * ((rho_r**2)/(T_pr**3)) * np.exp(-0.7210*(rho_r**2))
        F_val = 1 + term1 + term2 + term3 + term4
        f_Z = Z - F_val
        delta = 1e-6
        Z_plus = Z + delta
        rho_r_plus = 0.27 * P_pr / (Z_plus * T_pr)
        term1_plus = (0.3265 - 1.07/T_pr - 0.5339/(T_pr**3) + 0.01569/(T_pr**4) - 0.05165/(T_pr**5)) * rho_r_plus
        term2_plus = (0.5475 - 0.7361/T_pr + 0.1844/(T_pr**2)) * (rho_r_plus**2)
        term3_plus = -0.1056 * ((-0.7361/T_pr) + (0.1844/(T_pr**2))) * (rho_r_plus**5)
        term4_plus = 0.6134 * (1 + 0.7210*(rho_r_plus**2)) * ((rho_r_plus**2)/(T_pr**3)) * np.exp(-0.7210*(rho_r_plus**2))
        F_plus = 1 + term1_plus + term2_plus + term3_plus + term4_plus
        f_Z_plus = Z_plus - F_plus
        Z_minus = Z - delta
        rho_r_minus = 0.27 * P_pr / (Z_minus * T_pr)
        term1_minus = (0.3265 - 1.07/T_pr - 0.5339/(T_pr**3) + 0.01569/(T_pr**4) - 0.05165/(T_pr**5)) * rho_r_minus
        term2_minus = (0.5475 - 0.7361/T_pr + 0.1844/(T_pr**2)) * (rho_r_minus**2)
        term3_minus = -0.1056 * ((-0.7361/T_pr) + (0.1844/(T_pr**2))) * (rho_r_minus**5)
        term4_minus = 0.6134 * (1 + 0.7210*(rho_r_minus**2)) * ((rho_r_minus**2)/(T_pr**3)) * np.exp(-0.7210*(rho_r_minus**2))
        F_minus = 1 + term1_minus + term2_minus + term3_minus + term4_minus
        f_Z_minus = Z_minus - F_minus
        derivative = (f_Z_plus - f_Z_minus) / (2 * delta)
        if abs(derivative) < 1e-12:
            break
        Z_new = Z - f_Z / derivative
        if abs(Z_new - Z) < tol:
            Z = Z_new
            break
        Z = Z_new
    return Z

def calculate_gas_viscosity(p, t, sg_g, Z):
    """Calculate gas viscosity using Lee et al. correlation."""
    M = 28.97 * sg_g
    rho_g = (28.97 * sg_g * p) / (Z * 10.73 * (t + 460))
    K = ((9.4 + 0.02 * M) * ((t + 460)**1.5)) / (209 + 19 * M + t + 460)
    X = 3.5 + (986 / (t + 460)) + 0.01 * M
    Y = 2.4 - 0.2 * X
    return 1e-4 * K * np.exp(X * ((rho_g / 62.4)**Y))

# Oil-specific functions
def calculate_Bo(p, p_b, r_s_input, sg_g, api, t):
    sg_o = 141.5 / (api + 131.5)
    if p <= p_b:
        Rs_p = calculate_Rs(p, p_b, r_s_input, sg_g, api, t)
        return 0.9759 + 0.00012 * (((Rs_p * (sg_g / sg_o) ** 0.5) + (1.25 * t)) ** 1.2)
    else:
        rs_bp = sg_g * ((p_b * (10 ** (0.0125 * api))) / (18 * (10 ** (0.00091 * t)))) ** 1.2048
        bo_b = 0.9759 + 0.00012 * (((rs_bp * (sg_g / sg_o) ** 0.5) + (1.25 * t)) ** 1.2)
        c_o = ((5 * rs_bp) + (17.2 * t) - (1180 * sg_g) + (12.61 * api) - 1433) / (p * 1e5)
        return bo_b * np.exp(-c_o * (p - p_b))

def calculate_Rs(p, p_b, r_s_input, sg_g, api, t):
    if p > p_b:
        return r_s_input
    else:
        return sg_g * (((p + 14.7) * (10 ** (0.0125 * api)) / (18 * (10 ** (0.00091 * t)))) ** 1.2048)

def calculate_Bg(p, sg_g, t):
    Z_val = compute_z(sg_g, p, t)
    return 0.005035 * Z_val * (t + 460) / p

def calculate_viscosity_oil(p, p_b, r_s_input, api, t, sg_g):
    v_od = 10 ** (10 ** (3.0324 - 0.02023 * api) * t ** (-1.163)) - 1
    if p <= p_b:
        Rs_val = calculate_Rs(p, p_b, r_s_input, sg_g, api, t)
        return (10.715 * (Rs_val + 100) ** (-0.515)) * v_od ** (5.44 * (Rs_val + 150) ** (-0.338))
    else:
        rs_bp = sg_g * ((p_b * (10 ** (0.0125 * api))) / (18 * (10 ** (0.00091 * t)))) ** 1.2048
        v_o_sat = (10.715 * (rs_bp + 100) ** (-0.515)) * v_od ** (5.44 * (r_s_input + 150) ** (-0.338))
        return v_o_sat + 0.001 * (p - p_b) * ((0.024 * v_o_sat ** 1.6) + (0.038 * v_o_sat ** 0.56))

# --- Main Application ---

def main():
    # ── Page config (must be first Streamlit command) ───────────────────
    st.set_page_config(
        page_title="MBE Simulator | Petroleum Engineering",
        page_icon="⛽",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ── Global matplotlib dark professional theme ────────────────────────
    plt.rcParams.update({
        'figure.facecolor': '#0d1b2a',
        'axes.facecolor':   '#091525',
        'axes.edgecolor':   '#1e3557',
        'axes.labelcolor':  '#8ba7cc',
        'text.color':       '#d4e0f0',
        'xtick.color':      '#6b8bb5',
        'ytick.color':      '#6b8bb5',
        'grid.color':       '#162a45',
        'grid.alpha':       0.65,
        'grid.linestyle':   '--',
        'legend.facecolor': '#0d1b2a',
        'legend.edgecolor': '#1e3557',
        'legend.labelcolor':'#8ba7cc',
        'axes.titlecolor':  '#e2eaf5',
        'axes.titlesize':   13,
        'axes.titleweight': 'bold',
        'axes.labelsize':   11,
        'figure.dpi':       120,
        'lines.linewidth':  2.2,
        'lines.markersize': 7,
    })

    # ── Custom CSS ───────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

    /* Main background */
    .stApp {
        background: linear-gradient(160deg, #060c18 0%, #091524 40%, #0c1a2e 100%);
        min-height: 100vh;
    }
    .main .block-container { padding-top: 0.5rem; max-width: 1400px; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07101e 0%, #0b1929 100%) !important;
        border-right: 1px solid #162d50;
    }
    [data-testid="stSidebar"] label {
        color: #7a9bbf !important;
        font-size: 0.79rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #5a7899 !important;
        font-size: 0.77rem !important;
        line-height: 1.5 !important;
    }

    /* Inputs */
    .stNumberInput input, .stTextInput input {
        background: #0d1c31 !important;
        color: #d4e0f0 !important;
        border: 1px solid #1a3356 !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
        transition: border-color 0.2s ease !important;
    }
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #e6a817 !important;
        box-shadow: 0 0 0 2px rgba(230,168,23,0.12) !important;
        outline: none !important;
    }
    .stSelectbox > div > div {
        background: #0d1c31 !important;
        color: #d4e0f0 !important;
        border: 1px solid #1a3356 !important;
        border-radius: 8px !important;
    }

    /* Checkboxes */
    .stCheckbox label {
        color: #8ba7cc !important;
        font-size: 0.93rem !important;
        font-weight: 500 !important;
        transition: color 0.2s ease !important;
    }
    .stCheckbox label:hover { color: #e6a817 !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #e6a817 0%, #c78b10 100%) !important;
        color: #060c18 !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 12px rgba(230,168,23,0.2) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(230,168,23,0.4) !important;
    }
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #1a4a8c 0%, #102f60 100%) !important;
        color: #d4e0f0 !important;
        border: 1px solid #2358a8 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(35,88,168,0.4) !important;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0d1b2e 0%, #112238 100%) !important;
        border: 1px solid #1a3558 !important;
        border-radius: 14px !important;
        padding: 1.2rem 1.4rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #7a9bbf !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }
    [data-testid="stMetricValue"] {
        color: #e6a817 !important;
        font-size: 1.6rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricDelta"] { color: #4ade80 !important; font-size: 0.8rem !important; }

    /* Alerts */
    .stSuccess { background: rgba(16,185,129,0.08) !important; border: 1px solid rgba(16,185,129,0.25) !important; border-radius: 10px !important; color: #6ee7b7 !important; }
    .stInfo    { background: rgba(37,99,235,0.08) !important;  border: 1px solid rgba(37,99,235,0.2) !important;   border-radius: 10px !important; color: #93c5fd !important; }
    .stWarning { background: rgba(245,158,11,0.08) !important; border: 1px solid rgba(245,158,11,0.25) !important; border-radius: 10px !important; color: #fcd34d !important; }
    .stError   { background: rgba(220,38,38,0.08) !important;  border: 1px solid rgba(220,38,38,0.25) !important;  border-radius: 10px !important; color: #fca5a5 !important; }

    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        background: #0d1c31 !important;
        border: 2px dashed #1a3356 !important;
        border-radius: 12px !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover { border-color: #e6a817 !important; }

    /* General text */
    p, li { color: #8ba7cc !important; }
    h1, h2, h3, h4 { color: #d4e0f0 !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #060c18; }
    ::-webkit-scrollbar-thumb { background: #1a3356; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #e6a817; }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #0b1829 0%, #102035 60%, #0d1a30 100%);
        border: 1px solid #1a3558;
        border-radius: 18px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute; top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(230,168,23,0.07) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero::after {
        content: '';
        position: absolute; bottom: -40px; left: 25%;
        width: 300px; height: 150px;
        background: radial-gradient(ellipse, rgba(35,88,168,0.05) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.94rem; font-weight: 800;
        color: #e8f0fc !important;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em; line-height: 1.2;
    }
    .hero-title span { color: #e6a817; }
    .hero-sub { color: #5a7899 !important; font-size: 0.93rem; margin: 0 0 1.1rem 0; font-weight: 400; }
    .hero-badges { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.3rem; }
    .badge {
        background: rgba(230,168,23,0.09);
        border: 1px solid rgba(230,168,23,0.22);
        color: #c99c25 !important;
        padding: 3px 12px; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
    }
    .creator-row {
        display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap;
        padding-top: 1rem; border-top: 1px solid #162d50;
    }
    .creator-name { color: #6b8bb5 !important; font-size: 0.82rem; font-weight: 400; }
    .creator-name strong { color: #c4d5ea !important; font-weight: 600; }
    .linkedin-btn {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(10,102,194,0.14);
        border: 1px solid rgba(10,102,194,0.32);
        color: #7ab4e8 !important;
        padding: 5px 14px; border-radius: 8px;
        font-size: 0.78rem; font-weight: 600;
        text-decoration: none !important;
        transition: all 0.2s ease;
    }
    .linkedin-btn:hover { background: rgba(10,102,194,0.28); color: #b8d4f5 !important; transform: translateY(-1px); }

    /* Section divider */
    .section-hdr {
        display: flex; align-items: center; gap: 10px;
        margin: 1.8rem 0 0.8rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #1a3558;
    }
    .section-hdr h4 { color: #d4e0f0 !important; font-size: 1.0rem !important; font-weight: 600 !important; margin: 0 !important; }
    .snum {
        background: #e6a817; color: #060c18;
        font-size: 0.68rem; font-weight: 800;
        width: 21px; height: 21px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0;
    }

    /* Sidebar section label */
    .sidebar-sec {
        color: #e6a817 !important; font-size: 0.68rem !important;
        font-weight: 700 !important; text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        padding-bottom: 0.3rem !important;
        border-bottom: 1px solid #1a3558 !important;
        margin: 1rem 0 0.5rem 0 !important;
    }

    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 1.4rem 2rem;
        background: linear-gradient(135deg, #07101e 0%, #0b1929 100%);
        border: 1px solid #162d50;
        border-radius: 14px;
        display: flex; align-items: center;
        justify-content: space-between; flex-wrap: wrap; gap: 1rem;
    }
    .footer-left { color: #3d5a78 !important; font-size: 0.78rem; }
    .footer-left strong { color: #6b8bb5 !important; font-weight: 600; }
    .footer-right { display: flex; align-items: center; gap: 0.8rem; }
    .footer-tag {
        color: #e6a817 !important; font-size: 0.68rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase;
        background: rgba(230,168,23,0.08);
        border: 1px solid rgba(230,168,23,0.2);
        padding: 3px 10px; border-radius: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero Header ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Material Balance <span>Simulator</span></div>
        <p class="hero-sub">Professional reservoir engineering platform for OOIP / OGIP estimation, drive mechanism analysis, and production forecasting using classical MBE equations.</p>
        <div class="hero-badges">
            <span class="badge">🛢️ Oil Reservoirs</span>
            <span class="badge">💨 Gas Reservoirs</span>
            <span class="badge">📊 Drive Mechanism Analysis</span>
            <span class="badge">🔮 Future Performance</span>
            <span class="badge">📈 Campbell &amp; Cole Plots</span>
            <span class="badge">💧 Water Influx Modeling</span>
            <span class="badge">🧪 Fluid PVT Properties</span>
        </div>
        <div class="creator-row">
            <span class="creator-name">Developed by &nbsp;<strong>Mohit Choudhary</strong></span>
            <a class="linkedin-btn" href="https://www.linkedin.com/in/mohit-choudhary-25165730a" target="_blank">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="#7ab4e8"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
                Connect on LinkedIn
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Reservoir type selector ──────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        reservoir_type = st.selectbox("🔬 Select Reservoir Type", ["Oil", "Gas"], key="reservoir_type")

    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.markdown('<p class="sidebar-sec">⚙️ Reservoir Parameters</p>', unsafe_allow_html=True)
    
    res_p = st.sidebar.number_input("Reservoir Pressure (psi)", min_value=0.0, value=4000.0)
    t = st.sidebar.number_input("Temperature (F)", min_value=0.0, value=200.0)
    sg_g = st.sidebar.number_input("Gas Specific Gravity", min_value=0.0, value=0.75)
    c_f = st.sidebar.number_input("Rock Compressibility (1/psi)", min_value=0.0, value=3e-6, format="%.2e")
    c_w = st.sidebar.number_input("Water Compressibility (1/psi)", min_value=0.0, value=3e-6, format="%.2e")
    S_wc = st.sidebar.number_input("Initial Water Saturation (fraction)", min_value=0.0, max_value=1.0, value=0.2)

    if reservoir_type == "Oil":
        api = st.sidebar.number_input("API Gravity", min_value=0.0, value=35.0)
        r_s_input = st.sidebar.number_input("Solution Gas-Oil Ratio (SCF/STB)", min_value=0.0, value=500.0)
        m = st.sidebar.number_input("Gas Cap Ratio (m)", min_value=0.0, value=0.0)

    # Data source selection and file handling
    data_source = st.sidebar.selectbox("Select Data Source", ["Upload File", "Use Example Data"], key="data_source")

    if data_source == "Upload File":
        if reservoir_type == "Oil":
            st.sidebar.markdown("**Note:** The excel file should have column names: Date, Pressure, Cum Oil Production (MMSTB), Cum Gas Production (MMSCF), Cum Water Production (MMSTB).")
        else:
            st.sidebar.markdown("**Note:** The excel file should have column names: Date, Pressure, Cum Gas Production (BSCF), and optionally Cum Water Production (MMSTB).")
        uploaded_file = st.sidebar.file_uploader("Upload Production Data (Excel)", type=["xls", "xlsx"], key=f"uploader_{reservoir_type}")
        if uploaded_file:
            history_data = pd.read_excel(uploaded_file)
        else:
            history_data = None
    elif data_source == "Use Example Data":
        try:
            if reservoir_type == "Oil":
                history_data = pd.read_excel("example_oil_data.xlsx")
            else:
                history_data = pd.read_excel("example_gas_data.xlsx")
            st.sidebar.write(f"Using example data for {reservoir_type} reservoir.")
        except FileNotFoundError:
            st.sidebar.error(f"Example data file for {reservoir_type} not found.")
            history_data = None

    if history_data is not None:
        required_columns = (
            ['Date', 'Pressure', 'Cum Oil Production', 'Cum Gas Production', 'Cum Water Production']
            if reservoir_type == "Oil"
            else ['Date', 'Pressure', 'Cum Gas Production']
        )
        missing_columns = [col for col in required_columns if col not in history_data.columns]
        if missing_columns:
            st.error(f"Missing columns: {', '.join(missing_columns)}")
        else:
            dates = pd.to_datetime(history_data['Date'])
            pressure_data = history_data['Pressure'].values
            if reservoir_type == "Oil":
                Np_data = history_data['Cum Oil Production'].values  # MMSTB
                Gp_data = history_data['Cum Gas Production'].values  # MMSCF
                Wp_data = history_data['Cum Water Production'].values  # MMSTB
            else:
                Gp_data = history_data['Cum Gas Production'].values  # BSCF
                Wp_data = history_data.get('Cum Water Production', np.zeros(len(pressure_data))).values  # MMSTB
            n_points = len(pressure_data)

            # --- Input Validation ---
            errors = []
            if sg_g <= 0:
                errors.append("⛔ **Gas Specific Gravity** must be greater than 0 (e.g. 0.65–0.85 for natural gas).")
            if t <= 0:
                errors.append("⛔ **Temperature** must be greater than 0 °F.")
            if res_p <= 0:
                errors.append("⛔ **Reservoir Pressure** must be greater than 0 psi.")
            if S_wc >= 1.0:
                errors.append("⛔ **Initial Water Saturation** must be less than 1.0 (e.g. 0.15–0.35).")
            if reservoir_type == "Oil":
                if r_s_input <= 0:
                    errors.append("⛔ **Solution Gas-Oil Ratio** must be greater than 0 SCF/STB.")
                if api <= 0:
                    errors.append("⛔ **API Gravity** must be greater than 0.")
            if errors:
                for e in errors:
                    st.error(e)
                st.info("💡 Please fill in all Reservoir Parameters in the sidebar before running the simulator.")
                st.stop()

            # Fluid Property Calculations
            if reservoir_type == "Oil":
                p_b = 18.2 * (((r_s_input / sg_g) ** 0.83) * (10 ** (0.00091 * t - 0.0125 * api)) - 1.4)
                _col1, _col2, _col3 = st.columns(3)
                with _col1:
                    st.metric("📌 Bubble Point Pressure (Pb)", f"{p_b:,.2f} psi")
                Rs_data = np.array([calculate_Rs(p, p_b, r_s_input, sg_g, api, t) for p in pressure_data])
                Bo_data = np.array([calculate_Bo(p, p_b, r_s_input, sg_g, api, t) for p in pressure_data])
                Z_prod = np.array([compute_z(sg_g, p, t) for p in pressure_data])
                Bg_data = np.array([calculate_Bg(p, sg_g, t) for p in pressure_data])
                Bw_data = np.array([calculate_Bw(p, t) for p in pressure_data])
                Viscosity_data = np.array([calculate_viscosity_oil(p, p_b, r_s_input, api, t, sg_g) for p in pressure_data])
                B_oi, R_si, B_gi = Bo_data[0], Rs_data[0], Bg_data[0]
            else:
                Z_prod = np.array([compute_z(sg_g, p, t) for p in pressure_data])
                Bg_data = 0.02827 * Z_prod * (t + 460) / pressure_data
                Bw_data = np.array([calculate_Bw(p, t) for p in pressure_data])
                Viscosity_data = np.array([calculate_gas_viscosity(p, t, sg_g, z) for p, z in zip(pressure_data, Z_prod)])
                B_gi = Bg_data[0]

            # Step 3: Option to Plot Reservoir Plots
            st.markdown('<div class="section-hdr"><span class="snum">1</span><h4>Production History Visualization</h4></div>', unsafe_allow_html=True)
            if st.checkbox("📊 Plot Reservoir Production Data"):
                if reservoir_type == "Oil":
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                    ax1.plot(dates, Np_data, 'b-', label='Cum Oil (MMSTB)')
                    ax1.set_xlabel('Time'); ax1.set_ylabel('Cum Oil (MMSTB)')
                    ax1.legend(); ax1.grid()
                    ax1_twin = ax1.twiny()
                    ax1_twin.plot(pressure_data, Np_data, 'r--', alpha=0)
                    ax1_twin.set_xlabel('Pressure (psi)'); ax1_twin.invert_xaxis()
                    
                    ax2.plot(dates, Gp_data, 'g-', label='Cum Gas (MMSCF)')
                    ax2.set_xlabel('Time'); ax2.set_ylabel('Cum Gas (MMSCF)')
                    ax2.legend(); ax2.grid()
                    ax2_twin = ax2.twiny()
                    ax2_twin.plot(pressure_data, Gp_data, 'r--', alpha=0)
                    ax2_twin.set_xlabel('Pressure (psi)'); ax2_twin.invert_xaxis()
                    
                    ax3.plot(dates, Wp_data, 'r-', label='Cum Water (MMSTB)')
                    ax3.set_xlabel('Time'); ax3.set_ylabel('Cum Water (MMSTB)')
                    ax3.legend(); ax3.grid()
                    ax3_twin = ax3.twiny()
                    ax3_twin.plot(pressure_data, Wp_data, 'r--', alpha=0)
                    ax3_twin.set_xlabel('Pressure (psi)'); ax3_twin.invert_xaxis()
                else:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    ax1.plot(dates, Gp_data, 'b-', label='Cum Gas (BSCF)')
                    ax1.set_xlabel('Time'); ax1.set_ylabel('Cum Gas (BSCF)')
                    ax1.legend(); ax1.grid()
                    ax1_twin = ax1.twiny()
                    ax1_twin.plot(pressure_data, Gp_data, 'r--', alpha=0)
                    ax1_twin.set_xlabel('Pressure (psi)'); ax1_twin.invert_xaxis()
                    
                    ax2.plot(dates, Wp_data, 'g-', label='Cum Water (MMSTB)')
                    ax2.set_xlabel('Time'); ax2.set_ylabel('Cum Water (MMSTB)')
                    ax2.legend(); ax2.grid()
                    ax2_twin = ax2.twiny()
                    ax2_twin.plot(pressure_data, Wp_data, 'r--', alpha=0)
                    ax2_twin.set_xlabel('Pressure (psi)'); ax2_twin.invert_xaxis()
                plt.tight_layout()
                st.pyplot(fig)

            st.markdown('<div class="section-hdr"><span class="snum">2</span><h4>Fluid PVT Properties</h4></div>', unsafe_allow_html=True)
            if st.checkbox("🧪 Plot Fluid Properties"):
                plots = (
                    [("Rs (SCF/STB)", Rs_data), ("Bo (RB/STB)", Bo_data), ("Bg (RB/SCF)", Bg_data),
                     ("Bw (RB/STB)", Bw_data), ("Viscosity (cP)", Viscosity_data), ("Z-factor", Z_prod)]
                    if reservoir_type == "Oil"
                    else [("Bg (RB/SCF)", Bg_data), ("Bw (RB/STB)", Bw_data), ("Viscosity (cp)", Viscosity_data),
                          ("Z-factor", Z_prod)]
                )
                fig, axs = plt.subplots((len(plots) + 1) // 2, 2, figsize=(12, 6 * ((len(plots) + 1) // 2)))
                axs = axs.flatten()
                for ax, (label, data) in zip(axs, plots):
                    ax.plot(pressure_data, data, 'b-', label=label)
                    if reservoir_type == "Oil":
                        ax.axvline(p_b, color='r', linestyle='--', label=f'Pb: {p_b:.2f} psi')
                    ax.set_xlabel('Pressure (psi)'); ax.set_ylabel(label)
                    ax.legend(); ax.grid()
                if len(plots) % 2:
                    fig.delaxes(axs[-1])
                plt.tight_layout()
                st.pyplot(fig)

            # Precompute c_e for oil (used by multiple sections below)
            if reservoir_type == "Oil":
                c_e = (c_w * S_wc + c_f) / (1 - S_wc)

            # Initialize result variables (may be computed inside checkboxes but used later)
            N_intercept = None
            G_intercept = None
            G_intercept_SCF = None

            # Step 4: Campbell/Cole Plots
            if reservoir_type == "Oil" and st.checkbox("Plot Campbell Plot"):
                F_data = np.zeros(n_points)
                for i in range(n_points):
                    gas_term = (Gp_data[i] - Np_data[i] * Rs_data[i]) * Bg_data[i] if pressure_data[i] < p_b else 0
                    F_data[i] = Np_data[i] * Bo_data[i] + gas_term + Wp_data[i] * Bw_data[i]
                E_o_data = np.array([(Bo_data[i] - B_oi) + (R_si - Rs_data[i]) * Bg_data[i] if pressure_data[i] < p_b else Bo_data[i] - B_oi for i in range(n_points)])
                E_g_data = B_oi * (Bg_data / B_gi - 1)
                E_fw_data = (1 + m) * B_oi * c_e * (res_p - pressure_data)
                E_t_data = E_o_data + m * E_g_data + E_fw_data
                F_over_Et = np.where(np.abs(E_t_data) > 1e-6, F_data / E_t_data, np.nan)
                fig, ax = plt.subplots()
                ax.plot(F_data, F_over_Et, 'bo-', label='F / E_t vs F')
                ax.set_xlabel('Cumulative Reservoir Voidage (F)'); ax.set_ylabel('F / E_t')
                ax.set_title('Campbell Plot'); ax.legend(); ax.grid()
                st.pyplot(fig)
            elif reservoir_type == "Gas" and st.checkbox("Plot Cole Plot"):
                cole_y = np.full(n_points, np.nan)
                for i in range(1, n_points):
                    if Bg_data[i] != B_gi:
                        cole_y[i] = Gp_data[i] * Bg_data[i] / (Bg_data[i] - B_gi)
                fig, ax = plt.subplots()
                ax.plot(Gp_data[1:], cole_y[1:], 'bo-', label='Cole Plot')
                ax.set_xlabel('Cumulative Gas (BSCF)'); ax.set_ylabel(r'$\frac{G_p\,Bg}{Bg - B_{gi}}$')
                ax.set_title('Cole Plot'); ax.legend(); ax.grid()
                st.pyplot(fig)

            # Step 5: Water Influx
            influx_present = st.selectbox("Is water influx present?", ["No", "Yes"]) == "Yes"
            if influx_present:
                W_ei = st.number_input("Aquifer Volume (MMft^3)", min_value=0.0, value=0.0)
                c_t = c_w + c_f
                We_data = (W_ei / 5.615) * c_t * (res_p - pressure_data)
                if st.checkbox("Plot Water Influx"):
                    fig, ax = plt.subplots()
                    ax.plot(dates, We_data, 'r--', label='Water Influx (MMBBL)')
                    ax.set_xlabel('Time'); ax.set_ylabel('Water Influx (MMBBL)')
                    ax.set_title('Water Influx Over Time'); ax.legend(); ax.grid()
                    st.pyplot(fig)
            else:
                We_data = np.zeros(n_points)

            # Step 6: Material Balance Calculations
            st.markdown('<div class="section-hdr"><span class="snum">3</span><h4>Material Balance Analysis</h4></div>', unsafe_allow_html=True)
            if st.checkbox("⚖️ Calculate Material Balance"):
                if reservoir_type == "Oil":
                    F_data = np.zeros(n_points)
                    for i in range(n_points):
                        gas_term = (Gp_data[i] - Np_data[i] * Rs_data[i]) * Bg_data[i] if pressure_data[i] < p_b else 0
                        F_data[i] = Np_data[i] * Bo_data[i] + gas_term + Wp_data[i] * Bw_data[i]
                    E_o_data = np.array([(Bo_data[i] - B_oi) + (R_si - Rs_data[i]) * Bg_data[i] if pressure_data[i] < p_b else Bo_data[i] - B_oi for i in range(n_points)])
                    E_g_data = B_oi * (Bg_data / B_gi - 1)
                    E_fw_data = (1 + m) * B_oi * c_e * (res_p - pressure_data)
                    E_t_data = E_o_data + m * E_g_data + E_fw_data
                    if not influx_present:
                        N_intercept = np.sum(E_t_data * F_data) / np.sum(E_t_data**2)
                        fig, ax = plt.subplots()
                        ax.plot(E_t_data, F_data, 'bo', label='F vs E_t')
                        ax.plot(E_t_data, N_intercept * E_t_data, 'r-', label=f'N = {N_intercept * 1e6:.2f} STB')
                        ax.set_xlabel('E_t (RB/STB)'); ax.set_ylabel('F (MMRB)')
                        ax.set_title('Material Balance (No Influx)'); ax.legend(); ax.grid()
                        st.pyplot(fig)
                        _c1, _c2, _c3 = st.columns(3)
                        with _c1:
                            st.metric("🛢️ Estimated OOIP", f"{N_intercept:.4f} MMSTB", f"{N_intercept*1e6:,.0f} STB")
                    else:
                        # We_data is in MMRB (same units as F_data) — no scaling needed
                        F_minus_We = F_data - We_data
                        N_intercept = np.sum(E_t_data * F_minus_We) / np.sum(E_t_data**2)
                        fig, ax = plt.subplots()
                        ax.plot(E_t_data, F_minus_We, 'bo', label='F - W_e vs E_t')
                        ax.plot(E_t_data, N_intercept * E_t_data, 'r-', label=f'N = {N_intercept * 1e6:.2f} STB')
                        ax.set_xlabel('E_t (RB/STB)'); ax.set_ylabel('F - W_e (MMRB)')
                        ax.set_title('Material Balance (With Influx)'); ax.legend(); ax.grid()
                        st.pyplot(fig)
                        _c1, _c2, _c3 = st.columns(3)
                        with _c1:
                            st.metric("🛢️ Estimated OOIP (With Influx)", f"{N_intercept:.4f} MMSTB", f"{N_intercept*1e6:,.0f} STB")
                else:
                    # F_data in RB, E_t_data in RB/SCF
                    F_data = (Gp_data * 1e9) * Bg_data + (Wp_data * 1e6) * Bw_data
                    C = (c_f + c_w * S_wc) / (1 - S_wc)
                    E_t_data = (Bg_data - B_gi) + B_gi * C * (res_p - pressure_data)
                    G_intercept_SCF = np.sum(E_t_data * F_data) / np.sum(E_t_data**2)
                    G_intercept = G_intercept_SCF / 1e9
                    if not influx_present:
                        ratio = F_data / E_t_data
                        fig, ax = plt.subplots()
                        ax.plot(Gp_data, ratio, 'bo-', label=f'OGIP = {G_intercept:.2f} BSCF')
                        ax.set_xlabel('Cum Gas (BSCF)'); ax.set_ylabel('F / E_t (SCF)')
                        ax.set_title('Material Balance (No Influx)'); ax.legend(); ax.grid()
                        st.pyplot(fig)
                        _c1, _c2, _c3 = st.columns(3)
                        with _c1:
                            st.metric("💨 Estimated OGIP", f"{G_intercept:.4f} BSCF", f"{G_intercept*1e3:,.1f} MMSCF")
                    else:
                        # We_data in MMRB → convert to RB (*1e6) to match F_data in RB
                        ratio = (F_data - We_data * 1e6) / E_t_data
                        fig, ax = plt.subplots()
                        ax.plot(Gp_data, ratio, 'bo-', label=f'OGIP = {G_intercept:.2f} BSCF')
                        ax.set_xlabel('Cum Gas (BSCF)'); ax.set_ylabel('(F - W_e) / E_t (SCF)')
                        ax.set_title('Material Balance (With Influx)'); ax.legend(); ax.grid()
                        st.pyplot(fig)
                        _c1, _c2, _c3 = st.columns(3)
                        with _c1:
                            st.metric("💨 Estimated OGIP (With Influx)", f"{G_intercept:.4f} BSCF", f"{G_intercept*1e3:,.1f} MMSCF")
                    G_initial = Gp_data[0]
                    model = LinearRegression().fit((Gp_data - G_initial).reshape(-1, 1), pressure_data / Z_prod)
                    OGIP_estimated = model.intercept_ / -model.coef_[0]
                    fig, ax = plt.subplots()
                    ax.plot(Gp_data - G_initial, pressure_data / Z_prod, 'bo', label='Data')
                    x_fit = np.linspace(min(Gp_data - G_initial), max(Gp_data - G_initial), 100)
                    ax.plot(x_fit, model.predict(x_fit.reshape(-1, 1)), 'r-', label=f'OGIP = {OGIP_estimated:.2f} BSCF')
                    ax.set_xlabel('Gp - Gi (BSCF)'); ax.set_ylabel('p/Z (psi)')
                    ax.set_title('Modified p/Z Plot'); ax.legend(); ax.grid()
                    st.pyplot(fig)

            # Step 7: Drive Mechanism Analysis
            st.markdown('<div class="section-hdr"><span class="snum">4</span><h4>Drive Mechanism Analysis</h4></div>', unsafe_allow_html=True)
            if st.checkbox("🔩 Plot Drive Mechanism Analysis"):
                if reservoir_type == "Oil":
                    E_o_drive = np.array([(Bo_data[i] - B_oi) + (r_s_input - Rs_data[i]) * Bg_data[i] if pressure_data[i] < p_b else Bo_data[i] - B_oi for i in range(n_points)])
                    E_g_drive = B_oi * (Bg_data / B_gi - 1)
                    E_fw_drive = (1 + m) * B_oi * c_e * (res_p - pressure_data)
                    # We_data in MMRB; divide by N_intercept (MMSTB) → RB/STB (same units as E terms)
                    if influx_present and N_intercept is not None and N_intercept != 0:
                        We_term = We_data / N_intercept
                    else:
                        We_term = np.zeros(n_points)
                    total_expansion = E_o_drive + m * E_g_drive + E_fw_drive + We_term
                    oil_gas_pct = np.nan_to_num((E_o_drive / total_expansion) * 100)
                    gas_cap_pct = np.nan_to_num((m * E_g_drive / total_expansion) * 100)
                    rock_water_pct = np.nan_to_num((E_fw_drive / total_expansion) * 100)
                    water_influx_pct = np.nan_to_num((We_term / total_expansion) * 100)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.stackplot(pressure_data[1:], oil_gas_pct[1:], gas_cap_pct[1:], rock_water_pct[1:], water_influx_pct[1:],
                                 labels=['Oil & Gas', 'Gas Cap', 'Rock & Water', 'Water Influx'],
                                 colors=['#FF9671', '#00C4B4', '#845EC2', '#FFC75F'])
                    ax.set_xlabel('Pressure (psi)'); ax.set_ylabel('Contribution (%)')
                    ax.set_title('Drive Mechanism Analysis'); ax.legend(); ax.grid()
                    ax.invert_xaxis()
                    st.pyplot(fig)
                else:
                    E_g_drive = Bg_data - B_gi  # RB/SCF
                    C_val = (c_f + c_w * S_wc) / (1 - S_wc)
                    E_fw_drive = B_gi * C_val * (res_p - pressure_data)  # RB/SCF
                    # We_data in MMRB → convert to RB (*1e6); G_intercept_SCF in SCF → units: RB/SCF (same as E_g_drive)
                    if influx_present and G_intercept_SCF is not None and G_intercept_SCF != 0:
                        We_term_drive = (We_data * 1e6) / G_intercept_SCF
                    else:
                        We_term_drive = np.zeros(n_points)
                    total_expansion_drive = E_g_drive + E_fw_drive + We_term_drive
                    gas_exp_pct = np.nan_to_num((E_g_drive / total_expansion_drive) * 100)
                    rock_water_pct = np.nan_to_num((E_fw_drive / total_expansion_drive) * 100)
                    water_influx_pct = np.nan_to_num((We_term_drive / total_expansion_drive) * 100)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.stackplot(pressure_data[1:], gas_exp_pct[1:], rock_water_pct[1:], water_influx_pct[1:],
                                 labels=['Gas Expansion', 'Rock & Fluid', 'Water Influx'],
                                 colors=['red', 'blue', 'green'])
                    ax.set_xlabel('Pressure (psi)'); ax.set_ylabel('Contribution (%)')
                    ax.set_title('Drive Mechanism Analysis'); ax.legend(); ax.grid()
                    ax.invert_xaxis()
                    st.pyplot(fig)

            # Step 8: Future Performance Prediction
            st.markdown('<div class="section-hdr"><span class="snum">5</span><h4>Future Performance Prediction</h4></div>', unsafe_allow_html=True)
            if st.checkbox("🔮 Predict Future Performance"):
                future_date_str = st.text_input("Enter Future Date (YYYY-MM-DD):")
                if future_date_str:
                    try:
                        future_date = pd.to_datetime(future_date_str)
                        if future_date <= dates.iloc[-1]:
                            st.error("Future date must be after the last historical date.")
                        else:
                            time_days = (dates - dates.iloc[0]).dt.days
                            time_diffs = np.diff(time_days)
                            future_dates = pd.date_range(start=dates.iloc[-1], end=future_date, freq='MS')
                            if future_dates[0] != dates.iloc[-1]:
                                future_dates = np.insert(future_dates, 0, dates.iloc[-1])
                            time_to_future = (future_dates - dates.iloc[-1]).days

                            # Calculate and Display Average Flow Rates
                            if reservoir_type == "Oil":
                                q_o = np.diff(Np_data * 1e6) / time_diffs
                                q_g = np.diff(Gp_data * 1e6) / time_diffs
                                q_w = np.diff(Wp_data * 1e6) / time_diffs
                                q_o_avg, q_g_avg, q_w_avg = np.mean(q_o), np.mean(q_g), np.mean(q_w)
                                _r1, _r2, _r3 = st.columns(3)
                                with _r1: st.metric("🛢️ Avg Oil Rate",   f"{q_o_avg:,.1f} STB/day")
                                with _r2: st.metric("💨 Avg Gas Rate",   f"{q_g_avg:,.0f} SCF/day")
                                with _r3: st.metric("💧 Avg Water Rate", f"{q_w_avg:,.1f} STB/day")
                            else:
                                q_g = np.diff(Gp_data * 1e9) / time_diffs
                                q_w = np.diff(Wp_data * 1e6) / time_diffs
                                q_g_avg, q_w_avg = np.mean(q_g), np.mean(q_w)
                                _r1, _r2 = st.columns(2)
                                with _r1: st.metric("💨 Avg Gas Rate",   f"{q_g_avg:,.0f} SCF/day")
                                with _r2: st.metric("💧 Avg Water Rate", f"{q_w_avg:,.1f} STB/day")

                            # Future Performance Calculations
                            if reservoir_type == "Oil":
                                if N_intercept is None:
                                    st.warning("Please run 'Calculate Material Balance' first to get OOIP before predicting future performance.")
                                    st.stop()
                                N_STB = N_intercept * 1e6
                                Np_future, Gp_future, Wp_future, valid_dates = [], [], [], []
                                for dt, f_date in zip(time_to_future, future_dates):
                                    Np_temp = Np_data[-1] * 1e6 + q_o_avg * dt
                                    if Np_temp > N_STB:
                                        break
                                    Gp_temp = Gp_data[-1] * 1e6 + q_g_avg * dt
                                    Wp_temp = Wp_data[-1] * 1e6 + q_w_avg * dt
                                    Np_future.append(Np_temp / 1e6)
                                    Gp_future.append(Gp_temp / 1e6)
                                    Wp_future.append(Wp_temp / 1e6)
                                    valid_dates.append(f_date)
                                future_dates = valid_dates

                                def mb_oil(p, Np, Gp, Wp):
                                    Bo = calculate_Bo(p, p_b, r_s_input, sg_g, api, t)
                                    Bg = calculate_Bg(p, sg_g, t)
                                    Bw = calculate_Bw(p, t)
                                    # Use current Rs(p) for free gas calculation (not initial R_si)
                                    Rs_p = calculate_Rs(p, p_b, r_s_input, sg_g, api, t)
                                    F = Np * 1e6 * Bo + (Gp * 1e6 - Np * 1e6 * Rs_p) * Bg + Wp * 1e6 * Bw
                                    E_o = (Bo - B_oi) + (R_si - Rs_p) * Bg if p < p_b else Bo - B_oi
                                    E_g = B_oi * (Bg / B_gi - 1)
                                    E_fw = (1 + m) * B_oi * c_e * (res_p - p)
                                    E_t = E_o + m * E_g + E_fw
                                    W_e = (W_ei / 5.615) * c_t * (res_p - p) if influx_present else 0
                                    return F - N_STB * E_t - W_e

                                pressure_future = []
                                for Np_val, Gp_val, Wp_val in zip(Np_future, Gp_future, Wp_future):
                                    try:
                                        p = brentq(lambda p: mb_oil(p, Np_val, Gp_val, Wp_val), 1000, res_p)
                                        pressure_future.append(p)
                                    except:
                                        pressure_future.append(pressure_future[-1] if pressure_future else pressure_data[-1])
                                pressure_future = np.array(pressure_future)
                                Bo_future = np.array([calculate_Bo(p, p_b, r_s_input, sg_g, api, t) for p in pressure_future])
                                Rs_future = np.array([calculate_Rs(p, p_b, r_s_input, sg_g, api, t) for p in pressure_future])
                                Bg_future = np.array([calculate_Bg(p, sg_g, t) for p in pressure_future])
                                Bw_future = np.array([calculate_Bw(p, t) for p in pressure_future])
                                Viscosity_future = np.array([calculate_viscosity_oil(p, p_b, r_s_input, api, t, sg_g) for p in pressure_future])
                                Z_future = np.array([compute_z(sg_g, p, t) for p in pressure_future])
                            else:
                                if G_intercept is None:
                                    st.warning("Please run 'Calculate Material Balance' first to get OGIP before predicting future performance.")
                                    st.stop()
                                G_SCF = G_intercept * 1e9
                                Gp_future, Wp_future, valid_dates = [], [], []
                                for dt, f_date in zip(time_to_future, future_dates):
                                    Gp_temp = Gp_data[-1] * 1e9 + q_g_avg * dt
                                    if Gp_temp > G_SCF:
                                        break
                                    Wp_temp = Wp_data[-1] * 1e6 + q_w_avg * dt
                                    Gp_future.append(Gp_temp / 1e9)
                                    Wp_future.append(Wp_temp / 1e6)
                                    valid_dates.append(f_date)
                                future_dates = valid_dates

                                def mb_gas(p, Gp, Wp):
                                    Z = compute_z(sg_g, p, t)
                                    Bg = 0.02827 * Z * (t + 460) / p
                                    Bw = calculate_Bw(p, t)
                                    F = (Gp * 1e9) * Bg + (Wp * 1e6) * Bw
                                    E_g = Bg - B_gi
                                    C = (c_f + c_w * S_wc) / (1 - S_wc)
                                    E_fw = B_gi * C * (res_p - p)
                                    E_t = E_g + E_fw
                                    W_e = (W_ei / 5.615) * c_t * (res_p - p) if influx_present else 0
                                    return F - G_SCF * E_t - W_e

                                pressure_future = []
                                for Gp_val, Wp_val in zip(Gp_future, Wp_future):
                                    try:
                                        p = brentq(lambda p: mb_gas(p, Gp_val, Wp_val), 100, res_p)
                                        pressure_future.append(p)
                                    except:
                                        pressure_future.append(pressure_future[-1] if pressure_future else pressure_data[-1])
                                pressure_future = np.array(pressure_future)
                                Z_future = np.array([compute_z(sg_g, p, t) for p in pressure_future])
                                Bg_future = 0.02827 * Z_future * (t + 460) / pressure_future
                                Bw_future = np.array([calculate_Bw(p, t) for p in pressure_future])
                                Viscosity_future = np.array([calculate_gas_viscosity(p, t, sg_g, z) for p, z in zip(pressure_future, Z_future)])

                            # Combine Historical and Future Data
                            all_dates = np.append(dates, future_dates)
                            all_pressure = np.append(pressure_data, pressure_future)
                            all_Gp = np.append(Gp_data, Gp_future if reservoir_type == "Gas" else Gp_future)
                            all_Wp = np.append(Wp_data, Wp_future)
                            all_Bg = np.append(Bg_data, Bg_future)
                            all_Bw = np.append(Bw_data, Bw_future)
                            all_Viscosity = np.append(Viscosity_data, Viscosity_future)
                            all_Z = np.append(Z_prod, Z_future)
                            if reservoir_type == "Oil":
                                all_Np = np.append(Np_data, Np_future)
                                all_Bo = np.append(Bo_data, Bo_future)
                                all_Rs = np.append(Rs_data, Rs_future)

                            # Plot P/Z vs Gp - Gi for Gas Reservoir
                            if reservoir_type == "Gas" and st.checkbox("Plot P/Z vs Gp - Gi (Future Prediction)"):
                                G_initial = Gp_data[0]
                                all_Gp_minus_Gi = all_Gp - G_initial
                                all_P_over_Z = all_pressure / all_Z
                                plt.figure(figsize=(8,6))
                                plt.plot(all_Gp_minus_Gi[:len(dates)], all_P_over_Z[:len(dates)], 'bo-', linewidth=2, label='Historical')
                                plt.plot(all_Gp_minus_Gi[len(dates):], all_P_over_Z[len(dates):], 'r--', linewidth=2, label='Future')
                                plt.xlabel('Gp - Gi (BSCF)', fontweight='bold')
                                plt.ylabel('P/Z (psi)', fontweight='bold')
                                plt.title('P/Z vs Gp - Gi (Future Prediction)', fontweight='bold')
                                plt.legend(prop={'weight':'bold'})
                                plt.grid()
                                st.pyplot(plt)

                            # Step 9: Plot Reservoir Properties Prediction
                            if st.checkbox("Plot Reservoir Properties Prediction"):
                                fig, axs = plt.subplots(3 if reservoir_type == "Oil" else 2, 2, figsize=(14, 15 if reservoir_type == "Oil" else 10))
                                axs = axs.flatten()
                                plot_data = (
                                    [("Bo (RB/STB)", all_Bo, all_Bo[len(dates):]), ("Rs (SCF/STB)", all_Rs, all_Rs[len(dates):]),
                                     ("Bg (RB/SCF)", all_Bg, all_Bg[len(dates):]), ("Bw (RB/STB)", all_Bw, all_Bw[len(dates):]),
                                     ("Viscosity (cP)", all_Viscosity, all_Viscosity[len(dates):]), ("Z-factor", all_Z, all_Z[len(dates):])]
                                    if reservoir_type == "Oil"
                                    else [("Bg (RB/SCF)", all_Bg, all_Bg[len(dates):]), ("Bw (RB/STB)", all_Bw, all_Bw[len(dates):]),
                                          ("Viscosity (cp)", all_Viscosity, all_Viscosity[len(dates):]), ("Z-factor", all_Z, all_Z[len(dates):])]
                                )
                                for ax, (label, hist_data, fut_data) in zip(axs, plot_data):
                                    ax.plot(all_pressure[:len(dates)], hist_data[:len(dates)], 'b-o', label=f'Hist {label}')
                                    ax.plot(all_pressure[len(dates):], fut_data, 'r--', label=f'Fut {label}')
                                    if reservoir_type == "Oil":
                                        ax.axvline(p_b, color='k', linestyle='--', label=f'Pb: {p_b:.2f} psi')
                                    ax.set_xlabel('Pressure (psi)'); ax.set_ylabel(label)
                                    ax.set_title(f'{label} vs Pressure'); ax.legend(); ax.grid()
                                if reservoir_type == "Oil" and len(axs) > len(plot_data):
                                    fig.delaxes(axs[-1])
                                plt.tight_layout()
                                st.pyplot(fig)

                            # Step 10: Plot Cumulative Production
                            if st.checkbox("Plot Cumulative Production"):
                                if reservoir_type == "Oil":
                                    # vs Time
                                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                                    ax1.plot(dates, Np_data, 'b-', label='Cum Oil (Hist)')
                                    ax1.plot(future_dates, Np_future, 'r--', label='Cum Oil (Fut)')
                                    ax1.set_xlabel('Date'); ax1.set_ylabel('Cum Oil (MMSTB)')
                                    ax1.set_title('Cum Oil vs Time'); ax1.legend(); ax1.grid()
                                    
                                    ax2.plot(dates, Gp_data, 'g-', label='Cum Gas (Hist)')
                                    ax2.plot(future_dates, Gp_future, 'r--', label='Cum Gas (Fut)')
                                    ax2.set_xlabel('Date'); ax2.set_ylabel('Cum Gas (MMSCF)')
                                    ax2.set_title('Cum Gas vs Time'); ax2.legend(); ax2.grid()
                                    
                                    ax3.plot(dates, Wp_data, 'm-', label='Cum Water (Hist)')
                                    ax3.plot(future_dates, Wp_future, 'r--', label='Cum Water (Fut)')
                                    ax3.set_xlabel('Date'); ax3.set_ylabel('Cum Water (MMSTB)')
                                    ax3.set_title('Cum Water vs Time'); ax3.legend(); ax3.grid()
                                    plt.tight_layout()
                                    st.pyplot(fig)

                                    # vs Pressure
                                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                                    ax1.plot(pressure_data, Np_data, 'b-', label='Cum Oil (Hist)')
                                    ax1.plot(pressure_future, Np_future, 'r--', label='Cum Oil (Fut)')
                                    ax1.set_xlabel('Pressure (psi)'); ax1.set_ylabel('Cum Oil (MMSTB)')
                                    ax1.set_title('Cum Oil vs Pressure'); ax1.legend(); ax1.grid()
                                    
                                    ax2.plot(pressure_data, Gp_data, 'g-', label='Cum Gas (Hist)')
                                    ax2.plot(pressure_future, Gp_future, 'r--', label='Cum Gas (Fut)')
                                    ax2.set_xlabel('Pressure (psi)'); ax2.set_ylabel('Cum Gas (MMSCF)')
                                    ax2.set_title('Cum Gas vs Pressure'); ax2.legend(); ax2.grid()
                                    
                                    ax3.plot(pressure_data, Wp_data, 'm-', label='Cum Water (Hist)')
                                    ax3.plot(pressure_future, Wp_future, 'r--', label='Cum Water (Fut)')
                                    ax3.set_xlabel('Pressure (psi)'); ax3.set_ylabel('Cum Water (MMSTB)')
                                    ax3.set_title('Cum Water vs Pressure'); ax3.legend(); ax3.grid()
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                else:
                                    # vs Time
                                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                                    ax1.plot(dates, Gp_data, 'b-', label='Cum Gas (Hist)')
                                    ax1.plot(future_dates, Gp_future, 'r--', label='Cum Gas (Fut)')
                                    ax1.set_xlabel('Date'); ax1.set_ylabel('Cum Gas (BSCF)')
                                    ax1.set_title('Cum Gas vs Time'); ax1.legend(); ax1.grid()
                                    
                                    ax2.plot(dates, Wp_data, 'g-', label='Cum Water (Hist)')
                                    ax2.plot(future_dates, Wp_future, 'r--', label='Cum Water (Fut)')
                                    ax2.set_xlabel('Date'); ax2.set_ylabel('Cum Water (MMSTB)')
                                    ax2.set_title('Cum Water vs Time'); ax2.legend(); ax2.grid()
                                    plt.tight_layout()
                                    st.pyplot(fig)

                                    # vs Pressure
                                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                                    ax1.plot(pressure_data, Gp_data, 'b-', label='Cum Gas (Hist)')
                                    ax1.plot(pressure_future, Gp_future, 'r--', label='Cum Gas (Fut)')
                                    ax1.set_xlabel('Pressure (psi)'); ax1.set_ylabel('Cum Gas (BSCF)')
                                    ax1.set_title('Cum Gas vs Pressure'); ax1.legend(); ax1.grid()
                                    
                                    ax2.plot(pressure_data, Wp_data, 'g-', label='Cum Water (Hist)')
                                    ax2.plot(pressure_future, Wp_future, 'r--', label='Cum Water (Fut)')
                                    ax2.set_xlabel('Pressure (psi)'); ax2.set_ylabel('Cum Water (MMSTB)')
                                    ax2.set_title('Cum Water vs Pressure'); ax2.legend(); ax2.grid()
                                    plt.tight_layout()
                                    st.pyplot(fig)

                            # Save Data
                            final_output = pd.DataFrame({
                                'Date': all_dates,
                                'Pressure (psi)': all_pressure,
                                'Cum Gas Production': all_Gp,
                                'Cum Water Production (MMSTB)': all_Wp,
                                'Bg (RB/SCF)': all_Bg,
                                'Bw (RB/STB)': all_Bw,
                                'Viscosity': all_Viscosity,
                                'Z-factor': all_Z
                            })
                            if reservoir_type == "Oil":
                                final_output['Cum Oil Production (MMSTB)'] = all_Np
                                final_output['Bo (RB/STB)'] = all_Bo
                                final_output['Rs (SCF/STB)'] = all_Rs
                                final_output.rename(columns={'Cum Gas Production': 'Cum Gas Production (MMSCF)', 'Viscosity': 'Viscosity (cP)'}, inplace=True)
                            else:
                                final_output.rename(columns={'Cum Gas Production': 'Cum Gas Production (BSCF)', 'Viscosity': 'Viscosity (cp)'}, inplace=True)
                            csv = final_output.to_csv(index=False)
                            st.download_button("Download Results", csv, f"{reservoir_type.lower()}_material_balance.csv", "text/csv")
                    except ValueError:
                        st.error("Invalid date format. Use YYYY-MM-DD.")
    else:
        if data_source == "Upload File":
            st.info("📂 Please upload a production data Excel file using the sidebar uploader to begin analysis.")
        elif data_source == "Use Example Data":
            st.error("Example data file not found in the application directory. Please upload a production data file.")

    # ── Footer ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        <div class="footer-left">
            <strong>MBE Simulator</strong> &nbsp;·&nbsp; Professional Material Balance Engineering Platform<br>
            Correlations: Standing, Vasquez-Beggs, McCain, Hall-Yarborough (Z-factor), Lee et al. (gas viscosity)
        </div>
        <div class="footer-right">
            <span class="footer-tag">⛽ Petroleum Engineering</span>
            <a class="linkedin-btn" href="https://www.linkedin.com/in/mohit-choudhary-25165730a" target="_blank" style="font-size:0.73rem;padding:4px 11px;">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="#7ab4e8"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
                Mohit Choudhary
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()