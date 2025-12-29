import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import base64
import os

# ========================================================
# PAGE CONFIG
# ========================================================
st.set_page_config(
    page_title="Interactive TET Degradation Model",
    page_icon="🌍",
    layout="wide"
)

# ========================================================
# IIT MADRAS BACKGROUND (TRANSPARENT)
# ========================================================
def set_background(image_file):
    if not os.path.exists(image_file):
        return
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}") 
                        no-repeat center center fixed;
            background-size: 40%;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("iitm_logo.png")

# ========================================================
# UNOFFICIAL WATERMARK (TOP-RIGHT)
# ========================================================
st.markdown(
    """
    <div style="
        position: fixed;
        top: 15px;
        right: 20px;
        font-size: 14px;
        color: #aa0000;
        font-weight: bold;
        opacity: 0.7;
        z-index: 9999;">
        ⚠️ Unofficial App
    </div>
    """,
    unsafe_allow_html=True
)

# ========================================================
# LOAD MODEL DATA  (UNCHANGED)
# ========================================================
uploaded_file = "data.xlsx"

sheets = {
    "Catalyst dose": "Catalyst dose",
    "PMS dose": "PMS dose",
    "pH study": "pH study",
    "Power study": "Power study",
    "TET concentration": "TET concentration"
}

k_tables = {}
xls = pd.ExcelFile(uploaded_file)

for label, sheet in sheets.items():
    df = pd.read_excel(xls, sheet_name=sheet)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["k_obs"] = pd.to_numeric(df["k_obs"], errors="coerce")
    df = df.dropna(subset=["value", "k_obs"])
    k_tables[label] = df

# ========================================================
# TRAIN MODELS (UNCHANGED)
# ========================================================
labels_units = {
    "Catalyst dose": ("Catalyst dose", "mg/L"),
    "PMS dose": ("PMS concentration", "mM"),
    "pH study": ("pH level", ""),
    "Power study": ("Microwave power", "W"),
    "TET concentration": ("Initial TET", "mg/L")
}

reg_models = {}

for key, df in k_tables.items():
    X = df[["value"]].values
    y = df["k_obs"].values
    model = LinearRegression()
    model.fit(X, y)
    reg_models[key] = {
        "model": model,
        "name": labels_units[key][0],
        "unit": labels_units[key][1]
    }

# ========================================================
# ALLOWED PARAMETER RANGES (UNCHANGED)
# ========================================================
param_ranges = {
    "Catalyst dose": (5, 30),
    "PMS dose": (20, 120),
    "pH study": (1, 14),
    "Power study": (200, 700),
    "TET concentration": (1, 20)
}

# ========================================================
# STREAMLIT UI (UNCHANGED)
# ========================================================
st.title("⭐ Interactive TET Degradation Model")
st.write("Predict **removal efficiency vs time** by changing any experimental parameter.")

parameter = st.selectbox("Select Parameter", list(reg_models.keys()))
p_min, p_max = param_ranges[parameter]

value = st.slider("Adjust Value", min_value=float(p_min), max_value=float(p_max), step=0.5)

info = reg_models[parameter]
model = info["model"]
name, unit = info["name"], info["unit"]

k = model.predict(np.array([[value]]))[0]

t = np.linspace(0, 10, 200)
removal = (1 - np.exp(-k * t)) * 100

# ========================================================
# PLOT (UNCHANGED)
# ========================================================
fig, ax = plt.subplots(figsize=(8, 5))
label_text = f"{name} = {value} {unit}" if unit else f"{name} = {value}"
ax.plot(t, removal, linewidth=3)
ax.set_title(f"TET Removal vs Time ({label_text})")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Removal (%)")
ax.grid(alpha=0.3)
st.pyplot(fig)

# ========================================================
# PRINT VALUES (UNCHANGED)
# ========================================================
st.subheader("🔍 Predicted Values")
st.write(f"**Predicted k_obs:** {k:.5f}")
st.write(f"**Removal at 1 min:** {(1 - np.exp(-k*1))*100:.2f}%")
st.write(f"**Removal at 3 min:** {(1 - np.exp(-k*3))*100:.2f}%")
st.write(f"**Removal at 5 min:** {(1 - np.exp(-k*5))*100:.2f}%")
st.write(f"**Removal at 6 min:** {(1 - np.exp(-k*6))*100:.2f}%")

# ========================================================
# FOOTER
# ========================================================
st.markdown(
    """
    <hr>
    <div style="text-align:center; font-size:14px; opacity:0.85;">
        Developed by <b>Sher Ali Momand</b>
    </div>
    """,
    unsafe_allow_html=True
)
