import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from simpeg.electromagnetics.static import resistivity as dc
from simpeg import maps

st.set_page_config(page_title="1D DC Forward (SimPEG)", page_icon="🪪", layout="wide")
st.title("1D DC Resistivity — Forward Modelling (SimPEG)")
st.markdown(
    "Configure a layered Earth and electrode geometry, then compute the **apparent resistivity** curve. "
    "Uses `simpeg.electromagnetics.static.resistivity.simulation_1d.Simulation1DLayers`."
)

# ---------- SIDEBAR: Survey setup ----------
with st.sidebar:
    st.header("Geometry")

    array_type = st.radio(
        "Array type", ["Schlumberger", "Wenner"], index=0, help="Wenner: MN/2 = AB/6. Schlumberger: set MN/2 ratio."
    )

    colA1, colA2 = st.columns(2)
    with colA1:
        ab2_min = st.number_input("AB/2 min (m)", min_value=0.1, value=5.0, step=0.1, format="%.2f")
    with colA2:
        ab2_max = st.number_input("AB/2 max (m)", min_value=ab2_min + 0.1, value=300.0, step=1.0, format="%.2f")

    n_stations = st.slider("Number of stations", min_value=8, max_value=60, value=25, step=1)

    if array_type == "Schlumberger":
        mn2_ratio = st.slider(
            "MN/2 as fraction of AB/2",
            min_value=0.02, max_value=0.49, value=0.10, step=0.01,
            help="MN/2 = ratio × AB/2. Must be < 0.5."
        )
    else:
        mn2_ratio = 1.0/3.0  # Wenner: MN/2 = AB/6 → ratio = (AB/6)/(AB/2) = 1/3

    st.divider()
    st.header("Layers")

    n_layers = st.slider("Number of layers", 3, 5, 4, help="Total layers (half-space is the last layer).")
    # Default models
    default_rho = [10.0, 30.0, 15.0, 50.0, 100.0][:n_layers]
    default_thk = [2.0, 8.0, 60.0, 120.0][:max(0, n_layers-1)]

    layer_rhos = []
    for i in range(n_layers):
        layer_rhos.append(
            st.number_input(f"ρ Layer {i+1} (Ω·m)", min_value=0.1, value=float(default_rho[i]), step=0.1)
        )

    thicknesses = []
    if n_layers > 1:
        st.caption("Thicknesses for the **upper** N−1 layers (last layer is half-space):")
        for i in range(n_layers - 1):
            thicknesses.append(
                st.number_input(f"Thickness L{i+1} (m)", min_value=0.1, value=float(default_thk[i]), step=0.1)
            )
if len(thicknesses):
    thicknesses = np.r_[thicknesses]
else:
    thicknesses = np.array([])

st.divider()

# ---------- Build survey ----------
# AB/2 spaced geometrically
AB2 = np.geomspace(ab2_min, ab2_max, n_stations)
# MN/2 from ratio; ensure MN/2 < AB/2
MN2 = np.minimum(mn2_ratio * AB2, 0.49 * AB2)
eps = 1e-6  # avoid exact coincidence M=A, N=B

src_list = []
for L, a in zip(AB2, MN2):
    # A,B at ±AB/2; M,N at ±MN/2 (nudged by eps)
    A = np.r_[-L, 0.0, 0.0]
    B = np.r_[ +L, 0.0, 0.0]
    M = np.r_[ -(a - eps), 0.0, 0.0]
    N = np.r_[ +(a - eps), 0.0, 0.0]
    rx = dc.receivers.Dipole(M, N, data_type="apparent_resistivity")
    src = dc.sources.Dipole([rx], A, B)
    src_list.append(src)

survey = dc.Survey(src_list)

# ---------- Simulation + forward ----------
rho = np.r_[layer_rhos]
rho_map = maps.IdentityMap(nP=len(rho))

sim = dc.simulation_1d.Simulation1DLayers(
    survey=survey, rhoMap=rho_map, thicknesses=thicknesses
)

# Compute dpred (apparent resistivity)
try:
    rho_app = sim.dpred(rho)
    ok = True
except Exception as e:
    ok = False
    st.error(f"Forward modelling failed: {e}")

# ---------- Layout ----------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Sounding curve (log–log)")
    if ok:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(AB2, rho_app, "o-", label="ρₐ (predicted)")
        ax.grid(True, which="both", ls=":")
        ax.set_xlabel("AB/2 (m)")
        ax.set_ylabel("Apparent resistivity (Ω·m)")
        ax.set_title(f"{array_type} VES (forward)")
        # Optional fixed ticks — comment out if you prefer auto:
        # ax.set_xlim(1, 1000)
        # ax.set_ylim(0.5, 1e4)
        st.pyplot(fig, clear_figure=True)

        # Download CSV
        df_out = pd.DataFrame({
            "AB/2 (m)": AB2,
            "MN/2 (m)": MN2,
            "Apparent resistivity (ohm·m)": rho_app,
        })
        st.download_button(
            "Download synthetic data (CSV)",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="synthetic_VES.csv",
            mime="text/csv",
        )

with col2:
    st.subheader("Layered model")
    if ok:
        # Quick “block” plot for layers
        fig2, ax2 = plt.subplots(figsize=(4, 5))
        rho_vals = rho
        # Build depth boundaries
        if len(thicknesses):
            interfaces = np.r_[0.0, np.cumsum(thicknesses)]
        else:
            interfaces = np.r_[0.0]
        # Add a bottom depth for plotting last layer
        z_bottom = interfaces[-1] + max(interfaces[-1]*0.3, 10.0)
        tops = np.r_[interfaces, interfaces[-1]]          # N values
        bottoms = np.r_[interfaces[1:], z_bottom]         # N values

        for i in range(n_layers):
            ax2.fill_betweenx([tops[i], bottoms[i]], 0, rho_vals[i], alpha=0.35)
            ax2.text(rho_vals[i]*1.05, (tops[i]+bottoms[i])/2, f"{rho_vals[i]:.1f} Ω·m", va="center", fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel("Resistivity (Ω·m)")
        ax2.set_ylabel("Depth (m)")
        ax2.grid(True, ls=":")
        ax2.set_title("Block model")
        st.pyplot(fig2, clear_figure=True)

    # Also display a tidy table of the model
    model_df = pd.DataFrame({
        "Layer": np.arange(1, n_layers+1),
        "Resistivity (Ω·m)": rho,
        "Thickness (m)": [*thicknesses, np.nan],
        "Note": [""]*(n_layers-1) + ["Half-space"]
    })
    st.dataframe(model_df, use_container_width=True)

st.caption(
    "Notes: (1) Schlumberger uses your MN/2 ratio; Wenner fixes MN/2 = AB/6. "
    "(2) A tiny epsilon is subtracted so M≠A and N≠B to avoid singularities. "
    "(3) If you hit numerical issues at extreme geometries, reduce AB/2 range or adjust MN/2."
)
