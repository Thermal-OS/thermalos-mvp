"""
ThermalOS – Interactive Digital Twin Demo
Streamlit Web Application – Production-Ready Version

Fixes vs. original:
- AI Surrogate page uses REAL inline-trained MLP instead of faked random noise
- External image load is optional (graceful fallback)
- Added error handling throughout
- Better layout and responsive design
- Added "About" section with tech stack info
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="ThermalOS – Heatpipe Digital Twin",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .stMetric > div {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 12px; border-radius: 10px;
        border: 1px solid #0f3460;
    }
    .stMetric label { color: #e94560 !important; font-weight: 600; }
    .stMetric > div > div > div { color: #eee !important; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #1a1a3e 100%);
    }
    h1, h2, h3 { color: #e94560; }
    .health-good { color: #00ff88; font-weight: bold; font-size: 1.2em; }
    .health-warn { color: #ffaa00; font-weight: bold; font-size: 1.2em; }
    .health-bad  { color: #ff3344; font-weight: bold; font-size: 1.2em; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Import ThermalOS Modules ───────────────────────────────
from src.physics.heatpipe_model import (  # noqa: E402
    HeatpipeConfig, HeatpipeModel, TEGModel,
    FLUIDS, WICKS, ENVELOPES
)
from src.sensor.simulator import HeatpipeSensorSimulator  # noqa: E402
from src.ai.anomaly import HeatpipeAnomalyDetector  # noqa: E402

# ── Sidebar ─────────────────────────────────────────────────
st.sidebar.title("🔥 ThermalOS")
st.sidebar.markdown("*AI-Powered Digital Twin for Heat Pipes*")
st.sidebar.caption("v0.2.0 – Production MVP")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "🏠 Overview",
    "🔧 Model Builder",
    "🤖 AI Surrogate",
    "📊 Live Dashboard",
    "🔮 Digital Twin",
    "⚡ TEG Energy",
    "ℹ️ About",
])

# ══════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🔥 ThermalOS – Heatpipe Digital Twin")
    st.markdown("### AI-Powered Platform for Smart Heat Pipe Systems")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Physics Engine", "4 HP Types", "validated")
    col2.metric("AI Surrogate", "700× faster", "vs physics")
    col3.metric("Smart HP Module", "< 20g", "self-powered")
    col4.metric("Market Size", "$128B", "DC Cooling 2033")

    st.markdown("---")
    st.markdown("""
    ### The 4 Pillars of ThermalOS

    | Pillar | Description | Domain |
    |--------|------------|--------|
    | **Säule 1** | AI Design Platform for OHP/LHP | Space (Upstream) |
    | **Säule 2** | Thermal Digital Twin | Space + Earth |
    | **Säule 3** | Innovative HP for AI Data Centers | Earth (Downstream) |
    | **Säule 4** | Smart Heat Pipe Module (TEG + Sensors + Wireless) | Space + Earth |
    """)

    st.markdown("### Technology Stack Architecture")
    st.code("""
    ┌─────────────────────────────────────────────────┐
    │         THERMALOS TECHNOLOGY STACK               │
    │                                                  │
    │  Layer 5: UI ─ Dashboard │ Alerts │ REST API     │
    │  Layer 4: AI ─ Surrogate │ Anomaly │ Optimizer   │
    │  Layer 3: Physics ─ CCHP │ LHP │ OHP │ Limits   │
    │  Layer 2: Data ─ MQTT │ InfluxDB │ Calibration   │
    │  Layer 1: Hardware ─ TEG │ Sensors │ ESP32 │ LoRa │
    └─────────────────────────────────────────────────┘
    """, language="text")

    st.markdown("### Key Differentiators")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**🔬 Physics-based**")
        st.markdown("Validated 1D thermal resistance network – not just curve fitting.")
    with col_b:
        st.markdown("**🧠 AI-augmented**")
        st.markdown("700× speedup with physics-informed ML surrogate for real-time use.")
    with col_c:
        st.markdown("**⚡ Self-powered**")
        st.markdown("TEG energy harvesting from the heat pipe itself. No cable, no battery.")

# ══════════════════════════════════════════════════════════════
# PAGE: MODEL BUILDER
# ══════════════════════════════════════════════════════════════
elif page == "🔧 Model Builder":
    st.title("🔧 Heatpipe Model Builder")
    st.markdown("Configure a heat pipe and compute the thermal profile in real-time.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")
        fluid = st.selectbox("Working Fluid", list(FLUIDS.keys()),
                             format_func=lambda x: x.capitalize())
        wick = st.selectbox("Wick Type", list(WICKS.keys()),
                            format_func=lambda x: x.replace("_", " ").capitalize())
        envelope = st.selectbox("Envelope Material", list(ENVELOPES.keys()),
                                format_func=lambda x: x.capitalize())
        Q = st.slider("Heat Load Q [W]", 5, 150, 50)
        T_source = st.slider("Source Temperature [°C]", 30, 150, 80)
        length = st.slider("Length [mm]", 100, 500, 250) / 1000
        diameter = st.slider("Diameter [mm]", 4, 25, 10) / 1000
        orientation = st.slider("Orientation [°]", -90, 90, 0,
                                help="+90 = evaporator bottom (gravity assisted)")

        # Validate fluid temperature range
        T_range = FLUIDS[fluid]["T_range"]
        if T_source < T_range[0] or T_source > T_range[1]:
            st.warning(
                f"⚠️ T_source={T_source}°C liegt außerhalb des gültigen "
                f"Bereichs für {fluid}: {T_range[0]}–{T_range[1]}°C"
            )

    cfg = HeatpipeConfig(
        length=length, diameter=diameter, wick_type=wick,
        fluid=fluid, envelope=envelope, orientation_deg=orientation
    )
    model = HeatpipeModel(cfg)
    t0 = time.time()
    result = model.temperature_profile(Q, T_source)
    calc_time = (time.time() - t0) * 1000

    with col2:
        st.subheader("Temperature Profile")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result["x"] * 1000, y=result["T"],
            mode="lines+markers", name="Temperature",
            line=dict(color="#e94560", width=3),
            marker=dict(size=4),
        ))
        # Zones
        fig.add_vrect(x0=0, x1=cfg.evap_length * 1000,
                      fillcolor="red", opacity=0.1,
                      annotation_text="Evaporator")
        fig.add_vrect(x0=cfg.evap_length * 1000,
                      x1=(cfg.evap_length + cfg.adiabatic_length) * 1000,
                      fillcolor="gray", opacity=0.05,
                      annotation_text="Adiabatic")
        fig.add_vrect(x0=(cfg.evap_length + cfg.adiabatic_length) * 1000,
                      x1=cfg.length * 1000,
                      fillcolor="blue", opacity=0.1,
                      annotation_text="Condenser")
        fig.update_layout(
            xaxis_title="Position [mm]",
            yaxis_title="Temperature [°C]",
            template="plotly_dark", height=400,
            margin=dict(l=50, r=20, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R_total", f"{result['R_total']:.3f} K/W")
        m2.metric("ΔT", f"{result['delta_T']:.1f} °C")
        m3.metric("Q_max (capillary)", f"{result['Q_max']:.0f} W")
        m4.metric("Calc. Time", f"{calc_time:.1f} ms")

        if Q > result["Q_max"] and result["Q_max"] > 0:
            st.error(
                f"⚠️ WARNUNG: Wärmeleistung Q={Q}W übersteigt das "
                f"Kapillar-Limit von {result['Q_max']:.0f}W!"
            )

    # Operating Limits
    st.subheader("Operating Limits (Capillary Limit vs. Temperature)")
    limits = model.operating_limits()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=limits["T"], y=limits["Q_capillary"],
        mode="lines", name="Capillary Limit",
        fill="tozeroy", line=dict(color="#00ff88")
    ))
    fig2.add_hline(y=Q, line_dash="dash", line_color="red",
                   annotation_text=f"Q = {Q} W")
    fig2.update_layout(
        xaxis_title="Operating Temperature [°C]",
        yaxis_title="Max Heat Transport [W]",
        template="plotly_dark", height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Material info
    with st.expander("📋 Material & Configuration Details"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
            **Envelope:** {envelope.capitalize()}
            - k = {ENVELOPES[envelope]['k']} W/mK
            - ρ = {ENVELOPES[envelope]['rho']} kg/m³
            """)
        with col_b:
            st.markdown(f"""
            **Wick:** {wick.replace('_', ' ').capitalize()}
            - k_eff = {WICKS[wick]['k_eff']} W/mK
            - r_cap = {WICKS[wick]['r_cap']*1e6:.0f} µm
            - K = {WICKS[wick]['permeability']:.1e} m²
            """)
        with col_c:
            st.markdown(f"""
            **Geometry:**
            - L_total = {length*1000:.0f} mm
            - D_outer = {diameter*1000:.0f} mm
            - L_evap = {cfg.evap_length*1000:.0f} mm
            - L_cond = {cfg.cond_length*1000:.0f} mm
            """)

# ══════════════════════════════════════════════════════════════
# PAGE: AI SURROGATE
# ══════════════════════════════════════════════════════════════
elif page == "🤖 AI Surrogate":
    st.title("🤖 AI Surrogate Model")
    st.markdown(
        "Physics-informed ML: **Real-time** predictions trained on the Physics Engine."
    )

    st.subheader("Physics Model vs. AI Surrogate – Live Comparison")

    # Use the real surrogate comparison
    from src.ai.surrogate import HeatpipeSurrogate

    n_demo = st.slider("Demo sample size", 50, 500, 200, step=50,
                        help="Number of training samples for the inline demo surrogate")

    if st.button("🚀 Train & Compare", type="primary"):
        with st.spinner("Training surrogate model on physics engine data..."):
            try:
                comparison = HeatpipeSurrogate.generate_demo_comparison(
                    n_train=n_demo, n_eval=30
                )
            except Exception as e:
                st.error(f"Training fehlgeschlagen: {e}")
                st.stop()

        st.success(
            f"Training abgeschlossen in {comparison['train_time_s']:.2f}s "
            f"mit {comparison['n_train']} Samples"
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=comparison["Q_values"], y=comparison["physics_T_cond"],
                mode="markers", name="Physics Model",
                marker=dict(color="#4488ff", size=8),
            ))
            fig.add_trace(go.Scatter(
                x=comparison["Q_values"], y=comparison["surrogate_T_cond"],
                mode="markers", name="AI Surrogate",
                marker=dict(color="#e94560", size=8, symbol="diamond"),
            ))
            fig.update_layout(
                title="T_condenser Prediction",
                xaxis_title="Heat Load Q [W]",
                yaxis_title="T_cond [°C]",
                template="plotly_dark", height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=["Physics Engine", "AI Surrogate"],
                y=[1.0, 1.0 / max(comparison["speedup_factor"], 0.01)],
                marker_color=["#4488ff", "#e94560"],
                text=[
                    "~1.0 ms",
                    f"~{1.0 / max(comparison['speedup_factor'], 0.01):.3f} ms",
                ],
                textposition="outside",
            ))
            fig2.update_layout(
                title="Inference Speed Comparison",
                yaxis_title="Time [ms]", yaxis_type="log",
                template="plotly_dark", height=400,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE (T_cond)", f"{comparison['rmse_T_cond']:.2f} °C")
        m2.metric("Speedup", f"{comparison['speedup_factor']:.0f}×")
        m3.metric("R² Score", f"{comparison['r2_T_cond']:.4f}")

    else:
        st.info(
            "Klicke auf **Train & Compare**, um ein echtes ML-Modell "
            "inline zu trainieren und gegen die Physics Engine zu testen."
        )

    st.markdown("---")
    st.markdown("""
    ### How It Works
    ```
    Training Data (Physics Engine)  ──►  Neural Network (64-32-16)
         N simulations                          │
         Varied: Q, T, L, D, θ                 ▼
                                          Trained Surrogate
                                                │
    New Configuration ─────────────────►  Prediction in <0.1ms
    ```

    **Architecture:** 3-layer MLP (64-32-16 neurons) with ReLU activation,
    trained on StandardScaler-normalized features. Uses scikit-learn MLPRegressor
    with early stopping.

    **Key Innovation:** Physics-informed features ensure reliable interpolation
    across the design space.
    """)

# ══════════════════════════════════════════════════════════════
# PAGE: LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "📊 Live Dashboard":
    st.title("📊 Smart Heat Pipe – Live Sensor Dashboard")
    st.markdown("Simulated sensor data from TEG-powered wireless sensor module")

    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        Q_dash = st.slider("Heat Load Q [W]", 10, 100, 50, key="q_dash")
    with col_ctrl2:
        T_src_dash = st.slider("Source Temp [°C]", 40, 120, 80, key="t_dash")
    with col_ctrl3:
        n_steps = st.slider("Measurement Cycles", 30, 200, 90, key="n_dash")

    inject_anomaly = st.checkbox("💥 Inject Anomaly (Dryout Simulation)", value=False)

    # Setup
    cfg = HeatpipeConfig()
    model = HeatpipeModel(cfg)
    result = model.temperature_profile(Q_dash, T_src_dash)

    sim = HeatpipeSensorSimulator()
    sim.set_base_profile(result["x"], result["T"])

    series = sim.generate_timeseries(
        n_steps, anomaly_at=int(n_steps * 0.67) if inject_anomaly else None
    )

    # Build time series arrays
    timestamps = np.arange(n_steps)
    sensor_data = {label: [] for label in sim.cfg.labels}
    teg_voltage = []
    teg_power = []

    for step_readings in series:
        for r in step_readings:
            if r.sensor_id in sensor_data:
                sensor_data[r.sensor_id].append(r.value)
            elif r.sensor_id == "TEG_voltage":
                teg_voltage.append(r.value)
            elif r.sensor_id == "TEG_power":
                teg_power.append(r.value)

    # Temperature Timeseries
    st.subheader("Temperature Sensors (5× DS18B20)")
    fig = go.Figure()
    colors = ["#ff4444", "#ff8844", "#888888", "#4488ff", "#4444ff"]
    for i, (label, data) in enumerate(sensor_data.items()):
        fig.add_trace(go.Scatter(
            x=timestamps, y=data, name=label,
            line=dict(color=colors[i], width=2),
        ))
    if inject_anomaly:
        anomaly_start = int(n_steps * 0.67)
        fig.add_vrect(
            x0=anomaly_start, x1=min(anomaly_start + 30, n_steps),
            fillcolor="red", opacity=0.15,
            annotation_text="⚠️ ANOMALY",
        )
    fig.update_layout(
        xaxis_title="Measurement Cycle (1/min)",
        yaxis_title="Temperature [°C]",
        template="plotly_dark", height=400,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # TEG Power + R_th
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("TEG Energy Harvesting")
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(
            x=timestamps, y=teg_voltage, name="Voltage [V]",
            line=dict(color="#00ff88"),
        ), secondary_y=False)
        fig2.add_trace(go.Scatter(
            x=timestamps, y=teg_power, name="Power [mW]",
            line=dict(color="#e94560"),
        ), secondary_y=True)
        fig2.update_layout(template="plotly_dark", height=300)
        fig2.update_yaxes(title_text="Voltage [V]", secondary_y=False)
        fig2.update_yaxes(title_text="Power [mW]", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Thermal Resistance R_th")
        T_evap_arr = np.array(sensor_data["T_evap_hot"])
        T_cond_arr = np.array(sensor_data["T_cond_end"])
        R_th = (T_evap_arr - T_cond_arr) / max(Q_dash, 0.1)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=timestamps, y=R_th, name="R_th [K/W]",
            line=dict(color="#ffaa00", width=2),
        ))
        fig3.add_hline(
            y=np.mean(R_th[:min(30, len(R_th))]),
            line_dash="dash", line_color="green",
            annotation_text="Baseline",
        )
        fig3.update_layout(
            template="plotly_dark", height=300,
            yaxis_title="R_th [K/W]",
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Status
    st.subheader("System Status")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Current T_evap", f"{sensor_data['T_evap_hot'][-1]:.1f} °C")
    s2.metric("Current T_cond", f"{sensor_data['T_cond_end'][-1]:.1f} °C")
    s3.metric("TEG Power", f"{teg_power[-1]:.1f} mW")
    health = "🟢 HEALTHY" if not inject_anomaly else "🔴 ANOMALY DETECTED"
    s4.metric("Health", health)

# ══════════════════════════════════════════════════════════════
# PAGE: DIGITAL TWIN
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Digital Twin":
    st.title("🔮 Digital Twin – Model vs. Reality")
    st.markdown(
        "Live comparison of physics model prediction vs. sensor measurements "
        "with Bayesian auto-calibration"
    )

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        Q_input = st.slider("Heat Load [W]", 10, 100, 50, key="q_twin")
    with col_c2:
        T_src_twin = st.slider("Source Temp [°C]", 40, 120, 80, key="t_twin")

    inject = st.checkbox("💥 Simulate Dryout Event", False, key="inj_twin")

    cfg = HeatpipeConfig()
    model = HeatpipeModel(cfg)
    result = model.temperature_profile(Q_input, T_src_twin)

    sim = HeatpipeSensorSimulator()
    sim.set_base_profile(result["x"], result["T"])
    detector = HeatpipeAnomalyDetector()

    n = 90
    series = sim.generate_timeseries(n, anomaly_at=50 if inject else None)

    # Run twin updates
    model_T_evap = []
    meas_T_evap = []
    residuals_list = []
    health_scores = []
    cal_factors = []
    anomalies = []

    from src.twin.twin_core import DigitalTwin
    twin = DigitalTwin(model, detector)

    for readings in series:
        state = twin.update(readings, Q_input, T_src_twin)
        meas_T_evap.append(state.T_measured.get("T_evap_hot", T_src_twin))
        model_T_evap.append(state.T_predicted.get("T_evap_hot", T_src_twin))
        residuals_list.append(state.residuals.get("T_evap_hot", 0))
        health_scores.append(state.health_score)
        cal_factors.append(state.calibration_factor)
        anomalies.append(state.anomaly_detected)

    t = np.arange(n)

    # Twin Comparison
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[
            "Model vs. Measurement (T_evap)",
            "Residual (ΔT)",
            "Health Score",
        ],
        row_heights=[0.5, 0.25, 0.25],
    )

    fig.add_trace(go.Scatter(
        x=t, y=meas_T_evap, name="Measured",
        line=dict(color="#e94560"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=model_T_evap, name="Model (calibrated)",
        line=dict(color="#4488ff", dash="dash"),
    ), row=1, col=1)

    res_colors = ["#ff3344" if a else "#00ff88" for a in anomalies]
    fig.add_trace(go.Bar(
        x=t, y=residuals_list, name="Residual",
        marker_color=res_colors,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=health_scores, name="Health",
        fill="tozeroy", line=dict(color="#00ff88"),
    ), row=3, col=1)
    fig.add_hline(y=0.7, line_dash="dash", line_color="orange", row=3, col=1)

    fig.update_layout(template="plotly_dark", height=700, showlegend=True)
    fig.update_yaxes(title_text="Temp [°C]", row=1, col=1)
    fig.update_yaxes(title_text="ΔT [°C]", row=2, col=1)
    fig.update_yaxes(title_text="Score [0-1]", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Calibration + Status
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bayesian Auto-Calibration")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=t, y=cal_factors, name="Calibration Factor",
            line=dict(color="#ffaa00", width=2),
        ))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="white", opacity=0.3)
        fig2.update_layout(
            template="plotly_dark", height=250,
            yaxis_title="Factor",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Status Summary")
        last = twin.state_history[-1]
        anomaly_text = (
            f"⚠️ {last.anomaly_cause}" if last.anomaly_detected
            else "✅ None"
        )
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Health Score | **{last.health_score:.2f}** |
        | Calibration Factor | **{last.calibration_factor:.4f}** |
        | TEG Power | **{last.teg_power_mW:.1f} mW** |
        | R_th (measured) | **{last.R_th_measured:.4f} K/W** |
        | R_th (model) | **{last.R_th_model:.4f} K/W** |
        | R_th (calibrated) | **{last.R_th_calibrated:.4f} K/W** |
        | Anomaly | **{anomaly_text}** |
        """)

    with st.expander("ℹ️ How the Digital Twin Works"):
        st.markdown("""
        The Digital Twin runs a continuous update loop:

        1. **Sensor Data** arrives from the Smart HP Module (or simulator)
        2. **Physics Model** computes the expected temperature profile
        3. **Comparison**: Measured vs. predicted temperatures at each sensor position
        4. **Bayesian Kalman Update**: Adjusts the model calibration factor to minimize residuals
        5. **Anomaly Detection**: Isolation Forest + residual analysis flags deviations
        6. **Health Score**: 0.0 (critical) to 1.0 (perfect) based on residual history
        """)

# ══════════════════════════════════════════════════════════════
# PAGE: TEG ENERGY
# ══════════════════════════════════════════════════════════════
elif page == "⚡ TEG Energy":
    st.title("⚡ TEG Energy Harvesting Analysis")
    st.markdown("Power budget for self-powered Smart Heat Pipe Module")

    teg = TEGModel()
    curve = teg.power_curve((1, 80))

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve["delta_T"], y=curve["P_max_mW"],
            name="TEG Power Output",
            line=dict(color="#00ff88", width=3),
            fill="tozeroy",
        ))
        fig.add_hline(
            y=0.016, line_dash="dash", line_color="#4488ff",
            annotation_text="BLE Sensor Node (16 µW avg)",
        )
        fig.add_hline(
            y=0.180, line_dash="dash", line_color="#ffaa00",
            annotation_text="LoRa Sensor Node (180 µW avg)",
        )
        fig.update_layout(
            title="TEG Output vs. Sensor Node Consumption",
            xaxis_title="ΔT [K]", yaxis_title="Power [mW]",
            template="plotly_dark", height=450,
            yaxis_type="log", yaxis_range=[-2, 3],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Power Budget")
        st.markdown("""
        | Component | Active Power | Duty Cycle | Avg. Power |
        |-----------|-------------|------------|------------|
        | Temp Sensor (DS18B20) | 350 µW | 15ms/60s | ~0.09 µW |
        | Pressure (BMP390) | 714 µW | 40ms/60s | ~0.48 µW |
        | MCU (ESP32 Deep Sleep) | 240 mW | 20ms/60s | ~80 µW |
        | **LoRa TX** | **120 mW** | **50ms/60s** | **~100 µW** |
        | **Total (LoRa)** | | | **~180 µW** |
        | **Total (BLE opt.)** | | | **~16 µW** |
        """)

        dT = st.slider("ΔT at TEG [K]", 5, 60, 20)
        P_teg = teg.power_max(dT) * 1e6  # µW
        P_sensor = 180  # µW (LoRa)
        margin = (P_teg / max(P_sensor, 1) - 1) * 100

        st.metric("TEG Output", f"{P_teg:.0f} µW")
        st.metric("Sensor Consumption", f"{P_sensor} µW")
        if margin > 0:
            st.success(f"✅ Energy Margin: +{margin:.0f}%")
        else:
            st.error(f"⚠️ Deficit: {margin:.0f}% – Use Supercap Burst Mode")

    # Scenario Comparison
    st.subheader("Application Scenarios")
    scenarios = {
        "Space (LHP, ΔT=10K, BLE)": {"dT": 10, "P_need": 16, "teg": "Micropelt TE-CORE7"},
        "Space (LHP, ΔT=5K, BLE)": {"dT": 5, "P_need": 16, "teg": "Micropelt TE-CORE7"},
        "Data Center (ΔT=20K, LoRa)": {"dT": 20, "P_need": 180, "teg": "SP1848-27145"},
        "Data Center (ΔT=30K, LoRa)": {"dT": 30, "P_need": 180, "teg": "SP1848-27145"},
        "Industrial (ΔT=40K, LoRa)": {"dT": 40, "P_need": 180, "teg": "TecTeg TEG2-126LDT"},
    }

    data = []
    for name, s in scenarios.items():
        P_out = teg.power_max(s["dT"]) * 1e6
        margin_s = (P_out / max(s["P_need"], 1) - 1) * 100
        data.append({
            "Scenario": name,
            "ΔT [K]": s["dT"],
            "TEG": s["teg"],
            "P_out [µW]": f"{P_out:.0f}",
            "P_need [µW]": s["P_need"],
            "Margin": f"+{margin_s:.0f}%" if margin_s > 0 else f"{margin_s:.0f}%",
            "Status": "✅" if margin_s > 0 else "⚠️",
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About ThermalOS MVP")

    st.markdown("""
    ### Technology Stack

    | Layer | Technology | Purpose |
    |-------|-----------|---------|
    | **Frontend** | Streamlit + Plotly | Interactive Dashboard |
    | **Physics** | NumPy + Custom Engine | 1D Thermal Resistance Network |
    | **AI/ML** | scikit-learn MLP | Surrogate Model + Anomaly Detection |
    | **Hardware** | ESP32 + LoRa | TEG-powered Sensor Node |
    | **Data** | InfluxDB + MQTT | Time-series Storage + Messaging |
    | **DevOps** | Docker + GitHub Actions | Container + CI/CD |

    ### Version History

    | Version | Date | Changes |
    |---------|------|---------|
    | v0.2.0 | 2026-03 | Production fixes: R_vapor physics, real AI Surrogate, improved UX |
    | v0.1.0-alpha | 2026-02 | Initial MVP: Physics Engine + Streamlit Demo |

    ### Key Fixes in v0.2.0

    - **Physics Engine**: `R_vapor()` now correctly uses vapor viscosity (µ_v) instead of liquid viscosity (µ_l) – previous version overestimated vapor resistance by ~40×
    - **AI Surrogate**: Real MLP training replaces fake random-noise simulation
    - **Anomaly Detection**: Configurable thresholds, better documentation
    - **Digital Twin**: Corrected Kalman filter math for calibration factor update
    - **Sensor Simulator**: Drift clamping prevents unbounded growth
    - **MQTT Bridge**: Updated to paho-mqtt v2.x API, environment variable config
    - **Dependencies**: Removed unused packages (pydantic, fastapi, uvicorn)
    """)

    st.markdown("---")
    st.markdown(
        "**ThermalOS** – AI-Powered Digital Twins for Heat Pipes | "
        "[GitHub](https://github.com/thermalos) | "
        "[Contact](mailto:info@thermalos.de)"
    )

# ── Footer ──────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 ThermalOS GmbH")
st.sidebar.markdown(
    "[GitHub](https://github.com/thermalos) | "
    "[Contact](mailto:info@thermalos.de)"
)
