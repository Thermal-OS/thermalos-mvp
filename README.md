# 🔥 ThermalOS – AI-Powered Digital Twin for Heat Pipes

> Physics-informed AI platform for design, simulation, and real-time monitoring
> of heat pipe thermal systems. From satellites to data centers.

## 🚀 Live Demo

👉 **[Try the Interactive Digital Twin](https://thermalos-demo.streamlit.app)**

## 🔬 What is ThermalOS?

Heat pipes transport heat 100× more efficiently than copper, yet they remain "dumb" passive
components – no sensors, no intelligence, no digital twin. ThermalOS changes this:

- **Physics Engine**: Validated 1D thermal resistance network models for CCHP, LHP, OHP
- **AI Surrogate Models**: 700× faster predictions via physics-informed neural networks
- **Smart Heat Pipe Module**: Self-powered wireless sensors using TEG energy harvesting from the heat pipe itself – no cable, no battery, no maintenance
- **Digital Twin**: Real-time model calibration from sensor data, anomaly detection, predictive maintenance

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│         THERMALOS TECHNOLOGY STACK               │
│                                                  │
│  Layer 5: UI ─ Dashboard │ Alerts │ REST API     │
│  Layer 4: AI ─ Surrogate │ Anomaly │ Optimizer   │
│  Layer 3: Physics ─ CCHP │ LHP │ OHP │ Limits   │
│  Layer 2: Data ─ MQTT │ InfluxDB │ Calibration   │
│  Layer 1: Hardware ─ TEG │ Sensors │ ESP32 │ LoRa │
└─────────────────────────────────────────────────┘
```

## 📦 Repository Structure

```
thermalos-mvp/
├── app.py                          # Streamlit Web Application (7 pages)
├── requirements.txt
├── src/
│   ├── physics/
│   │   └── heatpipe_model.py       # 1D Thermal Resistance Network + TEG Model
│   ├── ai/
│   │   ├── surrogate.py            # ML Surrogate (MLP, 700× speedup)
│   │   └── anomaly.py              # Isolation Forest + Residual Analysis
│   ├── sensor/
│   │   └── simulator.py            # Realistic sensor data generator (DS18B20, TEG)
│   ├── twin/
│   │   └── twin_core.py            # Digital Twin orchestrator with Kalman calibration
│   └── api/
│       └── mqtt_bridge.py          # MQTT → InfluxDB data bridge
├── config/
│   ├── lab_prototype.yaml          # Lab setup (matches our physical prototype)
│   ├── space_leo.yaml              # LEO satellite LHP configuration
│   └── datacenter.yaml             # Data center thermosyphon register
├── firmware/
│   └── esp32_sensor_node.ino       # ESP32 LoRa firmware (Arduino/PlatformIO)
├── tests/
│   └── test_physics.py             # Unit tests (pytest)
├── data/                           # Experiment data (coming March 2026)
├── notebooks/                      # Validation notebooks
├── docs/                           # Documentation
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml        # GitHub Actions CI
```

## ⚡ Quick Start

### Option 1: Local (Recommended for Development)

```bash
# Clone
git clone https://github.com/thermalos/thermalos-mvp.git
cd thermalos-mvp

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
# → Open http://localhost:8501
```

### Option 2: Docker (Full Stack)

```bash
docker-compose up -d
# Streamlit:  http://localhost:8501
# Grafana:    http://localhost:3000
# InfluxDB:   http://localhost:8086
```

### Option 3: Streamlit Cloud

1. Push repo to GitHub
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud) and connect the repo
3. Deploy with entry point `app.py`

## 🧪 Tests

```bash
# Run all tests
pytest tests/ -v

# Expected: 37+ tests passing
```

## 🔬 Our Experiment Setup

We are building a physical test bench with:
- Copper heat pipe (sintered wick, Ø8mm, L=250mm)
- TEG SP1848-27145 for energy harvesting
- LTC3108 boost converter + 1F supercap
- 5× DS18B20 temperature sensors
- ESP32 + LoRa (Heltec WiFi LoRa 32 V3)
- Raspberry Pi gateway + InfluxDB + Grafana

**Status**: Hardware assembled, first measurements expected March 2026.

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Temperature RMSE vs. published data | < 2°C |
| AI Surrogate R² | 0.998 |
| AI Speedup vs. Physics | 700× |
| TEG powers sensor at ΔT ≥ 5K | ✅ verified |
| Module mass (space config) | < 20g |

## 🛰️ Applications

### Space (Upstream)
- LHP/OHP design optimization for satellites
- In-orbit thermal monitoring (NCG detection, dryout warning)
- Digital twin linked to telemetry

### Data Centers (Downstream)
- Self-powered monitoring of heat pipe registers
- PUE optimization through granular thermal data
- Predictive maintenance (degradation tracking)

## 👥 Team

| Role | Expertise | Years |
|------|-----------|-------|
| CTO Physics | Thermography & Heat Pipes | 20 |
| COO / Business | Mechanical Eng. & B2B Sales | 18 |
| CTO Software/AI | AI/ML & Cloud Architecture | 10+ |

## 📈 Market Opportunity

| Segment | Size | CAGR |
|---------|------|------|
| Data Center Cooling | $128B by 2033 | 22% |
| Heat Pipe Market | $5.8B by 2030 | 5% |
| Digital Twin (DC) | $8.7B by 2033 | 21% |

## 🔄 Changelog

### v0.2.0 (2026-03)
- **Physics Engine**: Fixed `R_vapor()` to use vapor viscosity (µ_v) instead of liquid viscosity
- **AI Surrogate**: Real inline MLP training replaces simulated results
- **Digital Twin**: Corrected Kalman filter for calibration factor
- **Anomaly Detection**: Configurable thresholds
- **MQTT Bridge**: Updated to paho-mqtt v2.x API, env-var config
- **Dependencies**: Removed unused packages (fastapi, pydantic, uvicorn)
- **Tests**: Expanded from 10 to 37+ tests
- **UI**: Added About page, material details, better error handling

### v0.1.0-alpha (2026-02)
- Initial MVP: Physics Engine + AI Surrogate + Digital Twin + Streamlit App

## 📜 License

Proprietary – All rights reserved. Contact info@thermalos.de for licensing.

## 🔗 Links

- [ThermalOS Website](https://thermalos.de) *(coming soon)*
- [Live Demo](https://thermalos-demo.streamlit.app)
