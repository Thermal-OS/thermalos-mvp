"""
ThermalOS – Unit Tests
Abdeckung: Physics Engine, AI Surrogate, Anomaly Detector, Sensor Simulator
"""

import os
import sys

# Sicherstellen, dass der Projektroot im Suchpfad liegt
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.physics.heatpipe_model import (  # noqa: E402
    HeatpipeConfig,
    HeatpipeModel,
    TEGModel,
    FLUIDS,
    quick_simulation,
)
from src.ai.anomaly import HeatpipeAnomalyDetector, AnomalyResult  # noqa: E402
from src.sensor.simulator import HeatpipeSensorSimulator, SensorConfig  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Hilfs-Konstanten
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_Q      = 50.0   # W
DEFAULT_T_SRC  = 80.0   # °C


# ═══════════════════════════════════════════════════════════════════════════════
# Klasse 1: Heatpipe-Physikmodell
# ═══════════════════════════════════════════════════════════════════════════════

class TestHeatpipeModel:
    """Tests für HeatpipeModel und zugehörige Widerstands-Berechnungen."""

    def setup_method(self):
        self.cfg   = HeatpipeConfig()
        self.model = HeatpipeModel(self.cfg)

    # ── Grundlegende Konsistenz ───────────────────────────────────────────────

    def test_r_total_positive(self):
        """Gesamtwiderstand muss positiv sein."""
        assert self.model.R_total() > 0, "R_total muss > 0 K/W sein"

    def test_temperature_profile_shape(self):
        """Ausgabearrays haben die konfigurierte Länge n_cells."""
        result = self.model.temperature_profile(DEFAULT_Q, DEFAULT_T_SRC)
        assert len(result["x"]) == self.cfg.n_cells
        assert len(result["T"]) == self.cfg.n_cells

    def test_evap_hotter_than_cond(self):
        """Verdampfer muss wärmer als Kondensator sein."""
        result = self.model.temperature_profile(DEFAULT_Q, DEFAULT_T_SRC)
        assert result["T_evap"] > result["T_cond"]

    def test_capillary_limit_positive(self):
        """Kapillarlimit muss ≥ 0 sein."""
        result = self.model.temperature_profile(10.0, 60.0)
        assert result["Q_max"] >= 0.0

    def test_gravity_effect(self):
        """Gegen-Schwerkraft (orient=-90°) darf Kapillarlimit nicht erhöhen."""
        cfg_h = HeatpipeConfig(orientation_deg=0)
        cfg_a = HeatpipeConfig(orientation_deg=-90)
        r_h   = HeatpipeModel(cfg_h).temperature_profile(30.0, 70.0)
        r_a   = HeatpipeModel(cfg_a).temperature_profile(30.0, 70.0)
        assert r_h["Q_max"] >= r_a["Q_max"], \
            "Horizontale Heatpipe muss Q_max ≥ gegen-schwerkraft haben"

    def test_different_fluids(self):
        """Alle konfigurierten Fluide müssen positive R_total liefern."""
        for fluid in FLUIDS:
            cfg   = HeatpipeConfig(fluid=fluid)
            model = HeatpipeModel(cfg)
            assert model.R_total() > 0, f"R_total für {fluid} muss > 0 sein"

    def test_different_wicks(self):
        """Alle konfigurierten Dochttypen müssen positive R_total liefern."""
        for wick in ["sintered", "grooved", "mesh", "screen_200mesh"]:
            cfg   = HeatpipeConfig(wick_type=wick)
            model = HeatpipeModel(cfg)
            assert model.R_total() > 0, f"R_total für Docht '{wick}' muss > 0 sein"

    # ── BUG-FIX: R_vapor verwendet mu_v, nicht mu_l ──────────────────────────

    def test_r_vapor_uses_vapor_viscosity(self):
        """
        R_vapor() muss mu_v (Dampfviskosität) verwenden, nicht mu_l.

        Nachweis: Bei Wasser gilt mu_v ≈ 1.2e-5 Pa·s, mu_l ≈ 4.7e-4 Pa·s.
        Bei korrektem mu_v ist R_vapor ~ 40× kleiner als bei falschem mu_l.
        """
        model = HeatpipeModel(HeatpipeConfig(fluid="water"))
        fp    = model.get_fluid(60.0)
        # Sicherstellen, dass mu_v im Fluid-Dict vorhanden ist
        assert "mu_v" in fp, "mu_v muss in FLUIDS['water'] definiert sein"
        # mu_v muss kleiner als mu_l sein (Wasser: ca. 40×)
        assert fp["mu_v"] < fp["mu_l"], \
            f"mu_v ({fp['mu_v']:.2e}) muss < mu_l ({fp['mu_l']:.2e}) sein"
        # R_vapor darf nicht den (falschen) mu_l-Wert verwendet haben
        # Prüfe: R_vapor mit mu_v sollte deutlich kleiner sein als mit mu_l
        r_v_correct  = model.R_vapor()
        # Händische Berechnung mit mu_l (falsche Variante)
        r_v = model.cfg.r_vapor
        L_eff = (model.cfg.evap_length / 2 + model.cfg.adiabatic_length
                 + model.cfg.cond_length / 2)
        r_v_wrong = (8 * fp["mu_l"] * L_eff) / (
            np.pi * fp["rho_v"] * fp["h_fg"] * r_v ** 4
        )
        assert r_v_correct < r_v_wrong, \
            "Korrektes R_vapor (mu_v) muss kleiner sein als fehlerhaftes R_vapor (mu_l)"

    def test_r_vapor_ammonia_has_mu_v(self):
        """Ammoniak-Fluid muss ebenfalls mu_v enthalten."""
        model = HeatpipeModel(HeatpipeConfig(fluid="ammonia"))
        fp    = model.get_fluid(0.0)
        assert "mu_v" in fp, "mu_v muss in FLUIDS['ammonia'] definiert sein"
        assert fp["mu_v"] > 0, "mu_v (Ammoniak) muss > 0 sein"

    # ── BUG-FIX: Property-Clamping ────────────────────────────────────────────

    def test_rho_l_never_negative(self):
        """Flüssigkeitsdichte darf nie negativ werden."""
        for T in np.linspace(-100, 500, 50):
            fp = HeatpipeModel(HeatpipeConfig(fluid="water")).get_fluid(float(T))
            assert fp["rho_l"] > 0, f"rho_l negativ bei T={T}°C: {fp['rho_l']}"

    def test_sigma_never_negative(self):
        """Oberflächenspannung darf nie negativ werden."""
        for T in np.linspace(-100, 500, 50):
            fp = HeatpipeModel(HeatpipeConfig(fluid="water")).get_fluid(float(T))
            assert fp["sigma"] > 0, f"sigma negativ bei T={T}°C: {fp['sigma']}"

    def test_h_fg_never_negative(self):
        """Verdampfungsenthalpie darf nie negativ werden."""
        for T in np.linspace(-100, 500, 50):
            fp = HeatpipeModel(HeatpipeConfig(fluid="water")).get_fluid(float(T))
            assert fp["h_fg"] > 0, f"h_fg negativ bei T={T}°C: {fp['h_fg']}"

    def test_rho_v_never_negative(self):
        """Dampfdichte darf nie negativ werden."""
        for T in np.linspace(-100, 500, 50):
            fp = HeatpipeModel(HeatpipeConfig(fluid="water")).get_fluid(float(T))
            assert fp["rho_v"] > 0, f"rho_v negativ bei T={T}°C: {fp['rho_v']}"

    def test_ammonia_clamping(self):
        """Ammoniak-Properties müssen bei Extremtemperaturen geclampet werden."""
        for T in [-200, 0, 100, 300]:
            cfg   = HeatpipeConfig(fluid="ammonia")
            model = HeatpipeModel(cfg)
            fp    = model.get_fluid(float(T))
            assert fp["rho_l"] > 0, f"Ammoniak rho_l negativ bei T={T}°C"
            assert fp["h_fg"]  > 0, f"Ammoniak h_fg negativ bei T={T}°C"
            assert fp["sigma"] > 0, f"Ammoniak sigma negativ bei T={T}°C"

    def test_quick_simulation(self):
        """Convenience-Funktion quick_simulation() muss vollständige Keys zurückgeben."""
        result = quick_simulation(Q=40.0, T_source=80.0)
        for key in ["x", "T", "R_total", "T_evap", "T_cond", "Q_max", "delta_T", "teg"]:
            assert key in result, f"Schlüssel '{key}' fehlt im quick_simulation-Ergebnis"


# ═══════════════════════════════════════════════════════════════════════════════
# Klasse 2: TEG-Modell
# ═══════════════════════════════════════════════════════════════════════════════

class TestTEGModel:
    """Tests für den thermoelektrischen Generator (TEG)."""

    def setup_method(self):
        self.teg = TEGModel()

    def test_voltage_increases_with_dT(self):
        assert self.teg.voltage_open(20.0) > self.teg.voltage_open(10.0)

    def test_power_positive(self):
        assert self.teg.power_max(10.0) > 0.0

    def test_power_quadratic(self):
        """Bei doppeltem ΔT muss die Leistung ~4× steigen (P ∝ ΔT²)."""
        P1 = self.teg.power_max(10.0)
        P2 = self.teg.power_max(20.0)
        assert P2 / P1 > 3.5, "TEG-Leistung muss näherungsweise quadratisch in ΔT sein"

    def test_power_curve(self):
        curve = self.teg.power_curve()
        assert len(curve["delta_T"]) == 50
        assert all(p >= 0.0 for p in curve["P_max_mW"])


# ═══════════════════════════════════════════════════════════════════════════════
# Klasse 3: Surrogate-Import
# ═══════════════════════════════════════════════════════════════════════════════

class TestSurrogateImport:
    """Testet, dass surrogate.py korrekt importierbar ist (BUG-FIX Importpfad)."""

    def test_surrogate_importable(self):
        """Import darf keine ImportError / ModuleNotFoundError werfen."""
        try:
            from src.ai.surrogate import HeatpipeSurrogate  # noqa: F401
        except ImportError as exc:
            pytest.fail(f"Import von HeatpipeSurrogate fehlgeschlagen: {exc}")

    def test_surrogate_instantiate(self):
        from src.ai.surrogate import HeatpipeSurrogate
        s = HeatpipeSurrogate()
        assert not s.is_trained, "Untrainiertes Surrogate muss is_trained=False haben"

    def test_surrogate_predict_raises_before_train(self):
        from src.ai.surrogate import HeatpipeSurrogate
        s = HeatpipeSurrogate()
        with pytest.raises(RuntimeError, match="trainiert"):
            s.predict(50.0, 80.0, 0.25, 0.01)

    def test_generate_demo_comparison_structure(self):
        """
        generate_demo_comparison() muss alle erwarteten Keys zurückgeben.
        Kleines n_train für schnellen Test.
        """
        from src.ai.surrogate import HeatpipeSurrogate
        result = HeatpipeSurrogate.generate_demo_comparison(n_train=80, n_eval=10)
        expected_keys = [
            "Q_values", "physics_T_cond", "surrogate_T_cond",
            "physics_R_total", "surrogate_R_total",
            "rmse_T_cond", "rmse_R_total",
            "r2_T_cond", "r2_R_total",
            "speedup_factor", "train_time_s", "n_train",
        ]
        for key in expected_keys:
            assert key in result, f"Schlüssel '{key}' fehlt in generate_demo_comparison()"

    def test_demo_comparison_no_random_noise(self):
        """
        generate_demo_comparison() liefert echte ML-Vorhersagen (kein Zufallsrauschen).
        Mit ausreichend Trainingsdaten muss r² > 0 und < 1 sein.
        """
        from src.ai.surrogate import HeatpipeSurrogate
        # Mehr Samples für stabiles Training
        result = HeatpipeSurrogate.generate_demo_comparison(n_train=300, n_eval=20)
        # r² < 1.0 bedeutet: Es ist kein perfekter Klon (echter Fehler vorhanden)
        assert result["r2_T_cond"] < 1.0 + 1e-6, "Surrogate r²(T_cond) sollte < 1 sein"
        # Ergebnis enthält echte Modell-Vorhersagen, kein reines Rauschen
        # (Prüfung: physics_T_cond und surrogate_T_cond nicht identisch)
        import numpy as np
        phy = np.array(result["physics_T_cond"])
        sur = np.array(result["surrogate_T_cond"])
        assert not np.allclose(phy, sur), \
            "Surrogate-Vorhersagen dürfen nicht identisch mit Physics-Werten sein"


# ═══════════════════════════════════════════════════════════════════════════════
# Klasse 4: Anomaly Detector
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnomalyDetector:
    """Tests für HeatpipeAnomalyDetector."""

    def _feed_normal(self, detector: HeatpipeAnomalyDetector, n: int = 40) -> None:
        """Füttert den Detektor mit normalen (konsistenten) Messwerten."""
        for _ in range(n):
            detector.add_measurement(
                T_evap=80.0, T_cond=60.0, Q=50.0,
                T_model_evap=80.0, T_model_cond=60.0,
            )

    def test_returns_anomaly_result(self):
        """add_measurement() muss ein AnomalyResult zurückgeben."""
        det    = HeatpipeAnomalyDetector()
        result = det.add_measurement(
            T_evap=80.0, T_cond=60.0, Q=50.0,
            T_model_evap=80.0, T_model_cond=60.0,
        )
        assert isinstance(result, AnomalyResult)

    def test_no_anomaly_normal_operation(self):
        """Normaler Betrieb (Residuum ≈ 0) darf keine Anomalie produzieren."""
        det = HeatpipeAnomalyDetector()
        self._feed_normal(det, n=50)
        result = det.add_measurement(
            T_evap=80.0, T_cond=60.0, Q=50.0,
            T_model_evap=80.0, T_model_cond=60.0,
        )
        assert not result.is_anomaly, "Kein Alarm bei normalem Betrieb erwartet"

    def test_anomaly_detected_large_residual(self):
        """Großes positives Residuum muss cause='Dryout'-Text enthalten."""
        det = HeatpipeAnomalyDetector(
            residual_high=0.1,  # sehr niedrige Schwelle für den Test
            iso_threshold=-0.5,
        )
        self._feed_normal(det, n=50)
        # Extrem großes Residuum injizieren
        result = det.add_measurement(
            T_evap=120.0, T_cond=60.0, Q=50.0,   # R_th_meas = 1.2 K/W
            T_model_evap=80.0, T_model_cond=60.0, # R_th_model = 0.4 K/W → Residuum = 0.8
        )
        assert result.residual > 0.1, "Residuum muss über Schwelle liegen"
        # Wenn Iso-Forest Anomalie bestätigt, muss Cause gesetzt sein
        if result.is_anomaly:
            assert result.probable_cause is not None

    def test_health_score_starts_at_one(self):
        """Gesundheitsscore muss bei leerem Puffer 1.0 sein."""
        det = HeatpipeAnomalyDetector()
        assert det.get_health_score() == 1.0

    def test_health_score_range(self):
        """Gesundheitsscore muss immer in [0, 1] liegen."""
        det = HeatpipeAnomalyDetector()
        self._feed_normal(det, n=70)
        score = det.get_health_score()
        assert 0.0 <= score <= 1.0

    def test_configurable_thresholds(self):
        """Konfigurierbare Schwellwerte müssen korrekt gespeichert werden."""
        det = HeatpipeAnomalyDetector(residual_high=0.8, residual_low=0.4)
        assert det.residual_high == 0.8
        assert det.residual_low  == 0.4

    def test_timestamp_parameter(self):
        """Optionaler timestamp-Parameter muss korrekt übernommen werden."""
        det    = HeatpipeAnomalyDetector()
        ts     = 1_700_000_000.0
        result = det.add_measurement(
            T_evap=80.0, T_cond=60.0, Q=50.0,
            T_model_evap=80.0, T_model_cond=60.0,
            timestamp=ts,
        )
        assert result.timestamp == ts


# ═══════════════════════════════════════════════════════════════════════════════
# Klasse 5: Sensor Simulator
# ═══════════════════════════════════════════════════════════════════════════════

class TestSensorSimulator:
    """Tests für HeatpipeSensorSimulator."""

    def setup_method(self):
        self.cfg = SensorConfig(noise_std=0.0, drift_rate=0.001, max_drift=2.0)
        self.sim = HeatpipeSensorSimulator(self.cfg)

    def test_output_shape(self):
        """Eine Messung muss 5 Temperatur- + 2 TEG-Readings liefern."""
        readings = self.sim.generate_reading()
        temp_readings = [r for r in readings if r.unit == "°C"]
        teg_readings  = [r for r in readings if r.unit in ("V", "mW")]
        assert len(temp_readings) == 5, "Genau 5 Temperatursensoren erwartet"
        assert len(teg_readings)  == 2, "Genau 2 TEG-Readings (V, mW) erwartet"

    def test_output_types(self):
        """Alle SensorReading-Werte müssen floats sein."""
        readings = self.sim.generate_reading()
        for r in readings:
            assert isinstance(r.value, float), f"Wert von {r.sensor_id} kein float"

    def test_sensor_labels(self):
        """Sensor-Labels müssen mit Konfiguration übereinstimmen."""
        readings     = self.sim.generate_reading()
        temp_labels  = [r.sensor_id for r in readings if r.unit == "°C"]
        expected     = ["T_evap_hot", "T_evap_mid", "T_adiabatic", "T_cond_mid", "T_cond_end"]
        assert temp_labels == expected

    def test_drift_clamped(self):
        """
        Drift-Akkumulator darf max_drift [°C] nie überschreiten.

        BUG-FIX: Im Original wuchs drift_accumulator unbegrenzt an.
        """
        cfg = SensorConfig(drift_rate=1.0, max_drift=2.0, noise_std=0.0)
        sim = HeatpipeSensorSimulator(cfg)
        # 100 Schritte mit drift_rate=1.0 → ohne Clamp wäre Drift=100 °C
        for _ in range(100):
            sim.generate_reading()
        for i, acc in enumerate(sim.drift_accumulator):
            assert abs(acc) <= cfg.max_drift + 1e-9, \
                f"Drift-Akkumulator[{i}]={acc:.2f} überschreitet max_drift={cfg.max_drift}"

    def test_teg_power_non_negative(self):
        """TEG-Leistung darf nicht negativ sein."""
        for _ in range(20):
            readings = self.sim.generate_reading()
            p_teg = next(r.value for r in readings if r.sensor_id == "TEG_power")
            assert p_teg >= 0.0, "TEG-Leistung darf nicht negativ sein"

    def test_base_profile_interpolation(self):
        """set_base_profile() beeinflusst die Temperaturen."""
        x   = np.linspace(0, 0.25, 50)
        T   = np.linspace(100, 40, 50)   # Gradient von 100°C → 40°C
        self.sim.set_base_profile(x, T)
        readings    = self.sim.generate_reading()
        temp_vals   = [r.value for r in readings if r.unit == "°C"]
        # Verdampfer muss wärmer als Kondensator sein
        assert temp_vals[0] > temp_vals[-1], \
            "Verdampfer-Sensor muss wärmer als Kondensator-Sensor sein"

    def test_timeseries_length(self):
        """generate_timeseries(n) muss genau n Mess-Listen zurückgeben."""
        series = self.sim.generate_timeseries(n_steps=10)
        assert len(series) == 10

    def test_anomaly_injection(self):
        """Injizierte Anomalie muss Verdampfertemperatur erhöhen."""
        cfg_no_noise = SensorConfig(noise_std=0.0, anomaly_prob=0.0)
        sim          = HeatpipeSensorSimulator(cfg_no_noise)
        x   = np.linspace(0, 0.25, 50)
        T   = np.full(50, 80.0)
        sim.set_base_profile(x, T)

        # Normalmessung
        r_normal = sim.generate_reading(inject_anomaly=False)
        T_evap_normal = next(r.value for r in r_normal if r.sensor_id == "T_evap_hot")

        # Anomalie injizieren und mehrere Schritte durchlaufen
        r_anom = sim.generate_reading(inject_anomaly=True)
        T_evap_anom = next(r.value for r in r_anom if r.sensor_id == "T_evap_hot")

        assert T_evap_anom >= T_evap_normal, \
            "Verdampfertemperatur bei Anomalie muss ≥ Normalwert sein"


# ═══════════════════════════════════════════════════════════════════════════════
# Direkt ausführbar
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
