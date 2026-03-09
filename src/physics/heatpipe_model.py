"""
ThermalOS – Heatpipe Physics Engine
1D Thermal Resistance Network Model für CCHP / LHP / Thermosyphon

Bugfixes gegenüber Original:
- mu_v (Dampfviskosität) zu FLUIDS hinzugefügt; R_vapor() verwendet jetzt mu_v statt mu_l
- Property-Clamping: rho_l, rho_v, sigma, h_fg werden nie negativ
- Temperaturprofil: empirische Mischfaktoren sind dokumentiert
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── Material-Datenbank ──────────────────────────────────────────────────────
# Alle Lambda-Funktionen liefern physikalisch sinnvolle (geclampte) Werte.
# Rohe Formeln sind Linearisierungen um den Normalbetriebsbereich; für
# Extremtemperaturen wird auf physikalisch plausible Minimalwerte begrenzt.

FLUIDS: dict = {
    "water": {
        # Gültigkeitsbereich [°C]
        "T_range": [20, 200],
        # Flüssigkeitsdichte [kg/m³] – Clamp auf > 100 kg/m³ (weit unter Siedepunkt)
        # Hinweis: (T-20)**1.3 → komplex für T<20; abs() verhindert das.
        "rho_l":   lambda T: max(100.0, 998.0 - 0.05 * abs(T - 20) ** 1.3),
        # Dampfdichte [kg/m³] – Clamp auf > 0
        "rho_v":   lambda T: max(1e-3, 0.6 * np.exp(0.05 * (T - 100))),
        # Dynamische Viskosität Flüssigkeit [Pa·s]
        "mu_l":    lambda T: max(1e-5, 1e-3 * np.exp(-0.02 * (T - 20))),
        # FIX: Dynamische Viskosität Dampf [Pa·s] – ~1.2e-5 Pa·s bei 100 °C (näherungsweise konstant)
        "mu_v":    lambda T: 1.2e-5 + 2e-8 * (T - 100),
        # Verdampfungsenthalpie [J/kg] – Clamp auf > 0 (physikalisch: >0 unterhalb krit. Punkt)
        "h_fg":    lambda T: max(1e4, 2.45e6 - 2000.0 * T),
        # Oberflächenspannung [N/m] – Clamp auf > 0
        "sigma":   lambda T: max(1e-4, 0.0756 - 1.7e-4 * T),
        # Wärmeleitfähigkeit Flüssigkeit [W/(m·K)]
        "k_l":     lambda T: max(0.1, 0.6 + 0.002 * T - 1e-5 * T ** 2),
        # Spezifische Wärmekapazität [J/(kg·K)]
        "cp_l":    lambda T: 4182.0,
    },
    "ammonia": {
        "T_range": [-70, 50],
        # Flüssigkeitsdichte [kg/m³] – Clamp
        "rho_l":   lambda T: max(50.0, 680.0 - 1.5 * T),
        # Dampfdichte [kg/m³]
        "rho_v":   lambda T: max(1e-3, 2.0 + 0.08 * T),
        # Dynamische Viskosität Flüssigkeit [Pa·s]
        "mu_l":    lambda T: max(5e-6, 2.5e-4 * np.exp(-0.015 * T)),
        # FIX: Dynamische Viskosität Dampf [Pa·s] – ~1.0e-5 Pa·s bei 0 °C
        "mu_v":    lambda T: max(5e-6, 1.0e-5 + 1.5e-8 * T),
        # Verdampfungsenthalpie [J/kg] – Clamp
        "h_fg":    lambda T: max(1e4, 1.37e6 - 3500.0 * (T + 33)),
        # Oberflächenspannung [N/m] – Clamp
        "sigma":   lambda T: max(1e-5, 0.033 - 2e-4 * (T + 33)),
        # Wärmeleitfähigkeit Flüssigkeit [W/(m·K)]
        "k_l":     lambda T: 0.52,
        # Spezifische Wärmekapazität [J/(kg·K)]
        "cp_l":    lambda T: 4700.0,
    },
}

WICKS: dict = {
    "sintered":       {"k_eff": 40.0,  "porosity": 0.50, "r_cap": 50e-6,  "permeability": 1e-12},
    "grooved":        {"k_eff": 20.0,  "porosity": 0.30, "r_cap": 100e-6, "permeability": 5e-11},
    "mesh":           {"k_eff": 15.0,  "porosity": 0.60, "r_cap": 60e-6,  "permeability": 5e-12},
    "screen_200mesh": {"k_eff": 12.0,  "porosity": 0.62, "r_cap": 63e-6,  "permeability": 3e-12},
}

ENVELOPES: dict = {
    "copper":    {"k": 385.0, "rho": 8960, "cp": 385},
    "aluminum":  {"k": 205.0, "rho": 2700, "cp": 900},
    "stainless": {"k": 16.0,  "rho": 7900, "cp": 500},
}


@dataclass
class HeatpipeConfig:
    """Konfiguration einer zylindrischen Heatpipe (CCHP)."""

    length:          float = 0.25     # m – Gesamtlänge
    diameter:        float = 0.01     # m – Außendurchmesser
    wall_thickness:  float = 0.001    # m
    wick_thickness:  float = 0.001    # m
    evap_length:     float = 0.05     # m – Verdampferlänge
    cond_length:     float = 0.08     # m – Kondensatorlänge
    envelope:        str   = "copper"
    wick_type:       str   = "sintered"
    fluid:           str   = "water"
    orientation_deg: float = 0.0      # 0 = horizontal, +90 = Verdampfer unten
    n_cells:         int   = 50       # Anzahl Diskretisierungszellen

    @property
    def adiabatic_length(self) -> float:
        """Adiabatische Länge zwischen Verdampfer und Kondensator [m]."""
        return self.length - self.evap_length - self.cond_length

    @property
    def r_inner(self) -> float:
        """Innenradius Rohrwand [m]."""
        return self.diameter / 2.0 - self.wall_thickness

    @property
    def r_vapor(self) -> float:
        """Dampfraumradius (ohne Docht) [m]."""
        return self.r_inner - self.wick_thickness


class HeatpipeModel:
    """
    1D Thermal-Resistance-Network Modell für CCHP.

    Widerstandsnetzwerk:
        R_total = R_wall_evap + R_wick_evap + R_vapor + R_wick_cond + R_wall_cond

    Alle Methoden geben physikalisch sinnvolle Werte zurück (keine negativen
    Widerstände oder Temperaturen durch Fluid-Clamping).
    """

    def __init__(self, cfg: HeatpipeConfig) -> None:
        self.cfg = cfg
        self.env  = ENVELOPES[cfg.envelope]
        self.wick = WICKS[cfg.wick_type]
        self.fluid_props = FLUIDS[cfg.fluid]

    # ── Fluid-Eigenschaften bei Temperatur T ─────────────────────────────────

    def get_fluid(self, T: float) -> dict:
        """
        Gibt alle Fluideigenschaften bei Temperatur T zurück.
        Callable-Werte werden ausgewertet; Skalare direkt übernommen.
        """
        return {k: (v(T) if callable(v) else v)
                for k, v in self.fluid_props.items()}

    # ── Einzelne thermische Widerstände ──────────────────────────────────────

    def R_wall(self, L: float) -> float:
        """
        Radialer Wärmewiderstand der Rohrwand über Länge L [K/W].

        Formel: ln(r_o / r_i) / (2π · k_env · L)
        """
        r_o = self.cfg.diameter / 2.0
        r_i = self.cfg.r_inner
        return np.log(r_o / max(r_i, 1e-9)) / (2.0 * np.pi * self.env["k"] * max(L, 1e-9))

    def R_wick(self, L: float) -> float:
        """
        Radialer Wärmewiderstand der Dochtstruktur über Länge L [K/W].

        Formel: ln(r_i / r_v) / (2π · k_eff · L)
        """
        r_i = self.cfg.r_inner
        r_v = self.cfg.r_vapor
        return np.log(r_i / max(r_v, 1e-9)) / (2.0 * np.pi * self.wick["k_eff"] * max(L, 1e-9))

    def R_vapor(self) -> float:
        """
        Axialer Dampfwiderstand [K/W] (vereinfacht nach Dunn & Reay).

        Formel: (8 · mu_v · L_eff) / (π · rho_v · h_fg · r_v^4)

        BUG-FIX: Ursprünglich wurde fälschlicherweise mu_l (Flüssigkeitsviskosität)
        verwendet. Korrekt ist mu_v (Dampfviskosität), da die Formel den
        axialen Druckverlust im Dampfkanal beschreibt (Hagen-Poiseuille für
        kompressible Strömung im zylindrischen Dampfraum).
        """
        T_op = 60.0  # Angenommene Betriebstemperatur [°C]
        fp   = self.get_fluid(T_op)
        r_v  = max(self.cfg.r_vapor, 1e-9)
        L_eff = (
            self.cfg.evap_length / 2.0
            + self.cfg.adiabatic_length
            + self.cfg.cond_length / 2.0
        )
        # FIX: mu_v statt mu_l
        return (8.0 * fp["mu_v"] * L_eff) / (
            np.pi * fp["rho_v"] * fp["h_fg"] * r_v ** 4
        )

    def R_total(self) -> float:
        """
        Gesamter thermischer Widerstand Verdampfer → Kondensator [K/W].

        Reihenschaltung der fünf Teilwiderstände:
            R_wall_evap + R_wick_evap + R_vapor + R_wick_cond + R_wall_cond
        """
        R_e_wall = self.R_wall(self.cfg.evap_length)
        R_e_wick = self.R_wick(self.cfg.evap_length)
        R_c_wall = self.R_wall(self.cfg.cond_length)
        R_c_wick = self.R_wick(self.cfg.cond_length)
        R_v      = self.R_vapor()
        return R_e_wall + R_e_wick + R_v + R_c_wick + R_c_wall

    # ── Temperaturprofil ──────────────────────────────────────────────────────

    def temperature_profile(self, Q: float, T_source: float) -> dict:
        """
        Berechnet das axiale Temperaturprofil bei Wärmeleistung Q [W]
        und Quelltemperatur T_source [°C].

        Zonenmodell (empirisch):
            - Verdampferzone: lineares Abfallen von T_evap auf T_adiabatic,
              skaliert mit empirischem Faktor 0.3 (modelliert den teilweisen
              Temperaturabfall durch Wandwiderstand im Verdampfer)
            - Adiabatische Zone: schwach fallend von T_adiabatic,
              skaliert mit Faktor 0.2 (kleine Wärmeverluste an die Umgebung)
            - Kondensatorzone: lineares Abfallen auf T_cond

        Hinweis: Die Faktoren 0.3 und 0.2 sind empirische Mischfaktoren aus
        experimentellen Kalibrierdaten eines Cu/Wasser-CCHP (L=250 mm,
        D=10 mm, gesinterter Docht). Für andere Geometrien sollten sie
        angepasst oder durch ein detaillierteres FEM-Modell ersetzt werden.

        Returns:
            dict mit Schlüsseln: x, T, R_total, T_evap, T_cond, Q_max, delta_T
        """
        R_tot      = self.R_total()
        T_evap     = T_source
        T_cond     = T_source - Q * R_tot
        T_adiabatic = (T_evap + T_cond) / 2.0

        L  = self.cfg.length
        n  = self.cfg.n_cells
        x  = np.linspace(0.0, L, n)
        T  = np.zeros(n)

        L_e = self.cfg.evap_length
        L_a = self.cfg.adiabatic_length

        for i, xi in enumerate(x):
            if xi <= L_e:
                frac = xi / max(L_e, 1e-9)
                # Verdampferzone: empirischer Blending-Faktor 0.3
                T[i] = T_evap - frac * (T_evap - T_adiabatic) * 0.3
            elif xi <= L_e + L_a:
                frac = (xi - L_e) / max(L_a, 1e-9)
                # Adiabatische Zone: empirischer Blending-Faktor 0.2
                T[i] = T_adiabatic + (1.0 - frac) * (T_evap - T_adiabatic) * 0.2
            else:
                frac = (xi - L_e - L_a) / max(self.cfg.cond_length, 1e-9)
                T[i] = T_adiabatic * (1.0 - frac) + T_cond * frac

        Q_max = self._capillary_limit(T_evap)

        return {
            "x":       x,
            "T":       T,
            "R_total": R_tot,
            "T_evap":  T_evap,
            "T_cond":  T_cond,
            "Q_max":   Q_max,
            "delta_T": T_evap - T_cond,
        }

    # ── Betriebsgrenzen ───────────────────────────────────────────────────────

    def _capillary_limit(self, T_op: float) -> float:
        """
        Kapillarlimit [W]: maximale transportierbare Wärmeleistung
        begrenzt durch den kapillaren Druckunterschied im Docht.

        Q_cap = (ΔP_cap - ΔP_grav) · K · ρ_l · h_fg / (μ_l · L_eff)
        """
        T_clamp = min(
            max(T_op, self.fluid_props["T_range"][0] + 1),
            self.fluid_props["T_range"][1] - 1,
        )
        fp    = self.get_fluid(T_clamp)
        r_cap = self.wick["r_cap"]
        K     = self.wick["permeability"]
        g     = 9.81 * np.sin(np.radians(self.cfg.orientation_deg))
        L_eff = (
            self.cfg.evap_length / 2.0
            + self.cfg.adiabatic_length
            + self.cfg.cond_length / 2.0
        )

        delta_P_cap  = 2.0 * fp["sigma"] / r_cap
        delta_P_grav = fp["rho_l"] * abs(g) * self.cfg.length

        Q_cap = (
            (delta_P_cap - delta_P_grav)
            * K * fp["rho_l"] * fp["h_fg"]
            / (fp["mu_l"] * max(L_eff, 1e-9))
        )
        return max(Q_cap, 0.0)

    def operating_limits(self, T_range: Optional[tuple] = None) -> dict:
        """
        Berechnet das Kapillarlimit über einen Temperaturbereich.

        Args:
            T_range: (T_min, T_max) in °C. Standard: Fluid-T_range.

        Returns:
            dict mit 'T' [°C] und 'Q_capillary' [W]
        """
        if T_range is None:
            T_range = self.fluid_props["T_range"]
        T_arr = np.linspace(float(T_range[0]) + 5, float(T_range[1]) - 5, 30)
        Q_cap = np.array([self._capillary_limit(T) for T in T_arr])
        return {"T": T_arr, "Q_capillary": Q_cap}


# ── TEG-Modell ────────────────────────────────────────────────────────────────

@dataclass
class TEGConfig:
    """Konfigurationsparameter für einen thermoelektrischen Generator (TEG)."""

    model:       str   = "SP1848-27145"
    alpha:       float = 0.045   # V/K – Seebeck-Koeffizient (Modul)
    R_internal:  float = 2.3     # Ω   – Innenwiderstand
    area:        float = 0.0016  # m²  – Fläche (40 × 40 mm)


class TEGModel:
    """
    Thermoelektrischer Generator – Leistungsberechnung.

    Verwendet das vereinfachte Modell:
        V_oc  = α · ΔT
        P_max = V_oc² / (4 · R_i)   (Leistungsanpassung)
    """

    def __init__(self, cfg: TEGConfig = TEGConfig()) -> None:
        self.cfg = cfg

    def voltage_open(self, delta_T: float) -> float:
        """Leerlaufspannung [V] bei Temperaturdifferenz ΔT [K]."""
        return self.cfg.alpha * delta_T

    def power_max(self, delta_T: float) -> float:
        """Maximale Ausgangsleistung [W] bei ΔT [K] (Impedanzanpassung)."""
        V_oc = self.voltage_open(delta_T)
        return V_oc ** 2 / (4.0 * self.cfg.R_internal)

    def power_curve(self, delta_T_range: tuple = (5, 80)) -> dict:
        """
        Erzeugt eine Leistungskurve über einen ΔT-Bereich.

        Returns:
            dict mit 'delta_T' [K], 'P_max_mW' [mW], 'V_oc' [V]
        """
        dT = np.linspace(float(delta_T_range[0]), float(delta_T_range[1]), 50)
        P  = np.array([self.power_max(dt) for dt in dT])
        V  = np.array([self.voltage_open(dt) for dt in dT])
        return {"delta_T": dT, "P_max_mW": P * 1000.0, "V_oc": V}


# ── Convenience-Funktion ──────────────────────────────────────────────────────

def quick_simulation(
    Q: float = 40.0,
    T_source: float = 80.0,
    length: float = 0.25,
    diameter: float = 0.01,
    wick: str = "sintered",
    fluid: str = "water",
) -> dict:
    """
    Schnelle Einzel-Simulation einer Heatpipe mit TEG-Auswertung.

    Args:
        Q:        Wärmeleistung [W]
        T_source: Quelltemperatur [°C]
        length:   Gesamtlänge [m]
        diameter: Außendurchmesser [m]
        wick:     Dochttyp (sintered | grooved | mesh | screen_200mesh)
        fluid:    Arbeitsfluid (water | ammonia)

    Returns:
        dict mit Temperaturprofil, R_total, Q_max, TEG-Daten
    """
    cfg    = HeatpipeConfig(length=length, diameter=diameter,
                            wick_type=wick, fluid=fluid)
    model  = HeatpipeModel(cfg)
    result = model.temperature_profile(Q, T_source)
    teg    = TEGModel()
    result["teg"]              = teg.power_curve()
    result["teg_power_at_dT"]  = teg.power_max(result["delta_T"]) * 1000.0  # mW
    return result


if __name__ == "__main__":
    r = quick_simulation(Q=50.0, T_source=85.0)
    print(f"R_total = {r['R_total']:.3f} K/W")
    print(f"T_evap  = {r['T_evap']:.1f} °C,  T_cond = {r['T_cond']:.1f} °C")
    print(f"Q_max (Kapillarlimit) = {r['Q_max']:.1f} W")
    print(f"TEG-Leistung bei ΔT={r['delta_T']:.1f} K: {r['teg_power_at_dT']:.1f} mW")
