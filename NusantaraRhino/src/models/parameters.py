"""
parameters.py — Validasi database parameter & konfigurasi skenario berbasis Pydantic.
Memastikan setiap parameter memiliki sumber, tahun, metode, distribusi ketidakpastian,
dan assumption_type yang sah (Observed / Literature-derived / Estimated / Assumed / Scenario-defined).
"""

from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import yaml
from pydantic import BaseModel, Field

AssumptionType = Literal[
    "Observed",
    "Literature-derived",
    "Estimated",
    "Assumed",
    "Scenario-defined",
]
ConfidenceLevel = Literal["High", "Medium", "Low"]


class ParameterRecord(BaseModel):
    parameter_name: str
    value: float
    unit: str
    source: str
    source_year: int = Field(ge=1950, le=2030)
    method: str
    uncertainty_distribution: str
    confidence_level: ConfidenceLevel
    assumption_type: AssumptionType


class ScenarioConfig(BaseModel):
    label: str
    n_sites: int = Field(ge=1, le=2)
    initial_n_site1: int = Field(ge=2)
    k_site1: int = Field(ge=10)
    chronic_survival_drag: float = Field(ge=0.0, le=0.2)
    management_surv_boost: float = Field(ge=0.0, le=0.2)
    catastrophe_prob: float = Field(ge=0.0, le=0.5)
    catastrophe_severity: float = Field(ge=0.0, le=0.95)
    catastrophe_corr: float = Field(ge=0.0, le=1.0)
    founding_year: Optional[int] = None
    founder_size: int = Field(ge=0)
    k_site2: int = Field(ge=0)
    translocation_mortality: float = Field(default=0.0, ge=0.0, le=0.5)
    migrant_interval: Optional[int] = None
    migrants_per_event: int = Field(default=0, ge=0)


def load_validated_parameters(csv_path: Path) -> dict[str, ParameterRecord]:
    df = pd.read_csv(csv_path)
    out = {}
    for _, row in df.iterrows():
        rec = ParameterRecord(**row.to_dict())
        out[rec.parameter_name] = rec
    return out


def load_scenarios(yaml_path: Path) -> tuple[dict, dict[str, ScenarioConfig]]:
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    scenarios = {k: ScenarioConfig(**v) for k, v in raw["scenarios"].items()}
    return raw, scenarios
