from pathlib import Path

import pytest
import yaml


@pytest.fixture
def config():
    cfg = yaml.safe_load((Path(__file__).parents[1] / "presets/mock_demo.yaml").read_text())
    cfg.update(population_size=8, max_generations=2, budget=200)
    cfg["reproduce"].update(families=3, children=3)
    return cfg
