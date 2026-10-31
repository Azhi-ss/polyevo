from dataclasses import dataclass
import logging
import sqlite3

import numpy as np


@dataclass
class BudgetLedger:
    limit: int | None = None
    calls: int = 0

    @property
    def exhausted(self):
        return self.limit is not None and self.calls >= self.limit


class CachedOracle:
    """Persist absolute properties and attempted evaluations, never relative scores."""

    def __init__(self, path, predictors, budget):
        self.predictors, self.budget = predictors, budget
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS cache (
                predictor TEXT, smiles TEXT, value REAL,
                PRIMARY KEY (predictor, smiles));
            CREATE TABLE IF NOT EXISTS ledger (
                call_index INTEGER PRIMARY KEY, predictor TEXT,
                predictor_call_index INTEGER, smiles TEXT, value REAL, generation INTEGER);
        """)
        self.budget.calls = self.db.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]

    def evaluate(self, individuals, generation):
        unique = list({i.id: i for i in individuals}.values())
        # Reserve every missing property before admitting a structure. A partial
        # final structure still consumes its real calls but is not scored.
        remaining = (self.budget.limit - self.budget.calls
                     if self.budget.limit is not None else float("inf"))
        admitted, work = [], {p.name: [] for p in self.predictors}
        for individual in unique:
            complete = True
            for predictor in self.predictors:
                row = self.db.execute(
                    "SELECT value FROM cache WHERE predictor=? AND smiles=?",
                    (predictor.name, individual.smiles)).fetchone()
                if row is not None:
                    individual.props[predictor.name] = float("nan") if row[0] is None else row[0]
                elif remaining > 0:
                    work[predictor.name].append(individual)
                    remaining -= 1
                else:
                    complete = False
            if complete:
                admitted.append(individual)
        for predictor in self.predictors:
            pending = work[predictor.name]
            size = predictor.batch_size or max(1, len(pending))
            count = self.db.execute(
                "SELECT COUNT(*) FROM ledger WHERE predictor=?", (predictor.name,)).fetchone()[0]
            for start in range(0, len(pending), size):
                batch = pending[start:start + size]
                try:
                    values = np.asarray(predictor.predict([i.smiles for i in batch]), dtype=float)
                    if values.shape != (len(batch),):
                        raise ValueError(f"Expected prediction shape {(len(batch),)}, got {values.shape}")
                except Exception:
                    logging.exception("Predictor %s failed; recording NaN without retry", predictor.name)
                    values = np.full(len(batch), np.nan)
                with self.db:
                    for individual, value in zip(batch, values):
                        value = float(value) if np.isfinite(value) else float("nan")
                        individual.props[predictor.name] = value
                        self.budget.calls += 1
                        count += 1
                        stored = value if np.isfinite(value) else None
                        self.db.execute("INSERT INTO cache VALUES (?, ?, ?)",
                                        (predictor.name, individual.smiles, stored))
                        self.db.execute("INSERT INTO ledger VALUES (?, ?, ?, ?, ?, ?)",
                                        (self.budget.calls, predictor.name, count,
                                         individual.smiles, stored, generation))
        return admitted

    def ledger_rows(self):
        return self.db.execute("SELECT * FROM ledger ORDER BY call_index").fetchall()

    def close(self):
        self.db.close()
