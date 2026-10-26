"""Data module for the in-class exercise "Which drug stops the tumor?"
Students import load(); instructor calls reveal() at the end of class.
Keep this file out of the notebook so notebook assistants cannot read the truth.
"""
import numpy as np, pandas as pd, anndata as ad

_TYPES = ["tumor", "fibroblast", "Tcell"]
_TRUTH = {  # (division per day, death per day)
    "vehicle": {"tumor": (1.00, 0.10), "fibroblast": (0.30, 0.10), "Tcell": (0.20, 0.15)},
    "drugA":   {"tumor": (1.00, 1.20), "fibroblast": (0.30, 0.12), "Tcell": (0.20, 0.15)},
    "drugB":   {"tumor": (0.05, 0.10), "fibroblast": (0.30, 0.10), "Tcell": (0.20, 0.15)},
    "drugC":   {"tumor": (0.60, 0.90), "fibroblast": (0.10, 0.90), "Tcell": (0.05, 1.00)},
}
_N0 = {"tumor": 4000, "fibroblast": 3000, "Tcell": 3000}
_DAYS = 3
_cache = {}


def _simulate(seed=0, n_genes=1500, cells_per_well=400):
    rng = np.random.default_rng(seed)
    G = np.arange(n_genes); rng.shuffle(G)
    prog = {"tumor": G[0:40], "fibroblast": G[40:80], "Tcell": G[80:120], "cycle": G[120:170], "apoptosis": G[170:200]}
    base = rng.gamma(2, 0.5, n_genes)
    X, obs, abs_counts = [], [], []
    for arm, rates in _TRUTH.items():
        for rep in range(3):
            well = f"{arm}_r{rep+1}"
            N = {t: _N0[t] * np.exp((rates[t][0] - rates[t][1]) * _DAYS) * rng.lognormal(0, 0.08) for t in _TYPES}
            abs_counts.append(dict(well=well, arm=arm, **{t: int(N[t]) for t in _TYPES}))
            p = np.array([N[t] for t in _TYPES]); p /= p.sum()
            for t in rng.choice(_TYPES, size=cells_per_well, p=p):
                div, death = rates[t]; mu = base.copy(); mu[prog[t]] *= 6
                cycling = rng.random() < min(0.9, div / 1.2); dying = rng.random() < min(0.8, death / 1.5)
                if cycling: mu[prog["cycle"]] *= 5
                if dying:   mu[prog["apoptosis"]] *= 5
                lib = rng.lognormal(np.log(3000), 0.3)
                X.append(rng.negative_binomial(2, 2 / (2 + mu / mu.sum() * lib)))
                obs.append(dict(well=well, arm=arm, true_type=t, true_cycling=cycling, true_dying=dying))
    a = ad.AnnData(X=np.array(X, dtype=np.float32), obs=pd.DataFrame(obs))
    a.var_names = [f"g{i}" for i in range(n_genes)]
    programs = {k: [f"g{i}" for i in v] for k, v in prog.items()}
    return a, programs, pd.DataFrame(abs_counts).set_index("well")


def _get(seed):
    if seed not in _cache:
        _cache[seed] = _simulate(seed)
    return _cache[seed]


def load(seed=0):
    """Return (adata, markers). adata.obs has only 'well' and 'arm'."""
    full, programs, _ = _get(seed)
    a = full.copy(); a.obs = a.obs[["well", "arm"]]
    markers = {k: programs[k] for k in _TYPES}
    return a, markers


def state_programs(seed=0):
    """Released at Step 4: gene sets for 'cycle' and 'apoptosis'."""
    _, programs, _ = _get(seed)
    return {k: programs[k] for k in ["cycle", "apoptosis"]}


def total_live_cells(seed=0):
    """Released at Step 7: hemocytometer count per well."""
    _, _, ac = _get(seed)
    return ac[_TYPES].sum(1).rename("total_live_cells")


def reveal(adata, seed=0):
    """Instructor only. Prints truth and the student's annotation accuracy."""
    full, _, ac = _get(seed)
    truth = pd.DataFrame({(arm, t): r for arm, d in _TRUTH.items() for t, r in d.items()},
                         index=["division_per_day", "death_per_day"]).T
    print("True division / death rates per day:"); print(truth)
    print("\nTrue absolute cells per well at 72 h (mean of 3 wells):"); print(ac.groupby("arm").mean().round(0))
    if "cell_type" in adata.obs:
        acc = (adata.obs["cell_type"].values == full.obs["true_type"].values).mean()
        print(f"\nAnnotation accuracy: {acc:.3f}")
    return truth, ac
