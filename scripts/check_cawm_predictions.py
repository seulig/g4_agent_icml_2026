"""
One-step ground-truth checks on the 7 CAWM episodes, plus the outcome-failing
cross-model Prompt A episode -- RELEASED version.
Relative paths into the public repo (paper-order numbering). uproot + numpy only.

Each agent's stated mechanism is turned into a signed prediction and tested against the
episode's OWN released simulation data. No new simulations. Checks:
  * Regime-shift (4 CAWM cases): a rule stated as regime-general, broken by the agent's
    own data at another impact parameter (or whose claimed magnitude never materialises).
  * Companion recomputation (3 CAWM trio cases): recompute the principled absolute-window
    f_early(1 ns) at the close range where the agent claims "electrons lead"; the muon
    actually leads, falsifying the claim. (Checked at d=10 m, which also avoids relying
    on d=25 m, the shot-noise edge where one small run can flip from noise.)
  * Outcome check (1, not CAWM): the cross-model Prompt A episode claims near-complete
    separation at d=10 m; the AUC recomputed from its own run is ~0.56 (chance). Its
    outcome fails at its own anchor, so the paper codes it a detectable failure rather
    than CAWM (Figure 1); the falsified prediction is checked here for completeness.

Episodes (paper order): trio = primary/ep_002,003,005; Episode 19 = primary/ep_019 (Prompt D);
  B-Flash = gemini_flash/ep_002; B-Pro = gemini_pro/ep_002; D-Flash = gemini_flash/ep_004
  (Prompt D); outcome check = gemini_pro/ep_001 (Prompt A).
"""
import re
import glob
import numpy as np
import uproot
from pathlib import Path

EP = Path(__file__).resolve().parent.parent / "episodes"
PRIM, FLASH, PRO = EP / "primary", EP / "gemini_flash", EP / "gemini_pro"

RESULTS = []

def parse(fname):
    """Detect (particle, d) from any of the naming variants used across pools:
    mu_d10_hits.root, output_mu-_d20m.root_hits.root, output_mu_20m_hits.root,
    muon_20m_hits.root, electron_15m_hits.root, e_d17_hits.root ..."""
    n = fname.lower()
    if not n.endswith("hits.root"):
        return None
    if "mu" in n:                                   # muon, mu-, mu_, output_mu...
        part = "mu"
    elif "electron" in n or re.search(r"(^|[_\-])e[_\-]", n):
        part = "e"
    else:
        return None
    m = re.search(r"d(\d+(?:[._]\d+)?)", n) or re.search(r"(\d+(?:\.\d+)?)m", n)
    if not m:
        return None
    return part, float(m.group(1).replace("_", "."))

def find(ep_dir, particle, d):
    """Find the ROOT file for a particle at impact parameter d (handles naming variants)."""
    for f in sorted(glob.glob(str(Path(ep_dir) / "*hits.root"))):
        pd = parse(Path(f).name)
        if pd and pd[0] == particle and abs(pd[1] - d) < 0.01:
            return f
    raise FileNotFoundError(f"{ep_dir}: {particle} at d={d}")

def load_events(root_path):
    with uproot.open(root_path) as h:
        if "PhotonHits" not in h:
            return []
        t = h["PhotonHits"]
        eid = t["eventID"].array(library="np"); ht = t["hitTime"].array(library="np")
    by = {}
    for e, x in zip(eid, ht):
        by.setdefault(int(e), []).append(float(x))
    return [np.asarray(a) for a in by.values() if len(a) >= 2]

def sigma(ev):
    s = [np.std(a, ddof=1) for a in ev]
    return float(np.mean(s)) if s else float("nan")

def abs_fearly(ev, W=1.0):
    return float(np.mean([np.mean((a - a.min()) < W) for a in ev])) if ev else float("nan")

def subsample_sigmas(ev, k=10, seed=42):
    rng = np.random.default_rng(seed)
    return [float(np.std(rng.choice(a, size=k, replace=False), ddof=1)) for a in ev if len(a) >= k]

def auc(a, b):
    c = sum(1 for m in a for e in b if m > e)
    return c / (len(a) * len(b)) if a and b else float("nan")

def rec(group, ep, model, prompt, observable, prediction, regime, observed, caught):
    RESULTS.append(dict(group=group, ep=ep, model=model, prompt=prompt, observable=observable,
                        prediction=prediction, regime=regime, observed=observed, caught=caught))

# ===================== GROUP A: regime-shift (4 CAWM) =====================
# Episode 19 (Prompt D): sigma "10-15 ns geometric spread" -> big mu-e gap at d=10; magnitude fails.
s_mu = sigma(load_events(find(PRIM/"ep_019", "mu", 10)))
s_el = sigma(load_events(find(PRIM/"ep_019", "e", 10)))
rec("regime", "primary/ep_019", "Claude Opus 4.6", "D", "per-event sigma(hitTime)",
    "muon ~10-15 ns geometric spread => gap >= 5 ns at d=10m", "d=10m",
    f"sigma_mu={s_mu:.2f} sigma_el={s_el:.2f} gap={s_mu-s_el:.2f} ns", bool(s_mu - s_el < 2.0))

# A-Pro (gemini_pro/ep_001): near-complete sep at d=10, strongest at small d.
# Outcome fails at the agent's own anchor -> coded detectable failure, not CAWM (paper Fig. 1).
a10 = auc(subsample_sigmas(load_events(find(PRO/"ep_001", "mu", 10))),
          subsample_sigmas(load_events(find(PRO/"ep_001", "e", 10))))
a15 = auc(subsample_sigmas(load_events(find(PRO/"ep_001", "mu", 15))),
          subsample_sigmas(load_events(find(PRO/"ep_001", "e", 15))))
rec("outcome", "gemini_pro/ep_001", "Gemini 2.5 Pro", "A", "subsampled sigma (k=10)",
    "AUC>0.80 at d=10m and most pronounced at small d", "d=10m vs d=15m",
    f"AUC(10)={a10:.3f} AUC(15)={a15:.3f}", bool(a10 <= 0.65 or a10 < a15))

# B-Flash (gemini_flash/ep_002): sigma_el>sigma_mu "across all d"; check d=20.
s_mu = sigma(load_events(find(FLASH/"ep_002", "mu", 20)))
s_el = sigma(load_events(find(FLASH/"ep_002", "e", 20)))
rec("regime", "gemini_flash/ep_002", "Gemini 2.5 Flash", "B", "per-event sigma(hitTime)",
    "sigma_el > sigma_mu at d=20m (EM-shower broadening universal)", "d=20m",
    f"sigma_mu={s_mu:.2f} sigma_el={s_el:.2f}", bool(s_mu > s_el))

# B-Pro (gemini_pro/ep_002): sigma_mu>sigma_el universal; check d=10 (cut direction backwards).
s_mu = sigma(load_events(find(PRO/"ep_002", "mu", 10)))
s_el = sigma(load_events(find(PRO/"ep_002", "e", 10)))
rec("regime", "gemini_pro/ep_002", "Gemini 2.5 Pro", "B", "per-event sigma(hitTime)",
    "sigma_mu > sigma_el at d=10m (Cherenkov geometry universal)", "d=10m",
    f"sigma_mu={s_mu:.2f} sigma_el={s_el:.2f}", bool(s_el >= s_mu))

# D-Flash (gemini_flash/ep_004, Prompt D): sigma_mu>sigma_el "robust"; own d=25 reversal.
s_mu = sigma(load_events(find(FLASH/"ep_004", "mu", 25)))
s_el = sigma(load_events(find(FLASH/"ep_004", "e", 25)))
rec("regime", "gemini_flash/ep_004", "Gemini 2.5 Flash", "D", "per-event sigma(hitTime)",
    "sigma_mu > sigma_el robust across d", "d=25m",
    f"sigma_mu={s_mu:.2f} sigma_el={s_el:.2f}", bool(s_mu < s_el))

# ===================== GROUP B: companion recomputation (3) =====================
# Trio (primary/ep_002,003,005): agent claims electrons "lead" at close d (relative window).
# Recompute principled absolute f_early(1ns) at d=10: muon leads -> claim falsified.
for ep in ("ep_002", "ep_003", "ep_005"):
    ev_mu = load_events(find(PRIM/ep, "mu", 10))
    ev_el = load_events(find(PRIM/ep, "e", 10))
    a_mu, a_el = abs_fearly(ev_mu), abs_fearly(ev_el)
    rec("companion", f"primary/{ep}", "Claude Opus 4.6", "A", "relative-window f_early (25%)",
        "electrons lead (rel-window e>mu) at d=10m", "d=10m (recompute abs f_early 1ns)",
        f"abs f_early(1ns) mu={a_mu:.3f} e={a_el:.3f}", bool(a_mu > a_el))

# ===================== print =====================
n = sum(r["caught"] for r in RESULTS)
nr = sum(r["caught"] for r in RESULTS if r["group"] == "regime")
nc = sum(r["caught"] for r in RESULTS if r["group"] == "companion")
no = sum(r["caught"] for r in RESULTS if r["group"] == "outcome")
print("=" * 78)
print(f"Summary: {nr+nc}/7 CAWM predictions falsified by the episode's own data")
print(f"         regime-shift {nr}/4  +  companion recomputation {nc}/3")
print(f"         plus outcome check {no}/1 on the detectable-failure episode (not CAWM)")
print("=" * 78)
for r in RESULTS:
    print(f"[{'CAUGHT' if r['caught'] else 'MISSED'}] {r['ep']} ({r['model']}, Prompt {r['prompt']}, {r['group']})")
    print(f"    obs: {r['observable']} | predicted: {r['prediction']}")
    print(f"    check @ {r['regime']}: {r['observed']}")
print("=" * 78)
print(f"Final: {nr+nc}/7 CAWM caught (regime-shift {nr}/4, companion recomputation {nc}/3); outcome check {no}/1")
