import json, os, glob
import numpy as np
from scipy.stats import mannwhitneyu

ROOT = r"C:\Users\Mao\Desktop\DRL\Final Project\runs"
SEEDS = [42, 43, 44, 45, 46]
ENVS = ["cp", "mc", "acr", "ll"]
ENV_NAME = {"cp": "CartPole", "mc": "MountainCar", "acr": "Acrobot", "ll": "LunarLander"}

# vanilla data dirs (reused from Part 1)
VANILLA_DIR = {"cp": "final_cp", "mc": "final_mc", "acr": "final_acr", "ll": "final"}


def last_iter_mean(seed_dir):
    iters = sorted(glob.glob(os.path.join(seed_dir, "iter_*")))
    if not iters:
        raise FileNotFoundError(seed_dir)
    cfg = os.path.join(iters[-1], "config.json")
    d = json.load(open(cfg))
    return d["env_native_mean"]


def collect(variant, env, condition):
    """variant in {'vanilla','double','dueling'}"""
    if variant == "vanilla":
        base = os.path.join(ROOT, VANILLA_DIR[env], condition)
    else:
        base = os.path.join(ROOT, f"part2_{variant}_{env}", condition)
    vals = []
    for s in SEEDS:
        sd = os.path.join(base, f"seed_{s}")
        vals.append(last_iter_mean(sd))
    return np.array(vals)


def fmt_p(p):
    return f"{p:.4f}"


# ---------- Table 5 ----------
print("=" * 70)
print("TABLE 5: B3-hermes-full vs B0-env-native")
print("=" * 70)

claimed_t5 = {
    ("ll", "vanilla"): (173.22, 153.56, -11.4, 1.0000),
    ("ll", "double"): (171.90, 131.68, -23.4, 1.0000),
    ("ll", "dueling"): (166.86, 137.98, -17.3, 0.6905),
    ("cp", "vanilla"): (154.80, 334.44, 116.1, 0.0317),
    ("cp", "double"): (182.13, 388.60, 113.4, 0.1508),
    ("cp", "dueling"): (227.22, 315.99, 39.1, 0.2492),
    ("mc", "vanilla"): (-193.44, -132.53, 31.5, 0.0112),
    ("mc", "double"): (-195.16, -134.74, 31.0, 0.0097),
    ("mc", "dueling"): (-198.70, -146.89, 26.1, 0.0449),
    ("acr", "vanilla"): (-194.96, -82.92, 57.5, 0.0952),
    ("acr", "double"): (-233.82, -81.17, 65.3, 0.4206),
    ("acr", "dueling"): (-103.79, -80.05, 22.9, 0.0556),
}

results = {}  # (env,variant) -> dict
t5_fail = []
order = [("ll", "vanilla"), ("ll", "double"), ("ll", "dueling"),
         ("cp", "vanilla"), ("cp", "double"), ("cp", "dueling"),
         ("mc", "vanilla"), ("mc", "double"), ("mc", "dueling"),
         ("acr", "vanilla"), ("acr", "double"), ("acr", "dueling")]

for env, var in order:
    b0 = collect(var, env, "B0-env-native")
    b3 = collect(var, env, "B3-hermes-full")
    b0m, b3m = b0.mean(), b3.mean()
    delta = (b3m - b0m) / abs(b0m) * 100
    U, p = mannwhitneyu(b3, b0, alternative="two-sided")
    results[(env, var)] = dict(b0=b0, b3=b3, b0m=b0m, b3m=b3m, delta=delta, p=p)

    cb0, cb3, cdelta, cp = claimed_t5[(env, var)]
    ok_b0 = abs(b0m - cb0) <= 0.05
    ok_b3 = abs(b3m - cb3) <= 0.05
    ok_d = abs(delta - cdelta) <= 0.5
    ok_p = abs(p - cp) <= 0.005
    cell_ok = ok_b0 and ok_b3 and ok_d and ok_p
    status = "PASS" if cell_ok else "FAIL"
    print(f"[{status}] {ENV_NAME[env]:12s} {var:8s} | "
          f"B0 {b0m:8.2f}(c{cb0:8.2f}{'' if ok_b0 else ' X'}) | "
          f"B3 {b3m:8.2f}(c{cb3:8.2f}{'' if ok_b3 else ' X'}) | "
          f"d {delta:+6.1f}%(c{cdelta:+6.1f}{'' if ok_d else ' X'}) | "
          f"p {p:.4f}(c{cp:.4f}{'' if ok_p else ' X'})")
    if not cell_ok:
        t5_fail.append((env, var, dict(b0=(b0m, cb0, ok_b0), b3=(b3m, cb3, ok_b3),
                                       d=(delta, cdelta, ok_d), p=(p, cp, ok_p))))

# ---------- Table 6 ----------
print()
print("=" * 70)
print("TABLE 6: B3-hermes-full mean by variant + pairwise p")
print("=" * 70)

claimed_t6_means = {
    "cp": (334.44, 388.60, 315.99),
    "mc": (-132.53, -134.74, -146.89),
    "acr": (-82.92, -81.17, -80.05),
    "ll": (153.56, 131.68, 137.98),
}
claimed_t6_p = {
    "cp": (0.548, 0.841, 0.548),
    "mc": (1.000, 1.000, 0.841),
    "acr": (0.690, 0.310, 0.548),
    "ll": (1.000, 0.841, 0.841),
}

t6_fail = []
t6_order = ["cp", "mc", "acr", "ll"]
all_pairwise_p = []
for env in t6_order:
    van = results[(env, "vanilla")]["b3"]
    dbl = results[(env, "double")]["b3"]
    due = results[(env, "dueling")]["b3"]
    vm, dm, um = van.mean(), dbl.mean(), due.mean()
    _, p_vd = mannwhitneyu(van, dbl, alternative="two-sided")
    _, p_vu = mannwhitneyu(van, due, alternative="two-sided")
    _, p_du = mannwhitneyu(dbl, due, alternative="two-sided")
    all_pairwise_p.extend([(env, "V-Db", p_vd), (env, "V-Du", p_vu), (env, "Db-Du", p_du)])

    cvm, cdm, cum = claimed_t6_means[env]
    cpvd, cpvu, cpdu = claimed_t6_p[env]
    ok_means = (abs(vm - cvm) <= 0.05 and abs(dm - cdm) <= 0.05 and abs(um - cum) <= 0.05)
    ok_p = (abs(p_vd - cpvd) <= 0.005 and abs(p_vu - cpvu) <= 0.005 and abs(p_du - cpdu) <= 0.005)
    cell_ok = ok_means and ok_p
    status = "PASS" if cell_ok else "FAIL"
    print(f"[{status}] {ENV_NAME[env]:12s} | "
          f"means {vm:8.2f}/{dm:8.2f}/{um:8.2f} (c {cvm}/{cdm}/{cum}) | "
          f"p {p_vd:.3f}/{p_vu:.3f}/{p_du:.3f} (c {cpvd}/{cpvu}/{cpdu})")
    if not cell_ok:
        t6_fail.append((env, dict(means=((vm, dm, um), (cvm, cdm, cum), ok_means),
                                  p=((p_vd, p_vu, p_du), (cpvd, cpvu, cpdu), ok_p))))

# ---------- Prose claims ----------
print()
print("=" * 70)
print("PROSE CLAIMS")
print("=" * 70)

# 9/9 sparse positive
sparse_deltas = [results[(env, var)]["delta"] for env in ["cp", "mc", "acr"]
                 for var in ["vanilla", "double", "dueling"]]
n_pos = sum(1 for d in sparse_deltas if d > 0)
print(f"9/9 sparse positive: {n_pos}/9 positive  -> {'CONFIRMED' if n_pos == 9 else 'FAIL'}")

# MC 3/3 significant
mc_ps = [results[("mc", var)]["p"] for var in ["vanilla", "double", "dueling"]]
mc_sig = all(p < 0.05 for p in mc_ps)
print(f"MC 3/3 significant (<0.05): {[f'{p:.4f}' for p in mc_ps]} -> {'CONFIRMED' if mc_sig else 'FAIL'}")

# all pairwise p > 0.3
viol = [(e, lbl, p) for (e, lbl, p) in all_pairwise_p if p <= 0.3]
print(f"all 12 pairwise p>0.3: {'CONFIRMED' if not viol else 'VIOLATIONS: ' + str([(e,lbl,round(p,3)) for e,lbl,p in viol])}")
print(f"   (min pairwise p = {min(p for _,_,p in all_pairwise_p):.4f})")

# CartPole Dueling B0 highest across all variants
cp_b0_dueling = results[("cp", "dueling")]["b0m"]
all_b0 = {(env, var): results[(env, var)]["b0m"] for env, var in order}
max_b0_key = max(all_b0, key=all_b0.get)
# among CartPole variants
cp_b0s = {var: results[("cp", var)]["b0m"] for var in ["vanilla", "double", "dueling"]}
cp_highest = max(cp_b0s, key=cp_b0s.get)
print(f"CartPole Dueling B0 highest among CP variants: {cp_b0s} -> highest={cp_highest} "
      f"({'CONFIRMED' if cp_highest == 'dueling' else 'FAIL'})")
print(f"   CartPole Dueling B0 = {cp_b0_dueling:.2f} (claimed 227.22)")

# Acrobot Dueling B0 stats: -103.79 vs -194.96 vanilla, std 179->42
acr_du_b0 = results[("acr", "dueling")]["b0"]
acr_va_b0 = results[("acr", "vanilla")]["b0"]
print(f"Acrobot Dueling B0 mean={acr_du_b0.mean():.2f} (claimed -103.79), std={acr_du_b0.std(ddof=1):.2f} (claimed ~42)")
print(f"Acrobot vanilla B0 mean={acr_va_b0.mean():.2f} (claimed -194.96), std={acr_va_b0.std(ddof=1):.2f} (claimed ~179)")

# MC B3 std progression 3.08 / 17.04 / 32.35
mc_va_std = results[("mc", "vanilla")]["b3"].std(ddof=1)
mc_db_std = results[("mc", "double")]["b3"].std(ddof=1)
mc_du_std = results[("mc", "dueling")]["b3"].std(ddof=1)
print(f"MC B3 std progression: vanilla={mc_va_std:.2f}(c3.08) double={mc_db_std:.2f}(c17.04) dueling={mc_du_std:.2f}(c32.35)")

# Acrobot B0 std up to 211 (prose in 5.2)
acr_b0_stds = {var: results[("acr", var)]["b0"].std(ddof=1) for var in ["vanilla", "double", "dueling"]}
print(f"Acrobot B0 stds (prose 'up to 211'): {[f'{k}={v:.1f}' for k,v in acr_b0_stds.items()]}")

# ---------- Summary ----------
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Table 5 failures: {len(t5_fail)}")
for env, var, d in t5_fail:
    print(f"  {ENV_NAME[env]}/{var}: {d}")
print(f"Table 6 failures: {len(t6_fail)}")
for env, d in t6_fail:
    print(f"  {ENV_NAME[env]}: {d}")
