import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from method import concentration
from method import e_value
from method import p_value
from method import sim2real
from method import vincent
from method.distributions import BernoulliRare
from method.distributions import BetaSkewed
from method.distributions import BimodalMixture
from method.distributions import GaussianMixture
from method.distributions import TruncatedNormal
from method.distributions import UniformSpike


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
N_SAMPLES = 200
N_SEEDS = 100
CONFIDENCE = 0.95
GLOBAL_BANK_SIZE = 10080
HORIZONS = (10, 20, 50, 100, 200)
VINCENT_GAPS = (0.05, 0.10, 0.20, 0.30)
ETA_ABLATION_VALUES = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0)
ETA_ABLATION_BANKS = ("Sim_35", "Sim_252", "Sim_756", f"Sim_{GLOBAL_BANK_SIZE}", "Sim_7_biased")
SIM2REAL_ETA_BY_BANK = {
    "Sim_35": 5.0,
    "Sim_252": 2.0,
    "Sim_756": 1.0,
    f"Sim_{GLOBAL_BANK_SIZE}": 25.0,
    "Sim_7_biased": 5.0,
}
ETA_ABLATION_SEEDS = tuple(range(N_SEEDS))
ETA_ABLATION_MAX_DISTS = 20
N_WORKERS = max(1, min(os.cpu_count() or 1, 8))
WORKER_BANKS = None


def real_sets():
    real_6 = [
        ("GaussianMix_03_07", GaussianMixture([0.3, 0.7], [0.12, 0.12], [0.5, 0.5])),
        ("GaussianMix_05_08", GaussianMixture([0.5, 0.8], [0.10, 0.10], [0.4, 0.6])),
        ("UniformSpike_05_06", UniformSpike(0.5, 0.6)),
        ("Beta_10_10", BetaSkewed(10.0, 10.0)),
        ("Beta_2_2", BetaSkewed(2.0, 2.0)),
        ("GaussianMix_3comp", GaussianMixture([0.2, 0.5, 0.8], [0.08, 0.08, 0.08], [0.3, 0.4, 0.3])),
    ]
    real_20 = [
        ("Bernoulli_005", BernoulliRare(0.05)),
        ("Bernoulli_02", BernoulliRare(0.2)),
        ("Bernoulli_08", BernoulliRare(0.8)),
        ("Beta_05_2", BetaSkewed(0.5, 2.0)),
        ("Beta_2_05", BetaSkewed(2.0, 0.5)),
        ("Beta_5_1", BetaSkewed(5.0, 1.0)),
        ("Beta_1_5", BetaSkewed(1.0, 5.0)),
        ("Beta_2_2", BetaSkewed(2.0, 2.0)),
        ("Beta_10_10", BetaSkewed(10.0, 10.0)),
        ("TruncNorm_02_01", TruncatedNormal(0.2, 0.1)),
        ("TruncNorm_05_01", TruncatedNormal(0.5, 0.1)),
        ("TruncNorm_08_01", TruncatedNormal(0.8, 0.1)),
        ("TruncNorm_05_005", TruncatedNormal(0.5, 0.05)),
        ("Bimodal_03", BimodalMixture(0.3)),
        ("Bimodal_09", BimodalMixture(0.9)),
        ("UniformSpike_01_03", UniformSpike(0.1, 0.3)),
        ("UniformSpike_05_04", UniformSpike(0.5, 0.4)),
        ("UniformSpike_05_06", UniformSpike(0.5, 0.6)),
        ("GaussianMix_03_07", GaussianMixture([0.3, 0.7], [0.12, 0.12], [0.5, 0.5])),
        ("GaussianMix_02_05", GaussianMixture([0.2, 0.5], [0.10, 0.10], [0.6, 0.4])),
    ]
    real_100 = []
    for mean in np.linspace(0.08, 0.92, 20):
        concentration = 85.0 if mean < 0.5 else 95.0
        real_100.append((
            f"Beta_tight_{mean:.3f}".replace(".", "_"),
            BetaSkewed(mean * concentration, (1.0 - mean) * concentration),
        ))

    for i, mean in enumerate(np.linspace(0.08, 0.92, 20)):
        concentration = 2.4 + 0.6 * (i % 5)
        real_100.append((
            f"Beta_wide_{mean:.3f}_{i % 5}".replace(".", "_"),
            BetaSkewed(mean * concentration, (1.0 - mean) * concentration),
        ))

    for p in np.linspace(0.025, 0.975, 20):
        real_100.append((
            f"Bernoulli_{p:.3f}".replace(".", "_"),
            BernoulliRare(float(p)),
        ))

    for i, loc in enumerate(np.linspace(0.02, 0.98, 20)):
        spike_prob = [0.15, 0.35, 0.55, 0.75][i % 4]
        real_100.append((
            f"UniformSpike_grid_{i:02d}",
            UniformSpike(float(loc), spike_prob),
        ))

    for i, center in enumerate(np.linspace(0.15, 0.85, 20)):
        gap = 0.16 + 0.04 * (i % 4)
        means = [max(center - gap, 0.02), min(center + gap, 0.98)]
        sigmas = [0.04 + 0.015 * (i % 3), 0.05 + 0.015 * ((i + 1) % 3)]
        weights = [0.35 + 0.10 * (i % 4), 0.65 - 0.10 * (i % 4)]
        real_100.append((
            f"GaussianMix_broad_{i:02d}",
            GaussianMixture(means, sigmas, weights),
        ))
    return {
        "Real_6": real_6,
        "Real_20": real_20,
        "Real_100": real_100,
    }


def make_sparse_bank():
    bank = []
    for p in [0.017, 0.047, 0.097, 0.197, 0.497, 0.797, 0.947]:
        bank.append(BernoulliRare(p))
    for a, b in [
        (0.297, 0.303), (0.497, 0.503), (0.497, 1.01), (1.01, 0.497),
        (0.497, 2.01), (2.01, 0.497), (1.01, 4.99), (4.99, 1.01),
        (1.99, 2.01), (1.99, 5.01), (5.01, 1.99), (9.98, 10.02),
    ]:
        bank.append(BetaSkewed(a, b))
    for mu, sig in [(0.097, 0.051), (0.197, 0.101), (0.297, 0.151), (0.497, 0.101), (0.797, 0.101), (0.497, 0.051), (0.497, 0.201)]:
        bank.append(TruncatedNormal(mu, sig))
    for w in [0.097, 0.297, 0.497, 0.697, 0.897]:
        bank.append(BimodalMixture(w))
    for loc, prob in [(0.097, 0.303), (0.097, 0.603), (0.497, 0.403), (0.897, 0.303), (0.897, 0.603), (0.977, 0.083)]:
        bank.append(UniformSpike(loc, prob))
    bank.extend([
        GaussianMixture([0.297, 0.703], [0.121, 0.119], [0.51, 0.49]),
        GaussianMixture([0.197, 0.503], [0.101, 0.099], [0.61, 0.39]),
        GaussianMixture([0.497, 0.803], [0.101, 0.099], [0.39, 0.61]),
        GaussianMixture([0.197, 0.503, 0.803], [0.081, 0.079, 0.081], [0.30, 0.40, 0.30]),
        GaussianMixture([0.097, 0.903], [0.061, 0.059], [0.51, 0.49]),
        GaussianMixture([0.047, 0.953], [0.031, 0.029], [0.49, 0.51]),
        GaussianMixture([0.247, 0.753], [0.051, 0.179], [0.69, 0.31]),
        GaussianMixture([0.253, 0.747], [0.179, 0.051], [0.31, 0.69]),
    ])
    return bank


def make_biased_bank():
    return [
        BetaSkewed(0.43, 3.07),
        BetaSkewed(0.73, 4.11),
        BetaSkewed(1.03, 5.17),
        TruncatedNormal(0.15, 0.08),
        TruncatedNormal(0.25, 0.10),
        UniformSpike(0.11, 0.47),
        BimodalMixture(0.91),
    ]


def _clip01(x, eps=0.001):
    return min(max(float(x), eps), 1.0 - eps)


def _perturb_distribution(dist, level):
    sign = -1.0 if level % 2 == 0 else 1.0
    mag = 1 + level // 2

    if isinstance(dist, BernoulliRare):
        return BernoulliRare(_clip01(dist.p + sign * 0.004 * mag))
    if isinstance(dist, BetaSkewed):
        eps = sign * 0.025 * mag
        return BetaSkewed(max(dist.alpha * (1.0 + eps), 0.05), max(dist.beta * (1.0 - eps), 0.05))
    if isinstance(dist, TruncatedNormal):
        return TruncatedNormal(_clip01(dist.mu + sign * 0.004 * mag, eps=0.01), max(dist.sigma * (1.0 + sign * 0.04 * mag), 0.01))
    if isinstance(dist, BimodalMixture):
        return BimodalMixture(_clip01(dist.weight + sign * 0.008 * mag, eps=0.01))
    if isinstance(dist, UniformSpike):
        return UniformSpike(
            _clip01(dist.spike_location + sign * 0.006 * mag, eps=0.01),
            _clip01(dist.spike_prob - sign * 0.008 * mag, eps=0.01),
        )
    if isinstance(dist, GaussianMixture):
        means = np.clip(dist.means + sign * 0.004 * mag, 0.01, 0.99)
        sigmas = np.maximum(dist.sigmas * (1.0 + sign * 0.035 * mag), 0.01)
        weights = np.maximum(dist.weights + sign * 0.01 * mag * np.linspace(-1.0, 1.0, dist.n_components), 0.01)
        weights = weights / np.sum(weights)
        return GaussianMixture(means.tolist(), sigmas.tolist(), weights.tolist())
    raise TypeError(f"unsupported distribution type: {type(dist).__name__}")


def make_support_bank(size):
    bank = []
    idx = 0
    while len(bank) < size:
        phase = idx % 5
        sweep = idx // 5
        u = ((sweep * 37 + phase * 11) % 101 + 0.37) / 102.0

        if phase == 0:
            p = _clip01(0.015 + 0.97 * u, eps=0.004)
            bank.append(BernoulliRare(p))
        elif phase == 1:
            mean = _clip01(0.025 + 0.95 * u, eps=0.01)
            concentration = 2.15 + 11.0 * (((sweep * 13) % 17) / 16.0)
            bank.append(BetaSkewed(mean * concentration, (1.0 - mean) * concentration))
        elif phase == 2:
            loc = _clip01(0.015 + 0.97 * u, eps=0.01)
            prob = _clip01(0.08 + 0.84 * (((sweep * 19 + 7) % 29) / 28.0), eps=0.02)
            bank.append(UniformSpike(loc, prob))
        elif phase == 3:
            center = 0.08 + 0.84 * u
            gap = 0.14 + 0.18 * (((sweep * 23 + 5) % 31) / 30.0)
            means = [max(center - gap, 0.01), min(center + gap, 0.99)]
            sigmas = [
                0.025 + 0.10 * (((sweep * 7 + 3) % 19) / 18.0),
                0.025 + 0.10 * (((sweep * 5 + 9) % 19) / 18.0),
            ]
            weight = _clip01(0.18 + 0.64 * (((sweep * 17 + 2) % 23) / 22.0), eps=0.03)
            bank.append(GaussianMixture(means, sigmas, [weight, 1.0 - weight]))
        else:
            mu = _clip01(0.02 + 0.96 * u, eps=0.02)
            sigma = 0.035 + 0.22 * (((sweep * 29 + 1) % 37) / 36.0)
            bank.append(TruncatedNormal(mu, sigma))
        idx += 1
    return bank


def make_global_bank(size):
    rng = np.random.default_rng(1260)
    bank = []

    for _ in range(size):
        mean = rng.uniform(0.01, 0.99)
        variance_fraction = rng.uniform(0.015, 0.985)
        concentration = 1.0 / variance_fraction - 1.0
        bank.append(BetaSkewed(mean * concentration, (1.0 - mean) * concentration))
    return bank


def make_matched_bank(variants_per_real):
    local_bank = []
    for dists in real_sets().values():
        for _, dist in dists:
            for level in range(max(1, variants_per_real // 2)):
                local_bank.append(_perturb_distribution(dist, level))
    target_size = variants_per_real * sum(len(dists) for dists in real_sets().values())
    support_bank = make_support_bank(target_size - len(local_bank))
    return local_bank + support_bank


def sim_banks():
    sparse = make_sparse_bank()[:35]
    matched_2 = make_matched_bank(2)
    global_bank = make_global_bank(GLOBAL_BANK_SIZE)
    matched_6 = make_matched_bank(6)
    biased = make_biased_bank()
    return {
        "Sim_35": sparse,
        f"Sim_{len(matched_2)}": matched_2,
        f"Sim_{len(matched_6)}": matched_6,
        f"Sim_{len(global_bank)}": global_bank,
        f"Sim_{len(biased)}_biased": biased,
    }


def vincent_shifted_simulator(distribution, shift=0.12):
    mean = float(distribution.true_mean())
    direction = 1.0 if mean <= 0.5 else -1.0

    if isinstance(distribution, BernoulliRare):
        return BernoulliRare(_clip01(distribution.p + direction * shift))
    if isinstance(distribution, BetaSkewed):
        concentration = distribution.alpha + distribution.beta
        shifted_mean = _clip01(mean + direction * shift, eps=0.02)
        return BetaSkewed(shifted_mean * concentration, (1.0 - shifted_mean) * concentration)
    if isinstance(distribution, TruncatedNormal):
        return TruncatedNormal(_clip01(distribution.mu + direction * shift, eps=0.02), distribution.sigma)
    if isinstance(distribution, BimodalMixture):
        shifted_mean = _clip01(mean + direction * shift, eps=0.02)
        weight = _clip01((0.8 - shifted_mean) / 0.6, eps=0.02)
        return BimodalMixture(weight)
    if isinstance(distribution, UniformSpike):
        return UniformSpike(
            _clip01(distribution.spike_location + direction * shift, eps=0.02),
            distribution.spike_prob,
        )
    if isinstance(distribution, GaussianMixture):
        means = np.clip(distribution.means + direction * shift, 0.02, 0.98)
        return GaussianMixture(means.tolist(), distribution.sigmas.tolist(), distribution.weights.tolist())
    raise TypeError(f"unsupported distribution type: {type(distribution).__name__}")


def baseline_methods():
    return [
        ("e_value_wsr", lambda dist, seed: e_value.certificate(dist, seed, N_SAMPLES, confidence=CONFIDENCE, method="wsr", refine=False)),
        ("e_value_constant_0.25", lambda dist, seed: e_value.certificate(
            dist,
            seed,
            N_SAMPLES,
            confidence=CONFIDENCE,
            method="constant",
            constant_lambdas=(0.25,),
            refine=False,
        )),
        ("e_value_constant_0.5", lambda dist, seed: e_value.certificate(
            dist,
            seed,
            N_SAMPLES,
            confidence=CONFIDENCE,
            method="constant",
            constant_lambdas=(0.5,),
            refine=False,
        )),
        ("hoeffding", lambda dist, seed: concentration.certificate(dist, seed, N_SAMPLES, confidence=CONFIDENCE, method="hoeffding")),
        ("empirical_bernstein", lambda dist, seed: concentration.certificate(dist, seed, N_SAMPLES, confidence=CONFIDENCE, method="empirical_bernstein")),
        ("t_test", lambda dist, seed: p_value.certificate(dist, seed, N_SAMPLES, confidence=CONFIDENCE, method="student_t")),
        ("z_test", lambda dist, seed: p_value.certificate(dist, seed, N_SAMPLES, confidence=CONFIDENCE, method="normal")),
        ("sequential_t_test", lambda dist, seed: p_value.certificate(dist, seed, N_SAMPLES, confidence=CONFIDENCE, method="sequential_t")),
    ]


def summarize(certificate, true_mean):
    widths = certificate[:, 1] - certificate[:, 0]
    widths = np.where(np.isnan(widths), 1.0, widths)
    contains = (certificate[:, 0] <= true_mean) & (true_mean <= certificate[:, 1])
    return widths, bool(contains[-1]), bool(np.all(contains))


def collect_result(real_set, distribution, method, bank, certificate, true_mean):
    widths, final_cover, anytime_cover = summarize(certificate, true_mean)
    contains = (certificate[:, 0] <= true_mean) & (true_mean <= certificate[:, 1])
    return {
        "real_set": real_set,
        "distribution": distribution,
        "method": method,
        "bank": bank,
        "true_mean": true_mean,
        "final_width": float(widths[-1]),
        "mean_width": float(np.mean(widths)),
        "final_coverage": float(final_cover),
        "anytime_coverage": float(anytime_cover),
        **{f"width_{h}": float(widths[h - 1]) for h in HORIZONS},
        **{f"coverage_{h}": float(contains[h - 1]) for h in HORIZONS},
    }


def init_worker(banks):
    global WORKER_BANKS
    WORKER_BANKS = banks


def run_main_task(task):
    real_set, dist_name, dist, seed = task
    banks = WORKER_BANKS
    true_mean = float(dist.true_mean())
    rows = []

    for bank_name, bank in banks.items():
        cert = sim2real.certificate(
            dist,
            seed,
            N_SAMPLES,
            bank,
            confidence=CONFIDENCE,
            eta=SIM2REAL_ETA_BY_BANK[bank_name],
            refine=True,
            kappa=1.0,
        )
        rows.append(collect_result(real_set, dist_name, "sim2real", bank_name, cert, true_mean))

    ideal_cert = sim2real.certificate(
        dist,
        seed,
        N_SAMPLES,
        [dist],
        confidence=CONFIDENCE,
        eta=1.0,
        refine=True,
        kappa=1.0,
    )
    rows.append(collect_result(real_set, dist_name, "sim2real", "Ideal", ideal_cert, true_mean))

    vincent_sim = vincent_shifted_simulator(dist)
    for gap in VINCENT_GAPS:
        cert = vincent.certificate(
            dist,
            vincent_sim,
            seed,
            N_SAMPLES,
            confidence=CONFIDENCE,
            sim2real_gap_upper=gap,
            sim2real_gap_lower=gap,
        )
        rows.append(collect_result(real_set, dist_name, "vincent", f"gap_{gap:g}", cert, true_mean))

    ideal_vincent = vincent.certificate(
        dist,
        dist,
        seed,
        N_SAMPLES,
        confidence=CONFIDENCE,
        sim2real_gap_upper=0.0,
        sim2real_gap_lower=0.0,
    )
    rows.append(collect_result(real_set, dist_name, "vincent", "Ideal", ideal_vincent, true_mean))

    for method_name, run_method in baseline_methods():
        cert = run_method(dist, seed)
        rows.append(collect_result(real_set, dist_name, method_name, "none", cert, true_mean))

    return rows


def run_eta_task(task):
    real_set, dist_name, dist, seed = task
    banks = WORKER_BANKS
    true_mean = float(dist.true_mean())
    rows = []

    for bank_name in ETA_ABLATION_BANKS:
        bank = banks[bank_name]
        for eta in ETA_ABLATION_VALUES:
            cert = sim2real.certificate(
                dist,
                seed,
                N_SAMPLES,
                bank,
                confidence=CONFIDENCE,
                eta=eta,
                refine=False,
                kappa=1.0,
            )
            row = collect_result(real_set, dist_name, "sim2real", bank_name, cert, true_mean)
            row["eta"] = eta
            rows.append(row)

    return rows


def flatten_results(results):
    rows = []
    for result in results:
        rows.extend(result)
    return rows


def collect_eta_ablation(banks):
    tasks = [
        (real_set, dist_name, dist, seed)
        for real_set, dists in real_sets().items()
        for dist_name, dist in dists[:ETA_ABLATION_MAX_DISTS]
        for seed in ETA_ABLATION_SEEDS
    ]
    print(f"running eta ablation: {len(tasks)} tasks on {N_WORKERS} workers")
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=init_worker, initargs=(banks,)) as executor:
        return flatten_results(executor.map(run_eta_task, tasks, chunksize=4))


def collect_main_results(banks):
    tasks = [
        (real_set, dist_name, dist, seed)
        for real_set, dists in real_sets().items()
        for dist_name, dist in dists
        for seed in range(N_SEEDS)
    ]
    print(f"running main comparison: {len(tasks)} tasks on {N_WORKERS} workers")
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=init_worker, initargs=(banks,)) as executor:
        return flatten_results(executor.map(run_main_task, tasks, chunksize=4))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    banks = sim_banks()
    rows = collect_main_results(banks)

    summary = aggregate(rows, ["real_set", "method", "bank"])
    per_distribution = aggregate(rows, ["real_set", "distribution", "method", "bank"])
    curves = curve_summary(rows)
    eta_summary = aggregate(collect_eta_ablation(banks), ["real_set", "bank", "eta"])

    write_csv(os.path.join(DATA_DIR, "summary.csv"), summary)
    write_csv(os.path.join(DATA_DIR, "per_distribution.csv"), per_distribution)
    write_csv(os.path.join(DATA_DIR, "width_curves.csv"), curves)
    write_csv(os.path.join(DATA_DIR, "coverage_curves.csv"), coverage_curve_summary(rows))
    write_csv(os.path.join(DATA_DIR, "eta_ablation.csv"), eta_summary)
    print(f"saved {os.path.join(DATA_DIR, 'summary.csv')}")
    print(f"saved {os.path.join(DATA_DIR, 'per_distribution.csv')}")
    print(f"saved {os.path.join(DATA_DIR, 'width_curves.csv')}")
    print(f"saved {os.path.join(DATA_DIR, 'coverage_curves.csv')}")
    print(f"saved {os.path.join(DATA_DIR, 'eta_ablation.csv')}")


def aggregate(rows, keys):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, []).append(row)

    out = []
    for key, values in sorted(grouped.items()):
        item = dict(zip(keys, key))
        for field in ["final_width", "mean_width", "final_coverage", "anytime_coverage"]:
            item[field] = float(np.mean([v[field] for v in values]))
        for horizon in HORIZONS:
            field = f"width_{horizon}"
            if field in values[0]:
                item[field] = float(np.mean([v[field] for v in values]))
        item["n_runs"] = len(values)
        out.append(item)
    return out


def curve_summary(rows):
    grouped = {}
    for row in rows:
        key = (row["real_set"], row["method"], row["bank"])
        grouped.setdefault(key, []).append(row)

    out = []
    for (real_set, method, bank), values in sorted(grouped.items()):
        for horizon in HORIZONS:
            out.append({
                "real_set": real_set,
                "method": method,
                "bank": bank,
                "n": horizon,
                "mean_width": float(np.mean([v[f"width_{horizon}"] for v in values])),
            })
    return out


def coverage_curve_summary(rows):
    grouped = {}
    for row in rows:
        key = (row["real_set"], row["method"], row["bank"])
        grouped.setdefault(key, []).append(row)

    out = []
    for (real_set, method, bank), values in sorted(grouped.items()):
        for horizon in HORIZONS:
            out.append({
                "real_set": real_set,
                "method": method,
                "bank": bank,
                "n": horizon,
                "coverage": float(np.mean([v[f"coverage_{horizon}"] for v in values])),
            })
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
