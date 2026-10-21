import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os import listdir
from os.path import join
from pathlib import Path
import argparse
import re
from scipy.stats import rankdata, wilcoxon
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
from matplotlib.transforms import blended_transform_factory
from stats import compute_speedups, compute_slowdown, compute_no_catch_up_diff

_SEED_RE = re.compile(r'_(\d+)\.csv$')


def _extract_seed(filename):
    m = _SEED_RE.search(filename)
    return int(m.group(1)) if m else None

plt.rcParams.update({'font.size': 20, 'font.family':'serif'})
#plt.rcParams.update(bundles.icml2024(family="serif", usetex=False, column="full", nrows=1))

BEST_PERFS = {
        'nas101': 1, # 0.9452,
        'nas201': 1, #0.945,
        'jahs_cifar10': 1.0,
        'jahs_co': 1.0,
        'jahs_fm': 1.0,
        'hpob_4796_3549': 1.0,
        'hpob_6767_31': 0.77,
        'hpob_6767_9914': 1.0,
        'hpob_6794_3': 1.05,
        'hpob_6794_31': 0.78,
        'hpob_6794_9914': 1.0,
        'hpob_6794_10101': 1.0,
        'hpob_4796_homo': 0.6,
        'hpob_4796_3549': 1.0, #1.0,
        'lc_adult': 1,
        'fcnet_slice': 0,
        'toy_branin': 0,
        'rosenbrock_v3': 0,
        'rosenbrock': 0,
        'quadratic': 0,
        'floris_cifar10': 4.4,
        'pv_crossed_barrel': 3.8,
    }

# Scale factor for test_performance values.
# JAHS and NAS benchmarks store accuracy as a percentage (e.g. 80.5 → divide by 100).
# HPOB, LCBench, and FCNet surrogates return values already in [0, 1] (scale=1).
PERF_SCALES = {
        'nas101': 100,
        'nas201': 100,
        'jahs_cifar10': 100,
        'jahs_co': 100,
        'jahs_fm': 100,
        'hpob_4796_3549': 1,
        'hpob_6767_31': 1,
        'hpob_6767_9914': 1,
        'hpob_6794_3': 1,
        'hpob_6794_31': 1,
        'hpob_6794_9914': 1,
        'hpob_6794_10101': 1,
        'hpob_4796_homo': 1,
        'hpob_4796_3549': 1,
        'lc_adult': 1,
        'fcnet_slice': 1,
        'toy_branin': 1,
        'rosenbrock_v3': 1,
        'rosenbrock': 1,
        'quadratic': 1,
        'floris_cifar10': 1,
        'pv_crossed_barrel': 1,
    }

BENCHMARK_LABELS = {
        'nas101': 'NAS-Bench-101',
        'nas201': 'NAS-Bench-201',
        'jahs_cifar10': 'JAHS (CIFAR-10)',
        'jahs_co': 'JAHS (CO)',
        'jahs_fm': 'JAHS (Fashion-MNIST)',
        'branin': 'Branin',
        'rosenbrock_v3': 'Rosenbrock',
        'rosenbrock': 'Rosenbrock (3D)',
        'quadratic': 'Quadratic',
        'hpob_6767_31': 'HPOB 1',
        'hpob_6794_3': 'HPOB 2',
        'lc_adult': 'LCBench',
        'fcnet_slice': 'FCNet (Slice)',
        'floris_cifar10': 'Floris',
        'pv_crossed_barrel': 'PV (crossed-barrel)',
        'hpob_4796_3549': 'HPO-B 4796:3549',
        'hpob_6767_31': 'HPO-B 6767:31',
        'hpob_6767_9914': 'HPO-B 6767:9914',
        'hpob_6794_3': 'HPO-B 6794:3',
        'hpob_6794_31': 'HPO-B 6794:31',
        'hpob_6794_9914': 'HPO-B 6794:9914',
        'hpob_6794_10101': 'HPO-B 6794:10101',
    }



# =====================================================================
# SCHEME A — family color + variant linestyle
#   Every algorithm in a family shares ONE hue.
#   Variant identity is encoded purely via linestyle, identically across
#   all interactive families. MC uses thicker strokes to stand out.
#   Non-interactive baselines get distinct hues (their "variant" is
#   trivially the base, so they're all solid).
# =====================================================================
ALGO_COLORS_A = {
    # MC family — vivid magenta-red, our hero
    'mc_stratified':              ('#D81B60', 'solid'),
    'mc_stratified_int_early':    ('#D81B60', (0, (5, 2))),
    'mc_stratified_int_dist':     ('#D81B60', (0, (1, 1.5))),
    'mc_stratified_int_recover':  ('#D81B60', (0, (3, 1, 1, 1))),
    'mc_stratified_int_many':     ('#D81B60', (0, (5, 1, 1, 1, 1, 1))),
    # PC family — blue
    'pc':                ('#1E88E5', 'solid'),
    'pc_int_early':      ('#1E88E5', (0, (5, 2))),
    'pc_int_dist':       ('#1E88E5', (0, (1, 1.5))),
    'pc_int_recover':    ('#1E88E5', (0, (3, 1, 1, 1))),
    'pc_int_many':       ('#1E88E5', (0, (5, 1, 1, 1, 1, 1))),
    # Dynabo family — green
    'dynabo':                ('#43A047', 'solid'),
    'dynabo_int_early':      ('#43A047', (0, (5, 2))),
    'dynabo_int_dist':       ('#43A047', (0, (1, 1.5))),
    'dynabo_int_recover':    ('#43A047', (0, (3, 1, 1, 1))),
    'dynabo_int_many':       ('#43A047', (0, (5, 1, 1, 1, 1, 1))),
    # Pibo family — purple
    'pibo':                ('#8E24AA', (0, (5, 2))),
    'pibo_int_early':      ('#8E24AA', (0, (5, 2))),
    'pibo_int_recover':    ('#8E24AA', (0, (3, 1, 1, 1))),
    'pibo_recover':    ('#8E24AA', (0, (3, 1, 1, 1))),
    # Non-interactive baselines — distinct hues, all solid
    'skoptbo':   ('#FB8C00', 'solid'),  # orange
    'optunabo':  ('#6D4C41', 'solid'),  # brown
    'hebo':      ('#00ACC1', 'solid'),  # cyan
    'smac':      ("#747A1F", 'solid'),  # lime/olive
    'cmaes':     ("#744980", 'solid'),  # slate
}
 
 
# =====================================================================
# SCHEME B — same as A, but baselines are dimmed to grays
#   Use this when the comparison story is "interactive families vs the
#   pack" — the five non-interactive baselines collapse into a gray
#   reference cloud, MC stays loud.
# =====================================================================
ALGO_COLORS_B = {
    # MC family
    'mc_stratified':              ('#D81B60', 'solid'),
    'mc_stratified_int_early':    ('#D81B60', (0, (5, 2))),
    'mc_stratified_int_dist':     ('#D81B60', (0, (1, 1.5))),
    'mc_stratified_int_recover':  ('#D81B60', (0, (3, 1, 1, 1))),
    'mc_stratified_int_many':     ('#D81B60', (0, (5, 1, 1, 1, 1, 1))),
    # PC family
    'pc':                ('#1E88E5', 'solid'),
    'pc_int_early':      ('#1E88E5', (0, (5, 2))),
    'pc_int_dist':       ('#1E88E5', (0, (1, 1.5))),
    'pc_int_recover':    ('#1E88E5', (0, (3, 1, 1, 1))),
    'pc_int_many':       ('#1E88E5', (0, (5, 1, 1, 1, 1, 1))),
    # Dynabo
    'dynabo':                ('#2E7D32', 'solid'),
    'dynabo_int_early':      ('#2E7D32', (0, (5, 2))),
    'dynabo_int_dist':       ('#2E7D32', (0, (1, 1.5))),
    'dynabo_int_recover':    ('#2E7D32', (0, (3, 1, 1, 1))),
    'dynabo_int_many':       ('#2E7D32', (0, (5, 1, 1, 1, 1, 1))),
    # Pibo
    'pibo':                ('#8E24AA', (0, (5, 2))),
    'pibo_int_early':      ('#8E24AA', (0, (5, 2))),
    'pibo_int_recover':    ('#8E24AA', (0, (3, 1, 1, 1))),
    'pibo_recover':    ('#8E24AA', (0, (3, 1, 1, 1))),
    # Baselines — gray ramp + distinct dash patterns so they're still
    # individually identifiable, but they recede visually.
    'skoptbo':   ('#9E9E9E', (0, (4, 2))),
    'optunabo':  ('#757575', (0, (2, 2))),
    'hebo':      ('#BDBDBD', (0, (6, 2))),
    'smac':      ('#616161', (0, (4, 1, 1, 1))),
    'cmaes':     ('#424242', (0, (1, 1.5))),
}
 
 
# =====================================================================
# SCHEME C — within-family shade gradient + linestyle accent
#   Within an interactive family, the base is the darkest shade and the
#   four variants get progressively lighter. Reads as "interactive
#   variants lift away from the base method". Strong family identity,
#   weaker variant readability than A/B (the lightest shades fade on
#   white backgrounds), so prefer this when you specifically want to
#   communicate the "variants of one method" story.
# =====================================================================
ALGO_COLORS_C = {
    # MC family — magenta gradient
    'mc_stratified':              ('#931147', 'solid'),
    'mc_stratified_int_early':    ('#B6175A', (0, (5, 2))),
    'mc_stratified_int_dist':     ('#C9568D', (0, (1, 1.5))),
    'mc_stratified_int_recover':  ('#D687AC', (0, (3, 1, 1, 1))),
    'mc_stratified_int_many':     ('#E2AEC8', (0, (5, 1, 1, 1, 1, 1))),
    # PC family — blue gradient
    'pc':                ('#0B3D89', 'solid'),
    'pc_int_early':      ('#0D47A1', (0, (5, 2))),
    'pc_int_dist':       ('#5478B6', (0, (1, 1.5))),
    'pc_int_recover':    ('#869EC8', (0, (3, 1, 1, 1))),
    'pc_int_many':       ('#ADBED7', (0, (5, 1, 1, 1, 1, 1))),
    # Dynabo — green gradient
    'dynabo':                ('#16501B', 'solid'),
    'dynabo_int_early':      ('#1B5E20', (0, (5, 2))),
    'dynabo_int_dist':       ('#5A8B5C', (0, (1, 1.5))),
    'dynabo_int_recover':    ('#8AAC8B', (0, (3, 1, 1, 1))),
    'dynabo_int_many':       ('#B0C7B0', (0, (5, 1, 1, 1, 1, 1))),
    # Pibo — purple gradient
    'pibo':                ('#3F1177', 'solid'),
    'pibo_int_early':      ('#4A148C', (0, (5, 2))),
    'pibo_int_recover':    ('#A488B7', (0, (3, 1, 1, 1))),
    'pibo_recover':    ('#A488B7', (0, (3, 1, 1, 1))),
    # Baselines — distinct hues, dashed
    'skoptbo':   ('#FB8C00', (0, (5, 2))),
    'optunabo':  ('#795548', (0, (5, 2))),
    'hebo':      ('#26A69A', (0, (5, 2))),
    'smac':      ("#A7B119", (0, (5, 2))),
    'cmaes':     ("#91659C", (0, (5, 2))),
}
ALGO_COLORS = ALGO_COLORS_C

# DisCoMBO ablation variants — registered against the active palette so they
# render through the standard plot_regrets path. cmax is the new default
# reference and inherits the base mc_stratified colour for visual continuity.
ALGO_COLORS.update({
    'mc_stratified_annealed_sine_cmax':           (ALGO_COLORS['mc_stratified'][0], 'solid'),
    'mc_stratified_annealed_sine_c20':            ('#D81B60', (0, (5, 2))),
    'mc_stratified_annealed_sine_c2':             ('#F06292', (0, (1, 1))),
    'mc_stratified_annealed_sine_p100':           ('#43A047', 'solid'),       # green 500 (lighter)
    'mc_stratified_annealed_sine_p10000':         ('#81C784', (0, (5, 2))),   # green 300 (lighter)
    'mc_stratified_annealed_sine_gamma_strong':   ('#0D47A1', (0, (5, 2))),       # dark blue, dashed
    'mc_stratified_annealed_sine_gamma_weak':     ('#1976D2', (0, (3, 1, 1, 1))), # medium blue, dash-dotted
    'mc_stratified_uniform':                      ('#42A5F5', (0, (1, 1))),       # light blue, dotted
    'mc_stratified_annealed_sine_noball':         ('#FB8C00', 'solid'),           # orange 600 (lighter), solid
})




ALGO_NAME_MAPPING = {
        'mc_stratified': 'DisCoMBO',
        'mc_stratified_annealed_sine': 'DisCoMBO',
        'mc_stratified_annealed_sine_c2': 'DisCoMBO (c=2)',
        'mc_stratified_annealed_sine_c20': 'DisCoMBO (c=20)',
        'mc_stratified_annealed_sine_cmax': 'DisCoMBO (c=50)',
        'mc_stratified_uniform': 'DisCoMBO (uniform bins)',
        'mc_stratified_annealed_sine_p100': 'DisCoMBO (c=20, pool=100)',
        'mc_stratified_annealed_sine_p10000': 'DisCoMBO (c=20, pool=10k)',
        'mc_stratified_annealed_sine_gamma_weak': r'DisCoMBO (c=20, $\beta$-weak)',
        'mc_stratified_annealed_sine_gamma_strong': r'DisCoMBO (c=20, $\beta$-strong)',
        'mc_stratified_annealed_sine_noball': 'DisCoMBO (c=20, no ball)',
        'mc_stratified_int_early':   'DisCoMBO (prior@5)',
        'mc_stratified_int_dist':    'DisCoMBO (dist. prior@5)',
        'mc_stratified_int_recover': 'DisCoMBO (bad prior@5)',
        'mc_stratified_int_many':    'DisCoMBO (4 priors)',
        'pc': 'IBO-HPC',
        'pc_int': 'IBO (prior@10)',
        'pc_int_early': 'IBO (prior@5)',
        'pc_int_dist': 'IBO (dist. intervention@5)',
        # 'pc_int_multi': 'IBO (prior@5,20)',
        # 'pc_int_autorecover': 'IBO (w/ prior@5)',
        'pc_int_recover': 'IBO (bad prior@5)',
        'pc_int_many': 'IBO (4 priors)',
        # 'pc_new': 'IBO',
        # 'pc_int_new': 'IBO (w/ prior@10)',
        # 'pc_int_late': 'IBO (w/ prior@10)',
        # 'pc_int_early_new': 'IBO (w/ prior@5)',
        # 'pc_int_multi_new': 'IBO (w/ prior@5,20)',
        # 'pc_ablation_cond_05': 'IBO q=0.5',
        # 'pc_ablation_cond_025': 'IBO q=0.25',
        # 'pc_ablation_cond_075': 'IBO q=0.75',
        # 'pc_ablation_sample_size_5': 'IBO @ 5 samples',
        # 'pc_ablation_sample_size_10': 'IBO @ 10 samples',
        # 'pc_ablation_sample_size_30': 'IBO @ 30 samples',
        # 'pc_ablation_decay_03': 'IBO $\gamma$=0.3',
        # 'pc_ablation_decay_07': 'IBO $\gamma$=0.7',
        # 'pc_ablation_decay_099': 'IBO $\gamma$=0.99',
        'cmaes': 'CMA-ES',
        'hebo': 'HEBO',
        # 'gp': 'BO w/ GP',
        # 'priorband': 'Priorband',
        'optunabo': 'BO w/ TPE',
        'skoptbo': 'BO w/ RF',
        'pibo': 'πBO',
        'smac': 'SMAC',
        'dynabo': 'DynaBO',
        'dynabo_int_early':   'DynaBO (prior@5)',
        'dynabo_int_dist':    'DynaBO (dist. prior@5)',
        'dynabo_int_recover': 'DynaBO (bad prior@5)',
        'dynabo_int_many':    'DynaBO (4 priors)',
        'pibo_int_early':     'πBO (prior@5)',
        'pibo_int_recover':   'πBO (bad prior@5)',
        'rs': 'RS',
        'rs_int_early': 'RS (w/ prior@5)',
        'rs_int_late': 'RS (w/ prior@10)',
        'rs_test': 'RSL',
        'ls': 'LS',
        'hyperband': 'Hyperband',
        'smac_test': 'SMAC',
        'hyperband_test': 'Hyperband',
        'bopro': 'BOPrO',
        'bopro_plain': 'BOPrO',
        'pibo_plain': 'πBO',
        'bopro_recover': 'BOPrO Recover',
        'pibo_recover': 'πBO (bad prior@5)',
    }

TIME_PLOT_FIGSIZE = (8, 7)
TIME_PLOT_MARGINS = {
    'left': 0.20,
    'right': 0.89,
    'bottom': 0.17,
    'top': 0.91,
}


def _format_y_tick_four_digits(value, _pos):
    """Format y ticks as exactly four digits and one dot.

    Pattern:
      - X.XXX for |value| < 10
      - XX.XX for |value| >= 10
    """
    if not np.isfinite(value):
        return ''
    sign = '-' if value < 0 else ''
    abs_value = abs(value)

    if abs_value < 10:
        return f'{sign}{abs_value:.3f}'

    # Keep the requested fixed pattern (XX.XX) even for very large values.
    if abs_value > 99.99:
        abs_value = 99.99
    return f'{sign}{abs_value:.2f}'


def _hide_edge_ticklabels(ax):
    """Hide the first and last major tick labels on both axes."""
    fig = ax.figure
    fig.canvas.draw()
    for axis in (ax.xaxis, ax.yaxis):
        ticks = axis.get_major_ticks()
        if len(ticks) >= 2:
            ticks[0].label1.set_visible(False)
            ticks[-1].label1.set_visible(False)


def _apply_timeplot_axis_style(ax):
    """Apply shared axis style for time-series comparison panels."""
    ax.yaxis.set_major_formatter(FuncFormatter(_format_y_tick_four_digits))
    ax.tick_params(axis='y', labelsize=24)


def _resolve_algo_names(raw_algos: str):
    return [a.strip() for a in raw_algos.split(',') if a.strip()]


def _resolve_benchmarks(raw_benchmarks: str):
    return [b.strip() for b in raw_benchmarks.split(',') if b.strip()]


def _collect_final_scores_for_cd(base_path, benchmarks, algos, max_reads,
                                 seeds=None):
    """Collect one score per (benchmark, algorithm) cell.

    Per (algo, benchmark) cell: take up to ``max_reads`` CSVs in a
    deterministic order, average each run's best ``test_performance``, and
    return that scalar. Order:
      1. Files whose trailing ``_<seed>.csv`` matches ``seeds``, in pool order.
      2. Any other seeded files, sorted by seed value.
      3. Files without a seed in the filename, lex order.
    Falling through to (3) means an algo without seeded filenames (e.g.
    timestamp-named runs) still contributes a score from its first
    ``max_reads`` files instead of being dropped.
    """
    scores = {algo: [] for algo in algos}
    valid_benchmarks = []
    seed_pool = list(seeds) if seeds is not None else None
    seed_pool_set = set(seed_pool) if seed_pool is not None else set()

    for benchmark in benchmarks:
        bench_scores = {}
        ok = True
        for algo in algos:
            path = join(base_path, f"{benchmark}_{algo}")
            try:
                all_files = listdir(path)
            except FileNotFoundError:
                ok = False
                break
            csvs = [f for f in all_files if f.endswith('.csv')]
            if not csvs:
                ok = False
                break

            seeded = {}  # seed -> first filename
            unseeded = []
            for f in csvs:
                s = _extract_seed(f)
                if s is None:
                    unseeded.append(f)
                elif s not in seeded:
                    seeded[s] = f

            ordered = []
            if seed_pool is not None:
                ordered.extend(seeded[s] for s in seed_pool if s in seeded)
            ordered.extend(seeded[s] for s in sorted(seeded)
                           if s not in seed_pool_set)
            ordered.extend(sorted(unseeded))
            ordered = ordered[:max_reads]

            run_scores = []
            for f in ordered:
                try:
                    df = pd.read_csv(join(path, f), index_col=0)
                except pd.errors.EmptyDataError:
                    continue
                if 'test_performance' not in df.columns:
                    continue
                perf = df['test_performance'].dropna()
                if perf.empty:
                    continue
                run_scores.append(float(perf.max()))
            if not run_scores:
                ok = False
                break
            bench_scores[algo] = float(np.mean(run_scores))

        if ok:
            valid_benchmarks.append(benchmark)
            for algo in algos:
                scores[algo].append(bench_scores[algo])

    return scores, valid_benchmarks


def _base_algo(algo):
    """Map an algo id to its standard (no-intervention) base.

    Used by the CD plot so each variant inherits the saturated colour and
    plain method name of its baseline (e.g. ``pc_int_early`` -> ``pc``).
    """
    if '_int_' in algo:
        return algo.split('_int_', 1)[0]
    if algo.endswith('_recover'):
        return algo[: -len('_recover')]
    return algo


def _short_label(algo):
    """Display name for ``algo`` with any ``(...)`` qualifier stripped."""
    name = ALGO_NAME_MAPPING.get(algo, algo)
    head = name.split('(', 1)[0].strip()
    return head or name


def _base_method_id(algo_id):
    """Map an intervention-variant algo id back to its base method id.

    Recognized suffixes: `_int_early`, `_int_dist`, `_int_many`,
    `_int_recover`, and the legacy `_recover` (only when stripping it
    yields a known base method). Falls back to `algo_id` unchanged.
    """
    for suffix in ('_int_early', '_int_dist', '_int_many', '_int_recover'):
        if algo_id.endswith(suffix):
            return algo_id[:-len(suffix)]
    if algo_id.endswith('_recover'):
        candidate = algo_id[:-len('_recover')]
        if candidate in ALGO_COLORS:
            return candidate
    return algo_id


def _resolve_color_style(algo_id, unify_color=False):
    """(color, linestyle) for `algo_id`. If `unify_color` is True the color
    comes from the base method's standard variant while the linestyle is
    kept from the original algo id (so intervention variants share a hue
    with their standard sibling but stay visually distinguishable)."""
    style = ALGO_COLORS.get(algo_id, ('#888888', 'solid'))
    if not unify_color:
        return style
    base = _base_method_id(algo_id)
    base_color = ALGO_COLORS.get(base, style)[0]
    return base_color, style[1]


def _build_nonsignificant_groups(avg_ranks, scores, alpha):
    """Create contiguous groups in rank order where all pairwise Wilcoxon tests are not significant."""
    ordered_algos = sorted(avg_ranks.keys(), key=lambda a: avg_ranks[a])
    groups = []

    for start in range(len(ordered_algos)):
        for end in range(start + 1, len(ordered_algos)):
            candidate = ordered_algos[start:end + 1]
            nonsig = True
            for i in range(len(candidate)):
                for j in range(i + 1, len(candidate)):
                    a_i = candidate[i]
                    a_j = candidate[j]
                    try:
                        test = wilcoxon(scores[a_i], scores[a_j], alternative='two-sided', zero_method='wilcox')
                        p_val = test.pvalue
                    except ValueError:
                        p_val = 1.0
                    if p_val < alpha:
                        nonsig = False
                        break
                if not nonsig:
                    break
            if nonsig:
                groups.append((start, end))

    pruned = []
    for s, e in sorted(groups, key=lambda x: (x[0], -(x[1] - x[0]))):
        is_subset = False
        for s2, e2 in groups:
            if (s2 < s or e2 > e) and s2 <= s and e2 >= e:
                is_subset = True
                break
        if not is_subset and (e - s) >= 1:
            pruned.append((s, e))

    return ordered_algos, pruned


def plot_critical_difference(
    file,
    benchmarks,
    algos,
    alpha=0.05,
    base_path='../../data/new-exp',
    max_reads=500,
    algo_fontsize=19,
    seeds=None,
):
    """Plot a critical difference style diagram using Wilcoxon signed-rank significance."""
    plt.rcParams.update(
        {
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'mathtext.fontset': 'stix',
            'font.size': 16,
        }
    )

    scores, valid_benchmarks = _collect_final_scores_for_cd(
        base_path=base_path,
        benchmarks=benchmarks,
        algos=algos,
        max_reads=max_reads,
        seeds=seeds,
    )
    if len(valid_benchmarks) < 2:
        raise ValueError('Need at least 2 benchmarks with valid data to compute Wilcoxon significance.')

    ranks_per_benchmark = []
    for idx in range(len(valid_benchmarks)):
        bench_scores = np.array([scores[a][idx] for a in algos], dtype=float)
        # Higher test_performance gets lower (better) rank.
        ranks = rankdata(-bench_scores, method='average')
        ranks_per_benchmark.append(ranks)
    ranks_per_benchmark = np.array(ranks_per_benchmark)

    avg_ranks = {algo: float(np.mean(ranks_per_benchmark[:, i])) for i, algo in enumerate(algos)}
    ordered_algos, nonsig_groups = _build_nonsignificant_groups(avg_ranks, scores, alpha)

    n_algos = len(ordered_algos)
    max_rank = float(n_algos)

    fig_h = max(4.0, 0.34 * n_algos + 1.5)
    fig, ax = plt.subplots(figsize=(13.5, fig_h))
    ax.set_xlim(0.75, max_rank + 0.25)
    ax.axis('off')

    axis_y = 1.0
    ax.plot([1.0, max_rank], [axis_y, axis_y], color='black', linewidth=2.3)
    for r in range(1, n_algos + 1):
        ax.plot([r, r], [axis_y - 0.006, axis_y + 0.006], color='black', linewidth=3.0)
        ax.text(r, axis_y + 0.02, str(r), ha='center', va='bottom', fontsize=15, fontweight='bold')

    left = ordered_algos[: (n_algos + 1) // 2]
    right = ordered_algos[(n_algos + 1) // 2 :]

    y0 = 0.94
    y_step = 0.045

    for i, algo in enumerate(left):
        y = y0 - i * y_step
        x = avg_ranks[algo]
        base = _base_algo(algo)
        col = ALGO_COLORS.get(base, ALGO_COLORS.get(algo, ('black', '-')))[0]
        label = ALGO_NAME_MAPPING.get(base, base)
        ax.plot([x, x], [axis_y, y], color=col, linewidth=3.5)
        ax.plot([x, 0.96], [y, y], color=col, linewidth=3.5)
        ax.text(0.93, y, label, ha='right', va='center',
                fontsize=algo_fontsize, color=col, weight='bold')

    for i, algo in enumerate(right):
        y = y0 - i * y_step
        x = avg_ranks[algo]
        base = _base_algo(algo)
        col = ALGO_COLORS.get(base, ALGO_COLORS.get(algo, ('black', '-')))[0]
        label = ALGO_NAME_MAPPING.get(base, base)
        ax.plot([x, x], [axis_y, y], color=col, linewidth=3.5)
        ax.plot([x, max_rank + 0.04], [y, y], color=col, linewidth=3.5)
        ax.text(max_rank + 0.08, y, label, ha='left', va='center',
                fontsize=algo_fontsize, color=col, weight='bold')

    group_ys = []
    group_y = axis_y - 0.01
    for s, e in nonsig_groups:
        x1 = avg_ranks[ordered_algos[s]]
        x2 = avg_ranks[ordered_algos[e]]
        ax.plot([x1, x2], [group_y, group_y], color='black', linewidth=2.0)
        group_ys.append(group_y)
        group_y -= 0.022

    max_side = max(len(left), len(right)) if n_algos > 0 else 1
    lowest_label_y = y0 - (max_side - 1) * y_step
    lowest_group_y = min(group_ys) if group_ys else axis_y - 0.03
    lower = min(lowest_label_y - 0.03, lowest_group_y - 0.025)
    upper = axis_y + 0.07
    ax.set_ylim(lower, upper)

    plt.tight_layout(pad=0.25)
    plt.savefig(file, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)

def read_csvs(result_path, max_reads=500):
    dfs = []
    files = listdir(result_path)
    print(result_path)
    if len(files) > max_reads:
        # randomly sample 500 runs
        indices = np.random.choice(np.arange(len(files)), max_reads, replace=False)
        files = [files[i] for i in indices]
    for file in files:
        pth = join(result_path, file)
        try:
            df = pd.read_csv(pth, index_col=0)
            dfs.append(df)
        except pd.errors.EmptyDataError:
            continue

    return dfs

def compute_regrets(result_path, best_perf, align_cost, num_exps=500, uncertainty='SE', perf_scale=100):
    dfs = read_csvs(result_path, num_exps)
    test_perfs = []
    costs = []
    max_iters = int(1e9)

    for df in dfs:
        # drop invalid columns
        df = df[df['cost'] != float('inf')]
        if df['cost'].isna().all():
            df.loc[:, 'cost'] = 1
        max_iters = min(max_iters, df.index.max())
        test_perf_per_iter = []
        cum_cost = []
        for i in range(0, max_iters):
            iterdf = df.iloc[:i]
            #iterdf = df.head(i)
            if i == 0 and align_cost:
                for j in range(iterdf.shape[0]):
                    tmp_df = iterdf[:j]
                    best = float(tmp_df['test_performance'].max())
                    regret = best_perf - (best / perf_scale)
                    test_perf_per_iter.append(regret)
                    np_cost = tmp_df['cost'].to_numpy()
                    cum_cost.append(np_cost[:j].sum())
            else:
                # get best result for that iteration
                best = float(iterdf['test_performance'].max())
                regret = best_perf - (best / perf_scale)
                cost = float(iterdf['cost'].sum())
                test_perf_per_iter.append(regret)
                cum_cost.append(cost)

        test_perfs.append(np.array(test_perf_per_iter))
        costs.append(np.array(cum_cost))
    
    # some algos don't run the full number of iterations in each execution (e.g. SMAC)
    # hence we cannot create an np.array -> go over each iteration and compute mean and std.
    results = []
    for i in range(max_iters - 2):
        perfs = [run_res[i] for run_res in test_perfs if (i - 1) < len(run_res)]
        rc = [run_cost[i] for run_cost in costs if (i-1) < len(run_cost)]
        perf_mean, perf_std = np.mean(perfs), np.std(perfs)
        if uncertainty == 'SE':
            perf_std = perf_std / np.sqrt(num_exps)
        mean_cost = np.mean(rc)
        results.append([perf_mean, perf_std, mean_cost])
    return np.array(results)

def collect_results(exp='nas101', num_exps=500, uncertainty='SE'):
    prefix = f'../../data/new-exp/{exp}_'
    #algos = [ 'pc', 'pc_int_recover', 'pc_int_multi', 'rs', 'pibo', 'bopro', 'ls', 'smac']
    #algos = [ 'pc_ablation_cond_025', 'pc_ablation_cond_05', 'pc_ablation_cond_075']
    #algos = [ 'pc_ablation_sample_size_5', 'pc_ablation_sample_size_10', 'pc', 'pc_ablation_sample_size_30']
    #algos = [ 'pc_ablation_decay_03', 'pc_ablation_decay_07', 'pc_ablation_decay_099']
    # algos = ['pc', 'pc_int_early', 'pc_int_dist', 'pc_int_recover', 'pc_int_many', 'mc_stratified_cmax', 'mc_int_early', 'mc_int_dist', 'mc_int_recover', 'mc_int_many', 'pibo_recover', 'bopro_recover', 'smac', 'skoptbo', 'optunabo']
    # algos = ['pc', 'pc_int_early', 'pc_int_dist', 'pc_int_recover', 'pc_int_many', 'mc_stratified_cmax', 'mc_int_early', 'mc_int_dist', 'mc_int_recover', 'mc_int_many', 'pibo_recover', 'bopro_recover', 'smac', 'skoptbo', 'optunabo']
    # algos = ['pc', 'mc_stratified_cmax', 'dynabo']
    algos = ['pc', 'mc_stratified', 'cmaes', 'hebo', 'smac', 'optunabo', 'skoptbo']
    result_paths = [prefix + algo for algo in algos]
    # align_costs = lambda algo: algo.startswith('pc') or algo.startswith('mc') or algo == 'ls'
    # align_costs = lambda algo: True
    align_costs = lambda algo: False
    perf_scale = PERF_SCALES.get(exp, 100)
    regrets = [(ALGO_NAME_MAPPING[algo], algo, compute_regrets(rp, BEST_PERFS[exp], align_costs(algo), num_exps, uncertainty, perf_scale)) for algo, rp in zip(algos, result_paths)]
    return regrets


def _resolve_curve_alpha(algo_id, other_alpha=0.35, algo_alphas=None):
    """Resolve alpha for one algorithm curve.

    Precedence:
      1) explicit per-algorithm override from ``algo_alphas``
      2) legacy behavior: PC/MC families at 1.0, others at ``other_alpha``
    """
    if algo_alphas is not None and algo_id in algo_alphas:
        return float(algo_alphas[algo_id])
    if 'pc' in algo_id or 'mc' in algo_id:
        return 1.0
    return float(other_alpha)


def _gaussian_smooth_1d(values, sigma_points=0.0):
    """Apply Gaussian smoothing on 1D data while preserving NaN gaps."""
    arr = np.asarray(values, dtype=float)
    sigma = float(sigma_points)
    if sigma <= 0.0 or arr.size < 3:
        return arr.copy()

    finite = np.isfinite(arr)
    if np.sum(finite) < 3:
        return arr.copy()

    idx = np.arange(arr.size)
    filled = arr.copy()
    missing = ~finite
    if np.any(missing):
        filled[missing] = np.interp(idx[missing], idx[finite], arr[finite])

    radius = max(1, int(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)

    smoothed = np.convolve(filled, kernel, mode='same')
    smoothed[missing] = np.nan
    return smoothed


def _find_standard_mc_pc(regret_results):
    """Return (mc_idx, pc_idx) for standard curves, if available."""
    mc_idx, pc_idx = None, None
    for idx, (_, algo_id, _) in enumerate(regret_results):
        if algo_id == 'mc_stratified':
            mc_idx = idx
        elif algo_id == 'pc':
            pc_idx = idx
    return mc_idx, pc_idx


def _interp_y_at_x(cost, y, x_target):
    """Interpolate y at x_target using sorted unique finite (x, y) pairs."""
    mask = np.isfinite(cost) & np.isfinite(y)
    if np.sum(mask) < 2:
        return np.nan
    df = pd.DataFrame({'x': cost[mask], 'y': y[mask]})
    df = df.sort_values('x').groupby('x', as_index=False)['y'].mean()
    xs = df['x'].to_numpy()
    ys = df['y'].to_numpy()
    if xs.size < 2 or x_target < xs[0] or x_target > xs[-1]:
        return np.nan
    return float(np.interp(x_target, xs, ys))


def _visible_tail_y(cost, y, x_cut):
    """Return the last finite y-value within the visible x-range (x <= x_cut)."""
    mask = np.isfinite(cost) & np.isfinite(y) & (cost <= x_cut)
    if np.sum(mask) == 0:
        return np.nan
    # Sort by x to robustly pick the visible tail, even if input is unsorted.
    order = np.argsort(cost[mask], kind='mergesort')
    y_sorted = y[mask][order]
    return float(y_sorted[-1])


def _is_mc_pc_family(algo_id):
    """Return True for MC/PC base and interactive variants."""
    return algo_id.startswith('mc_stratified') or algo_id.startswith('pc')


def _has_wallclock_time(exp):
    """Return whether a benchmark has wall-clock x-axis information."""
    return exp.startswith('jahs_') or exp == 'nas201'

def _regret_scales():
    return {
        #'jahs_cifar10': ((0, 0.3e8), (0.07, 0.3)),
        'jahs_cifar10': ((0, 0.57e8), (0.07, 0.2)),
        #'jahs_co': ((0, 0.2e6), (0.05, 0.135)),
        'jahs_co': ((0, 0.35e7), (0.045, 0.10)),
        #'jahs_fm': ((0, 0.9e8), (0.048, 0.115)),
        'jahs_fm': ((0, 0.42e8), (0.05, 0.08)),
        #'nas101': ((0, 1.5e6), (0.055, 0.07)),
        'nas101': ((0, 1.25e6), (0.055, 0.064)),
        'nas201': ((0, 10000), (0.083, 0.093)),
        'hpob_6767_31': [(0, 90), (0.0, 0.1)],
        'hpob_6767_9914': [(0, 90), (0.031, 0.055)],
        'hpob_6794_3': [(0, 90), (0.043, 0.06)],
        'hpob_6794_31': [(0, 90), (0.004, 0.012)],
        'hpob_6794_9914': [(0, 90), (0.01485, 0.0150)],
        'hpob_6794_10101': [(0, 200), (0.2045, 0.213)],
        'floris_cifar10': [(0, 200), (0, 4.0)],
        'pv_crossed_barrel': [(0, 240), (1.2, 2.3)],
        # 'hpob_4796_homo': [(0, 20), (0.59, 0.6)],
        'toy_branin': [(0, 2000), (-4, -1)],
        'rosenbrock_v3': [(0, 20), (0, 200)],
        'rosenbrock': [(0, 1000), (0, 20)],
        'quadratic': [(0, 100), (0, 8.)],
        'lc_adult': [(0, 100), (0, 1)],
        'fcnet_slice': [(0, 200), (0, 0.0015)]
    }


def _draw_regret_axis(ax, regret_results, exp, with_std=False,
                      other_alpha=0.35, algo_alphas=None,
                      smooth_sigma=0.0, highlight_mc_pc=False,
                      with_xlabel=True, with_ylabel=True,
                      unify_color=False):
    scales = _regret_scales()

    first_regrets = []
    for _, _, res in regret_results:
        y = np.asarray(res[:, 0], dtype=float)
        finite = np.where(np.isfinite(y))[0]
        if finite.size:
            first_regrets.append(float(abs(y[finite[0]])))
    norm_scale = max(first_regrets) if first_regrets else 1.0
    if not np.isfinite(norm_scale) or norm_scale <= 1e-12:
        norm_scale = 1.0

    curve_cache = {}
    for algo, algo_id, res in regret_results:
        mean_perf, perf_std, mean_cost = res[:, 0], res[:, 1], res[:, 2]
        mean_perf_plot = _gaussian_smooth_1d(mean_perf, sigma_points=smooth_sigma) / norm_scale
        perf_std_plot = perf_std / norm_scale
        curve_cache[algo_id] = (mean_cost, mean_perf_plot)
        if highlight_mc_pc and algo_alphas is None:
            alpha = 1.0 if _is_mc_pc_family(algo_id) else 0.3
        else:
            alpha = _resolve_curve_alpha(algo_id, other_alpha=other_alpha, algo_alphas=algo_alphas)
        c, line_type = _resolve_color_style(algo_id, unify_color=unify_color)
        ax.plot(mean_cost, mean_perf_plot, label=algo, alpha=alpha, linewidth=6., c=c, linestyle=line_type, zorder=3)
        if with_std:
            ax.fill_between(mean_cost, mean_perf_plot - perf_std_plot, mean_perf_plot + perf_std_plot, alpha=0.1, color=c, zorder=2)

    if with_xlabel:
        ax.set_xlabel('wall-clock time (sec.)', fontweight='bold')
    if with_ylabel:
        ax.set_ylabel('normalized regret', fontweight='bold', labelpad=10)

    if exp in scales:
        xlim, ylim = scales[exp]
        ax.set_xlim(*xlim)
        y0, y1 = float(ylim[0]), float(ylim[1])
        ax.set_ylim(y0 / norm_scale, y1 / norm_scale)
        # Force a finer x-tick grid on the few wide-cost benchmarks where
        # matplotlib's auto-locator picks step=100 and skips 150/250/etc.
        if exp in ('floris_cifar10', 'pv_crossed_barrel'):
            ax.xaxis.set_major_locator(MultipleLocator(50))

    if highlight_mc_pc and ('pc' in curve_cache) and ('mc_stratified' in curve_cache):
        pc_x, pc_y = curve_cache['pc']
        mc_x, mc_y = curve_cache['mc_stratified']
        x_cut = ax.get_xlim()[1]
        y_pc = _visible_tail_y(pc_x, pc_y, x_cut)
        y_mc = _visible_tail_y(mc_x, mc_y, x_cut)
        if np.isfinite(y_pc) and np.isfinite(y_mc):
            trans = blended_transform_factory(ax.transAxes, ax.transData)
            x_arrow = 1.03
            y_mid = 0.5 * (y_pc + y_mc)
            denom = abs(max(y_pc, 1e-12))
            reduction_norm_regret = (y_pc - y_mc) / denom * 100
            ax.annotate(
                '',
                xy=(x_arrow, y_mc),
                xytext=(x_arrow, y_pc),
                xycoords=trans,
                textcoords=trans,
                arrowprops=dict(arrowstyle='-|>', color='0.35', lw=3.2, shrinkA=0, shrinkB=0),
                annotation_clip=False,
                zorder=6,
            )
            ax.text(
                x_arrow + 0.065,
                y_mid,
                f"-{reduction_norm_regret:.2f}%",
                transform=trans,
                ha='center',
                va='center',
                rotation=90,
                color='0.30',
                fontsize=30,
                clip_on=False,
                zorder=7,
                weight='bold',
            )

    _apply_timeplot_axis_style(ax)
    ax.tick_params(axis='both', which='major', labelsize=24, length=11, width=2.2)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight('bold')
    ax.xaxis.get_offset_text().set_fontsize(20)
    ax.xaxis.get_offset_text().set_fontweight('bold')
    ax.grid(alpha=0.6)
    _hide_edge_ticklabels(ax)


def plot_regrets(regret_results, exp, file='./out.pdf', with_std=False,
                 with_legend=True, other_alpha=0.35, algo_alphas=None,
                 smooth_sigma=0.0, highlight_mc_pc=False):
    plt.rcParams.update({'font.size': 32, 'font.family':'serif'})
    fig, ax = plt.subplots(figsize=TIME_PLOT_FIGSIZE)
    _draw_regret_axis(
        ax,
        regret_results,
        exp,
        with_std=with_std,
        other_alpha=other_alpha,
        algo_alphas=algo_alphas,
        smooth_sigma=smooth_sigma,
        highlight_mc_pc=highlight_mc_pc,
        with_xlabel=True,
        with_ylabel=True,
    )
    if with_legend:
        ax.legend(fontsize=16)
    fig.subplots_adjust(**TIME_PLOT_MARGINS)
    fig.savefig(file)
    plt.close(fig)


def plot_ablations_combined(results_by_exp, ordered_labels, file='./out.pdf',
                            with_std=False, other_alpha=1.0,
                            smooth_sigma=0.0):
    """Single PDF, one panel per benchmark, shared bottom legend.

    ``results_by_exp``: list of ``(exp, regret_results)`` — each panel's data.
    ``ordered_labels``: list of ``(algo_id, custom_label)`` — defines legend
    order and override labels for both in-panel curves and the bottom legend.
    Curves whose algo_id is not in ``ordered_labels`` are dropped.
    """
    valid = [(exp, res) for exp, res in results_by_exp if res]
    if not valid:
        return

    label_map = dict(ordered_labels)
    keep_ids = [aid for aid, _ in ordered_labels]

    plt.rcParams.update({'font.size': 28, 'font.family': 'serif'})
    n_panels = len(valid)
    fig_w = max(8.0, TIME_PLOT_FIGSIZE[0] * n_panels)
    fig_h = TIME_PLOT_FIGSIZE[1]
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h))
    if n_panels == 1:
        axes = [axes]

    for idx, ((exp, res), ax) in enumerate(zip(valid, axes)):
        # Restrict + reorder + override display labels per the spec.
        by_id = {aid: tup for tup in res for _, aid, _ in [tup]}
        ordered_res = [
            (label_map[aid], aid, by_id[aid][2])
            for aid in keep_ids if aid in by_id
        ]
        if not ordered_res:
            continue
        _draw_regret_axis(
            ax, ordered_res, exp,
            with_std=with_std,
            other_alpha=other_alpha,
            algo_alphas=None,
            smooth_sigma=smooth_sigma,
            highlight_mc_pc=False,
            with_xlabel=True,
            with_ylabel=(idx == 0),
        )
        ax.set_title(BENCHMARK_LABELS.get(exp, exp),
                     fontsize=32, pad=10, fontweight='bold')

    # Shared bottom legend: split into a top row of "c + n" variants and a
    # bottom row of "the rest" (β / no-perturbation). Matplotlib fills the
    # legend column-major, so we interleave (top, bottom) per column.
    from matplotlib.lines import Line2D
    handles, labels = [], []
    for aid, label in ordered_labels:
        if aid not in ALGO_COLORS:
            continue
        color, line_style = ALGO_COLORS[aid]
        handles.append(Line2D([0], [0], color=color, linestyle=line_style,
                              linewidth=7.0))
        labels.append(label)

    if handles:
        # First entries whose label starts with 'c=' or 'n=' go on the top row.
        top_idx = [i for i, l in enumerate(labels)
                   if l.startswith('c=') or l.startswith('n=')]
        bot_idx = [i for i in range(len(labels)) if i not in set(top_idx)]
        ncol = max(len(top_idx), len(bot_idx))
        ordered_h, ordered_l = [], []
        for c in range(ncol):
            if c < len(top_idx):
                ordered_h.append(handles[top_idx[c]])
                ordered_l.append(labels[top_idx[c]])
            if c < len(bot_idx):
                ordered_h.append(handles[bot_idx[c]])
                ordered_l.append(labels[bot_idx[c]])
        fig.legend(
            ordered_h, ordered_l,
            loc='lower center', bbox_to_anchor=(0.5, 0.01),
            frameon=False,
            ncol=ncol,
            handlelength=2.6, columnspacing=1.4,
            prop={'weight': 'bold', 'size': 24},
        )
        bottom = 0.35
    else:
        bottom = max(TIME_PLOT_MARGINS.get('bottom', 0.17), 0.18)

    fig.subplots_adjust(
        left=0.10, right=0.97,
        top=TIME_PLOT_MARGINS.get('top', 0.92),
        bottom=bottom,
        wspace=0.40,
    )
    fig.savefig(file)
    plt.close(fig)
def plot_regrets_row(results_by_exp, file='./out.pdf', with_std=False,
                     with_legend=True, other_alpha=0.35, algo_alphas=None,
                     smooth_sigma=0.0, highlight_mc_pc=False):
    """Plot multiple benchmark regret panels in one horizontal row."""
    if not results_by_exp:
        return

    valid = [(exp, res) for exp, res in results_by_exp if res]
    if not valid:
        return

    plt.rcParams.update({'font.size': 28, 'font.family': 'serif'})
    n_panels = len(valid)
    fig_w = max(8.0, TIME_PLOT_FIGSIZE[0] * n_panels)
    fig_h = TIME_PLOT_FIGSIZE[1]
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h))
    if n_panels == 1:
        axes = [axes]

    legend_algo_ids = []
    for idx, ((exp, res), ax) in enumerate(zip(valid, axes)):
        _draw_regret_axis(
            ax,
            res,
            exp,
            with_std=with_std,
            other_alpha=other_alpha,
            algo_alphas=algo_alphas,
            smooth_sigma=smooth_sigma,
            highlight_mc_pc=highlight_mc_pc,
            with_xlabel=True,
            with_ylabel=(idx == 0),
        )
        ax.set_title(BENCHMARK_LABELS.get(exp, exp), fontsize=35, pad=10, fontweight='bold')

        for _, algo_id, _ in res:
            if algo_id not in legend_algo_ids:
                legend_algo_ids.append(algo_id)

    bottom = max(TIME_PLOT_MARGINS.get('bottom', 0.17), 0.22)
    if with_legend and legend_algo_ids:
        from matplotlib.lines import Line2D

        handles = []
        labels = []
        for algo_id in legend_algo_ids:
            color, line_style = ALGO_COLORS[algo_id]
            if highlight_mc_pc and algo_alphas is None:
                alpha = 1.0 if _is_mc_pc_family(algo_id) else 0.3
            else:
                alpha = _resolve_curve_alpha(algo_id, other_alpha=other_alpha, algo_alphas=algo_alphas)
            handles.append(Line2D([0], [0], color=color, linestyle=line_style, linewidth=7.0, alpha=alpha))
            labels.append(ALGO_NAME_MAPPING.get(algo_id, algo_id))

        ncol = len(handles)
        fig.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.01),
            frameon=False,
            fontsize=33,
            ncol=ncol,
            handlelength=2.8,
            columnspacing=1.4,
            prop={'weight': 'bold', 'size': 33},
        )
        bottom = max(bottom, 0.31)

    fig.subplots_adjust(
        left=0.08,
        right=0.96,
        top=TIME_PLOT_MARGINS.get('top', 0.94),
        bottom=bottom,
        wspace=0.40,
    )
    fig.savefig(file)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Multi-row "grid" plotters: shared bottom legend, one panel per benchmark.
# Cells in the input grid may be (exp, results), (exp, []), or None to leave
# the panel blank.
# ----------------------------------------------------------------------------

def _grid_dims(grid_cells):
    if not grid_cells:
        return 0, 0
    n_rows = len(grid_cells)
    n_cols = max(len(row) for row in grid_cells) if n_rows else 0
    return n_rows, n_cols


def _normalize_axes(axes, n_rows, n_cols):
    """matplotlib returns differently-shaped objects depending on n_rows/n_cols.
    Make `axes[r][c]` always work."""
    import numpy as _np
    arr = _np.atleast_2d(_np.asarray(axes, dtype=object))
    if n_rows == 1 and arr.shape[0] != 1:
        arr = arr.reshape(1, -1)
    if n_cols == 1 and arr.shape[1] != 1:
        arr = arr.reshape(-1, 1)
    return arr.tolist()


def _grid_legend_handles(ordered_algo_ids, *, highlight_mc_pc=False,
                         other_alpha=0.35, algo_alphas=None,
                         baseline_algo=None, unify_color=False):
    """Build (handles, labels) for the shared bottom legend.

    Uses `_short_label` so suffixes like "(prior@5)" / "(stratified)" are
    stripped — the legend shows the base method name only ("DisCoMBO",
    "IBO", "πBO", "DynaBO", ...). Duplicate labels are folded into a
    single legend entry, keeping the first algo's color/style.
    """
    from matplotlib.lines import Line2D
    handles, labels = [], []
    seen = set()
    for algo_id in ordered_algo_ids:
        if algo_id not in ALGO_COLORS:
            continue
        label = _short_label(algo_id)
        if label in seen:
            continue
        seen.add(label)
        color, line_style = _resolve_color_style(algo_id, unify_color=unify_color)
        if highlight_mc_pc and algo_alphas is None:
            alpha = 1.0 if _is_mc_pc_family(algo_id) else 0.3
        else:
            alpha = _resolve_curve_alpha(algo_id, other_alpha=other_alpha,
                                         algo_alphas=algo_alphas)
        handles.append(Line2D([0], [0], color=color, linestyle=line_style,
                              linewidth=7.0, alpha=alpha))
        labels.append(label)
    if baseline_algo is not None:
        handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=3.0))
        labels.append(f'{_short_label(baseline_algo)} baseline')
    return handles, labels


def _grid_make_figure(n_rows, n_cols):
    plt.rcParams.update({'font.size': 28, 'font.family': 'serif'})
    fig_w = max(8.0, TIME_PLOT_FIGSIZE[0] * n_cols)
    fig_h = TIME_PLOT_FIGSIZE[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes = _normalize_axes(axes, n_rows, n_cols)
    return fig, axes


_GRID_LEGEND_TOP_ROW = ('DisCoMBO', 'IBO-HPC', 'DynaBO')
_GRID_LEGEND_BOTTOM_ROW = ('CMA-ES', 'HEBO', 'BO w/ TPE', 'BO w/ RF', 'SMAC')


def _grid_finalize(fig, handles, labels, *, has_bottom_legend):
    bottom = max(TIME_PLOT_MARGINS.get('bottom', 0.17), 0.10)
    if has_bottom_legend and handles:
        top_idx = [i for i, l in enumerate(labels) if l in _GRID_LEGEND_TOP_ROW]
        bot_idx = [i for i, l in enumerate(labels) if l in _GRID_LEGEND_BOTTOM_ROW]
        other_idx = [i for i in range(len(labels))
                     if i not in set(top_idx) and i not in set(bot_idx)]
        # Use the two-row layout only when both groups are populated AND the
        # legend wouldn't lose entries (any "other" labels join the bottom row).
        if top_idx and bot_idx:
            bot_idx = bot_idx + other_idx
            ncol = max(len(top_idx), len(bot_idx))
            # Center the (shorter) top row above the (longer) bottom row.
            top_offset = max(0, (ncol - len(top_idx)) // 2)
            from matplotlib.lines import Line2D
            # Pad to ncol*2 entries: invisible placeholders fill empty slots
            # so matplotlib's column-major layout keeps nrows=2 strictly.
            ordered_h = [None] * (ncol * 2)
            ordered_l = [''] * (ncol * 2)
            for c in range(ncol):
                top_c = c - top_offset
                if 0 <= top_c < len(top_idx):
                    ordered_h[c * 2 + 0] = handles[top_idx[top_c]]
                    ordered_l[c * 2 + 0] = labels[top_idx[top_c]]
                if c < len(bot_idx):
                    ordered_h[c * 2 + 1] = handles[bot_idx[c]]
                    ordered_l[c * 2 + 1] = labels[bot_idx[c]]
            for i, h in enumerate(ordered_h):
                if h is None:
                    ordered_h[i] = Line2D([0], [0], visible=False)
            fig.legend(
                ordered_h, ordered_l,
                loc='lower center',
                bbox_to_anchor=(0.5, 0.005),
                frameon=False,
                ncol=ncol,
                handlelength=2.6,
                columnspacing=1.4,
                prop={'weight': 'bold', 'size': 26},
            )
            bottom = 0.14
        else:
            # Single-row legend (one ncol per entry).
            fig.legend(
                handles, labels,
                loc='lower center',
                bbox_to_anchor=(0.5, 0.005),
                frameon=False,
                ncol=len(handles),
                handlelength=2.6,
                columnspacing=1.4,
                prop={'weight': 'bold', 'size': 26},
            )
            bottom = 0.13
    fig.subplots_adjust(
        left=0.10, right=0.97,
        top=0.95, bottom=bottom,
        wspace=0.30, hspace=0.55,
    )


def plot_regrets_grid(grid_cells, file='./out.pdf', with_std=False,
                      with_legend=True, other_alpha=0.35, algo_alphas=None,
                      smooth_sigma=0.0, highlight_mc_pc=False):
    """Multi-row regret panels in a single PDF, shared bottom legend.

    grid_cells: 2D list of (exp, regret_results) tuples (or None to leave
    a panel blank).
    """
    n_rows, n_cols = _grid_dims(grid_cells)
    if n_rows == 0 or n_cols == 0:
        return
    fig, axes = _grid_make_figure(n_rows, n_cols)

    legend_algo_ids = []
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            entry = grid_cells[r][c] if c < len(grid_cells[r]) else None
            if entry is None or not entry[1]:
                ax.axis('off')
                continue
            exp, res = entry
            _draw_regret_axis(
                ax, res, exp,
                with_std=with_std,
                other_alpha=other_alpha,
                algo_alphas=algo_alphas,
                smooth_sigma=smooth_sigma,
                highlight_mc_pc=highlight_mc_pc,
                with_xlabel=True,
                with_ylabel=(c == 0),
                unify_color=True,
            )
            ax.set_title(BENCHMARK_LABELS.get(exp, exp),
                         fontsize=32, pad=10, fontweight='bold')
            for _, algo_id, _ in res:
                if algo_id not in legend_algo_ids:
                    legend_algo_ids.append(algo_id)

    handles, labels = _grid_legend_handles(
        legend_algo_ids, highlight_mc_pc=highlight_mc_pc,
        other_alpha=other_alpha, algo_alphas=algo_alphas,
        unify_color=True,
    )
    _grid_finalize(fig, handles, labels, has_bottom_legend=with_legend)
    fig.savefig(file)
    plt.close(fig)


def plot_rankings_grid(grid_cells, file='./out.pdf', with_legend=True,
                       n_grid=200, other_alpha=1.0, algo_alphas=None):
    """Multi-row ranking panels (one per benchmark)."""
    n_rows, n_cols = _grid_dims(grid_cells)
    if n_rows == 0 or n_cols == 0:
        return
    fig, axes = _grid_make_figure(n_rows, n_cols)

    legend_algo_ids = []
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            entry = grid_cells[r][c] if c < len(grid_cells[r]) else None
            if entry is None or not entry[1]:
                ax.axis('off')
                continue
            exp, res = entry
            drew = _draw_ranking_axis(
                ax, res, n_grid=n_grid,
                other_alpha=other_alpha, algo_alphas=algo_alphas,
                with_xlabel=True,
                with_ylabel=(c == 0),
                unify_color=True,
            )
            if not drew:
                ax.axis('off')
                continue
            ax.set_title(BENCHMARK_LABELS.get(exp, exp),
                         fontsize=32, pad=10, fontweight='bold')
            for _, algo_id, _ in res:
                if algo_id not in legend_algo_ids:
                    legend_algo_ids.append(algo_id)

    handles, labels = _grid_legend_handles(
        legend_algo_ids, other_alpha=other_alpha, algo_alphas=algo_alphas,
        unify_color=True,
    )
    _grid_finalize(fig, handles, labels, has_bottom_legend=with_legend)
    fig.savefig(file)
    plt.close(fig)


def plot_vs_pc_baseline_diff_grid(grid_cells, file='./out.pdf',
                                  baseline_algo='pc', with_legend=True,
                                  n_grid=200, other_alpha=1.0,
                                  algo_alphas=None, eps=1e-9,
                                  min_baseline_frac=0.03,
                                  max_abs_improvement_pct=100.0,
                                  smooth_sigma=0.0):
    """Multi-row "% improvement vs <baseline>" panels (one per benchmark)."""
    n_rows, n_cols = _grid_dims(grid_cells)
    if n_rows == 0 or n_cols == 0:
        return
    fig, axes = _grid_make_figure(n_rows, n_cols)

    legend_algo_ids = []
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            entry = grid_cells[r][c] if c < len(grid_cells[r]) else None
            if entry is None or not entry[1]:
                ax.axis('off')
                continue
            exp, res = entry
            drew, plotted_ids = _draw_vs_pc_diff_axis(
                ax, res, baseline_algo=baseline_algo,
                n_grid=n_grid, other_alpha=other_alpha,
                algo_alphas=algo_alphas, eps=eps,
                min_baseline_frac=min_baseline_frac,
                max_abs_improvement_pct=max_abs_improvement_pct,
                smooth_sigma=smooth_sigma,
                with_xlabel=True,
                with_ylabel=(c == 0),
                unify_color=True,
            )
            if not drew:
                ax.axis('off')
                continue
            ax.set_title(BENCHMARK_LABELS.get(exp, exp),
                         fontsize=32, pad=10, fontweight='bold')
            for algo_id in plotted_ids:
                if algo_id not in legend_algo_ids:
                    legend_algo_ids.append(algo_id)

    handles, labels = _grid_legend_handles(
        legend_algo_ids, other_alpha=other_alpha, algo_alphas=algo_alphas,
        baseline_algo=baseline_algo, unify_color=True,
    )
    _grid_finalize(fig, handles, labels, has_bottom_legend=with_legend)
    fig.savefig(file)
    plt.close(fig)


def _roc_curve_numpy(labels, scores):
    """Compute ROC points (fpr, tpr) without sklearn.

    labels: binary array-like with values in {0, 1}
    scores: continuous array-like; higher score => more likely positive
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if labels.size == 0 or scores.size == 0 or labels.size != scores.size:
        raise ValueError('labels and scores must be non-empty and same-length.')

    pos_total = int(np.sum(labels == 1))
    neg_total = int(np.sum(labels == 0))
    if pos_total == 0 or neg_total == 0:
        raise ValueError('Both positive and negative labels are required for ROC.')

    # Stable descending sort to preserve deterministic behavior on ties.
    order = np.argsort(scores, kind='mergesort')[::-1]
    y = labels[order]
    s = scores[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)

    # Keep only last index of each threshold plateau.
    distinct = np.r_[s[1:] != s[:-1], True]
    tp = tp[distinct]
    fp = fp[distinct]

    tpr = tp / pos_total
    fpr = fp / neg_total

    # Include origin to match standard ROC output shape.
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    return fpr, tpr


def _draw_intervention_roc_axis(ax, all_df, benchmark, task,
                                with_xlabel=True, with_ylabel=False):
    """Draw one intervention-classification ROC axis.

    Returns a list of plotted method ids in draw order.
    """
    task = task or ''
    sub = all_df[(all_df['benchmark'] == benchmark)
                 & (all_df['task'].fillna('') == task)]
    sub = sub.dropna(subset=['metric'])

    title = INTERVENTION_BENCHMARK_DISPLAY.get(
        (benchmark, task), _intervention_key_label(benchmark, task)
    )

    ax.plot([0, 1], [0, 1], color='#888888', linestyle=':', lw=1.5, zorder=0)
    plotted_methods = []
    for method in ('pval', 'score'):
        method_df = sub[sub['method'] == method]
        if method_df.empty or method_df['label'].nunique() < 2:
            continue
        labels = (method_df['label'] == 'good').astype(int).values
        signed = INTERVENTION_NATURAL_DIRECTION[method] * method_df['metric'].values
        try:
            fpr, tpr = _roc_curve_numpy(labels, signed)
        except ValueError:
            continue
        ax.plot(
            fpr,
            tpr,
            color=INTERVENTION_METHOD_COLOR[method],
            lw=6.0,
            linestyle=INTERVENTION_METHOD_LINESTYLE[method],
            zorder=2,
        )
        plotted_methods.append(method)

    ax.set_title(title, fontsize=35, pad=10, fontweight='bold')
    if with_xlabel:
        ax.set_xlabel('false positive rate', fontweight='bold')
    if with_ylabel:
        ax.set_ylabel('true positive rate', fontweight='bold')

    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(-0.025, 1.025)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, linestyle='-', alpha=0.6, linewidth=1.1)
    ax.tick_params(axis='both', which='major', labelsize=24, length=11, width=2.2)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight('bold')
    _hide_edge_ticklabels(ax)

    return plotted_methods


def _add_wrapped_centered_legend(fig, handles, labels, max_per_row=5,
                                 base_y=0.01, row_step=0.07,
                                 fontsize=20, handlelength=2.8,
                                 columnspacing=1.4):
    """Add wrapped figure-level legends with centered final row."""
    if not handles:
        return 0

    rows = [
        (handles[i:i + max_per_row], labels[i:i + max_per_row])
        for i in range(0, len(handles), max_per_row)
    ]
    for row_idx, (row_handles, row_labels) in enumerate(rows):
        fig.legend(
            row_handles,
            row_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, base_y + row_idx * row_step),
            frameon=False,
            ncol=len(row_handles),
            fontsize=fontsize,
            handlelength=handlelength,
            columnspacing=columnspacing,
            prop={'weight': 'bold', 'size': fontsize},
        )
    return len(rows)


def plot_regret_roc_row(regret_results_by_exp, all_df, roc_keys,
                        file='./out.pdf', with_std=False,
                        other_alpha=0.35, algo_alphas=None,
                        smooth_sigma=0.0, highlight_mc_pc=False):
    """Plot 2 regret panels + 2 intervention ROC panels in one row."""
    from matplotlib.lines import Line2D

    if len(regret_results_by_exp) != 2:
        raise ValueError('Expected exactly 2 regret benchmarks for mixed row plot.')
    if len(roc_keys) != 2:
        raise ValueError('Expected exactly 2 ROC keys for mixed row plot.')

    plt.rcParams.update({'font.size': 28, 'font.family': 'serif'})
    n_panels = 4
    fig_w = max(8.0, TIME_PLOT_FIGSIZE[0] * n_panels)
    fig_h = TIME_PLOT_FIGSIZE[1]
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h))

    legend_map = {}

    # Two regret panels.
    for idx, (exp, res) in enumerate(regret_results_by_exp):
        ax = axes[idx]
        _draw_regret_axis(
            ax,
            res,
            exp,
            with_std=with_std,
            other_alpha=other_alpha,
            algo_alphas=algo_alphas,
            smooth_sigma=smooth_sigma,
            highlight_mc_pc=highlight_mc_pc,
            with_xlabel=True,
            with_ylabel=(idx == 0),
        )
        ax.set_title(BENCHMARK_LABELS.get(exp, exp), fontsize=35, pad=10, fontweight='bold')

        for _, algo_id, _ in res:
            color, line_style = ALGO_COLORS[algo_id]
            if highlight_mc_pc and algo_alphas is None:
                alpha = 1.0 if _is_mc_pc_family(algo_id) else 0.3
            else:
                alpha = _resolve_curve_alpha(algo_id, other_alpha=other_alpha, algo_alphas=algo_alphas)
            label = ALGO_NAME_MAPPING.get(algo_id, algo_id)
            if label not in legend_map:
                legend_map[label] = Line2D([0], [0], color=color, linestyle=line_style,
                                           linewidth=7.0, alpha=alpha)

    # Two ROC panels.
    for idx, (benchmark, task) in enumerate(roc_keys):
        ax = axes[2 + idx]
        methods = _draw_intervention_roc_axis(
            ax,
            all_df,
            benchmark,
            task,
            with_xlabel=True,
            with_ylabel=(idx == 0),
        )
        for method in methods:
            label = INTERVENTION_METHOD_DISPLAY[method]
            if label not in legend_map:
                legend_map[label] = Line2D(
                    [0],
                    [0],
                    color=INTERVENTION_METHOD_COLOR[method],
                    linestyle=INTERVENTION_METHOD_LINESTYLE[method],
                    linewidth=7.0,
                    alpha=1.0,
                )

    handles = list(legend_map.values())
    labels = list(legend_map.keys())
    n_rows = _add_wrapped_centered_legend(
        fig,
        handles,
        labels,
        max_per_row=5,
        base_y=-0.05,
        row_step=0.07,
        fontsize=30,
        handlelength=2.8,
        columnspacing=1.4,
    )

    bottom = max(TIME_PLOT_MARGINS.get('bottom', 0.17), 0.22)
    if n_rows > 0:
        bottom = max(bottom, 0.24 + 0.07 * (n_rows - 1))

    fig.subplots_adjust(
        left=0.08,
        right=0.96,
        top=TIME_PLOT_MARGINS.get('top', 0.94),
        bottom=bottom,
        wspace=0.40,
    )
    fig.savefig(file)
    plt.close(fig)


def _draw_ranking_axis(ax, regret_results, n_grid=200, other_alpha=1.0,
                       algo_alphas=None, with_xlabel=True, with_ylabel=True,
                       unify_color=False):
    """Draw rank-vs-cost on a single axis. Returns True iff anything was
    drawn (>=2 algos with usable data and a non-empty cost overlap).
    """
    if len(regret_results) < 2:
        return False
    cmin = max(np.nanmin(res[:, 2]) for _, _, res in regret_results)
    cmax = min(np.nanmax(res[:, 2]) for _, _, res in regret_results)
    if not (np.isfinite(cmin) and np.isfinite(cmax)) or cmin >= cmax:
        return False
    grid = np.linspace(cmin, cmax, n_grid)

    interp = []
    for _, _, res in regret_results:
        cost = res[:, 2]
        regret = res[:, 0]
        mask = np.isfinite(cost) & np.isfinite(regret)
        if mask.sum() < 2:
            interp.append(np.full(grid.shape, np.nan))
            continue
        order = np.argsort(cost[mask])
        c_sorted = cost[mask][order]
        r_sorted = regret[mask][order]
        df = pd.DataFrame({'c': c_sorted, 'r': r_sorted})
        df = df.groupby('c', as_index=False)['r'].min()
        interp.append(np.interp(grid, df['c'].to_numpy(), df['r'].to_numpy()))

    regrets_grid = np.array(interp)
    ranks = np.full(regrets_grid.shape, np.nan)
    for j in range(regrets_grid.shape[1]):
        col = regrets_grid[:, j]
        valid = np.isfinite(col)
        if valid.sum() < 2:
            continue
        ranks[valid, j] = rankdata(col[valid], method='average')

    for i, (label, algo_id, _) in enumerate(regret_results):
        alpha = _resolve_curve_alpha(algo_id, other_alpha=other_alpha, algo_alphas=algo_alphas)
        c, line_type = _resolve_color_style(algo_id, unify_color=unify_color)
        ax.plot(grid, ranks[i], label=label, alpha=alpha, linewidth=6.,
                c=c, linestyle=line_type, zorder=3)

    if with_xlabel:
        ax.set_xlabel('wall-clock time (sec.)', fontweight='bold')
    if with_ylabel:
        ax.set_ylabel('average rank', fontweight='bold', labelpad=10)
    ax.invert_yaxis()
    _apply_timeplot_axis_style(ax)
    # Ranks are integers; override the time-plot float formatter so y-tick
    # labels render as plain integers ('1', '2', ...).
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.tick_params(axis='both', which='major', labelsize=24, length=11, width=2.2)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight('bold')
    ax.xaxis.get_offset_text().set_fontsize(20)
    ax.xaxis.get_offset_text().set_fontweight('bold')
    ax.grid(alpha=0.6)
    _hide_edge_ticklabels(ax)
    return True


def plot_rankings(regret_results, exp, file='./out.pdf', with_legend=True,
                  n_grid=200, other_alpha=1.0, algo_alphas=None):
    """Plot rank-vs-cost. Same input shape as plot_regrets."""
    plt.rcParams.update({'font.size': 32, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=TIME_PLOT_FIGSIZE)
    drew = _draw_ranking_axis(
        ax, regret_results, n_grid=n_grid,
        other_alpha=other_alpha, algo_alphas=algo_alphas,
        with_xlabel=True, with_ylabel=True,
    )
    if not drew:
        plt.close(fig)
        return
    if with_legend:
        ax.legend(fontsize=16)
    fig.subplots_adjust(**TIME_PLOT_MARGINS)
    fig.savefig(file)
    plt.close(fig)


def _interp_regret_on_grid(res, grid):
    cost = res[:, 2]
    regret = res[:, 0]
    mask = np.isfinite(cost) & np.isfinite(regret)
    if mask.sum() < 2:
        return np.full(grid.shape, np.nan)
    order = np.argsort(cost[mask])
    df = pd.DataFrame({'c': cost[mask][order], 'r': regret[mask][order]})
    df = df.groupby('c', as_index=False)['r'].min()
    return np.interp(grid, df['c'].to_numpy(), df['r'].to_numpy())


def plot_pc_vs_mc_diff(regret_results, pairs, exp, file='./out.pdf',
                       with_legend=True, n_grid=200, eps=1e-9):
    """Plot per-bucket relative improvement of PC over MC vs cost.

    For each (mc_algo, pc_algo) pair, interpolates both algos' mean regret onto
    a shared cost grid and plots ``(MC_regret - PC_regret) / MC_regret * 100``.
    Positive = PC has lower regret (improvement); negative = PC regression.
    Uses the PC algo's color/linestyle so curves visually match other plots.
    """
    plt.rcParams.update({'font.size': 32, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=TIME_PLOT_FIGSIZE)

    res_by_id = {aid: res for _, aid, res in regret_results}
    valid_pairs = [(m, p) for m, p in pairs if m in res_by_id and p in res_by_id]
    if not valid_pairs:
        plt.close(fig)
        return

    cmins, cmaxs = [], []
    for m, p in valid_pairs:
        cmins.append(np.nanmin(res_by_id[m][:, 2]))
        cmins.append(np.nanmin(res_by_id[p][:, 2]))
        cmaxs.append(np.nanmax(res_by_id[m][:, 2]))
        cmaxs.append(np.nanmax(res_by_id[p][:, 2]))
    cmin, cmax = max(cmins), min(cmaxs)
    if not (np.isfinite(cmin) and np.isfinite(cmax)) or cmin >= cmax:
        plt.close(fig)
        return
    grid = np.linspace(cmin, cmax, n_grid)

    for mc_id, pc_id in valid_pairs:
        mc_r = _interp_regret_on_grid(res_by_id[mc_id], grid)
        pc_r = _interp_regret_on_grid(res_by_id[pc_id], grid)
        denom = np.where(np.abs(mc_r) < eps, np.nan, mc_r)
        diff = (mc_r - pc_r) / denom * 100.0
        _, line_type = ALGO_COLORS.get(pc_id, ('red', '-'))
        mc_name = _short_label(mc_id)
        pc_name = _short_label(pc_id)
        ax.plot(grid, diff, label=f'{pc_name} vs {mc_name}',
                alpha=1.0, linewidth=6., c='red', linestyle=line_type, zorder=3)

    ax.axhline(0, color='blue', linewidth=1.4, linestyle='--', alpha=0.8)
    ax.set_xlabel('wall-clock time (sec.)', fontweight='bold')
    ax.set_ylabel('(MC - PC) / MC [%]', fontweight='bold')
    _apply_timeplot_axis_style(ax)
    ax.tick_params(axis='both', which='major', labelsize=24, length=11, width=2.2)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight('bold')
    ax.xaxis.get_offset_text().set_fontsize(20)
    ax.xaxis.get_offset_text().set_fontweight('bold')
    if with_legend:
        ax.legend(fontsize=14)
    ax.grid(alpha=0.6)
    _hide_edge_ticklabels(ax)
    fig.subplots_adjust(**TIME_PLOT_MARGINS)
    fig.savefig(file)
    plt.close(fig)


def _draw_vs_pc_diff_axis(ax, regret_results, baseline_algo='pc',
                          n_grid=200, other_alpha=1.0, algo_alphas=None,
                          eps=1e-9, min_baseline_frac=0.03,
                          max_abs_improvement_pct=100.0, smooth_sigma=0.0,
                          with_xlabel=True, with_ylabel=True,
                          unify_color=False):
    """Draw "% improvement vs <baseline>" curves on a single axis.

    Returns (drew, plotted_algo_ids). `drew` is False if the baseline isn't
    present, the cost overlap is empty, or no compare algos remain.
    """
    res_by_id = {aid: res for _, aid, res in regret_results}
    if baseline_algo not in res_by_id:
        return False, []

    compare_algos = [aid for _, aid, _ in regret_results if aid != baseline_algo]
    if not compare_algos:
        return False, []

    selected = [baseline_algo] + compare_algos
    cmin = max(np.nanmin(res_by_id[aid][:, 2]) for aid in selected)
    cmax = min(np.nanmax(res_by_id[aid][:, 2]) for aid in selected)
    if not (np.isfinite(cmin) and np.isfinite(cmax)) or cmin >= cmax:
        return False, []

    grid = np.linspace(cmin, cmax, n_grid)
    baseline_regret = _interp_regret_on_grid(res_by_id[baseline_algo], grid)
    baseline_name = _short_label(baseline_algo)

    baseline_abs = np.abs(baseline_regret)
    baseline_floor = max(eps, min_baseline_frac * np.nanmax(baseline_abs))
    valid_baseline = baseline_abs >= baseline_floor

    plotted_curves = []
    plotted_ids = []
    for algo_id in compare_algos:
        algo_regret = _interp_regret_on_grid(res_by_id[algo_id], grid)
        denom = np.where(valid_baseline, baseline_regret, np.nan)
        improvement_pct = (baseline_regret - algo_regret) / denom * 100.0
        improvement_pct = np.where(np.abs(improvement_pct) > max_abs_improvement_pct, np.nan, improvement_pct)
        improvement_plot = _gaussian_smooth_1d(improvement_pct, sigma_points=smooth_sigma)
        label = _short_label(algo_id)
        c, line_type = _resolve_color_style(algo_id, unify_color=unify_color)
        alpha = _resolve_curve_alpha(algo_id, other_alpha=other_alpha, algo_alphas=algo_alphas)
        ax.plot(grid, improvement_plot, label=label, alpha=alpha, linewidth=6.,
                c=c, linestyle=line_type, zorder=3)
        plotted_curves.append(improvement_plot)
        plotted_ids.append(algo_id)

    ax.axhline(0, color='black', linewidth=1.4, linestyle='--', alpha=0.8,
               label=f'{baseline_name} baseline')
    if with_xlabel:
        ax.set_xlabel('wall-clock time (sec.)', fontweight='bold')
    if with_ylabel:
        ax.set_ylabel(f'% improvement vs {baseline_name}', fontweight='bold', labelpad=10)

    if plotted_curves:
        all_vals = np.concatenate([v[np.isfinite(v)] for v in plotted_curves if np.any(np.isfinite(v))])
        if all_vals.size > 0:
            y_lo = float(np.percentile(all_vals, 3.0))
            y_hi = float(np.percentile(all_vals, 98.0))
            if np.isfinite(y_lo) and np.isfinite(y_hi):
                span = max(1.0, y_hi - y_lo)
                pad = 0.12 * span
                lower = min(y_lo - pad, -1.0)
                upper = max(y_hi + pad, 1.0)
                ax.set_ylim(lower, upper)

    _apply_timeplot_axis_style(ax)
    ax.tick_params(axis='both', which='major', labelsize=24, length=11, width=2.2)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight('bold')
    ax.xaxis.get_offset_text().set_fontsize(20)
    ax.xaxis.get_offset_text().set_fontweight('bold')
    ax.grid(alpha=0.6)
    _hide_edge_ticklabels(ax)
    return True, plotted_ids


def plot_vs_pc_baseline_diff(regret_results, exp, file='./out.pdf',
                             baseline_algo='pc', with_legend=True,
                             n_grid=200, other_alpha=1.0,
                             algo_alphas=None, eps=1e-9,
                             min_baseline_frac=0.03,
                             max_abs_improvement_pct=100.0,
                             smooth_sigma=0.0):
    """Plot regret deltas to a PC baseline over time."""
    plt.rcParams.update({'font.size': 32, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=TIME_PLOT_FIGSIZE)
    drew, _ = _draw_vs_pc_diff_axis(
        ax, regret_results, baseline_algo=baseline_algo,
        n_grid=n_grid, other_alpha=other_alpha, algo_alphas=algo_alphas,
        eps=eps, min_baseline_frac=min_baseline_frac,
        max_abs_improvement_pct=max_abs_improvement_pct,
        smooth_sigma=smooth_sigma,
        with_xlabel=True, with_ylabel=True,
    )
    if not drew:
        plt.close(fig)
        return
    if with_legend:
        ax.legend(fontsize=14)
    fig.subplots_adjust(**TIME_PLOT_MARGINS)
    fig.savefig(file)
    plt.close(fig)


def plot_best_vs_pc_bar(algo_stats, exp, file='./out.pdf', with_legend=False,
                        other_alpha=1.0, algo_alphas=None):
    """Plot mean±std of run-level best % improvements vs PC per algorithm."""
    if not algo_stats:
        return

    stats_sorted = sorted(algo_stats, key=lambda x: x[2], reverse=True)
    labels = [row[0] for row in stats_sorted]
    algo_ids = [row[1] for row in stats_sorted]
    means = np.array([row[2] for row in stats_sorted], dtype=float)
    stds = np.array([row[3] for row in stats_sorted], dtype=float)

    plt.rcParams.update({'font.size': 26, 'font.family': 'serif'})
    fig_w = max(8.5, 0.62 * len(labels) + 2.5)
    fig_h = 6.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = np.arange(len(labels), dtype=float)
    colors = [ALGO_COLORS.get(aid, ('#555555', 'solid'))[0] for aid in algo_ids]
    alphas = [_resolve_curve_alpha(aid, other_alpha=other_alpha, algo_alphas=algo_alphas) for aid in algo_ids]

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        width=0.82,
        color=colors,
        edgecolor='black',
        linewidth=0.6,
        capsize=4,
        alpha=1.0,
    )
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)

    ax.axhline(0.0, color='black', linewidth=1.3, linestyle='--', alpha=0.8)
    ax.set_ylabel('% improvement vs IBO-HPC')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.margins(x=0.01)
    ax.grid(axis='y', alpha=0.35, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if with_legend:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], color=ALGO_COLORS.get(aid, ('#555555', 'solid'))[0], lw=6,
                   alpha=_resolve_curve_alpha(aid, other_alpha=other_alpha, algo_alphas=algo_alphas),
                   label=lbl)
            for lbl, aid in zip(labels, algo_ids)
        ]
        ax.legend(handles=handles, fontsize=12, frameon=False, ncol=2)

    fig.tight_layout(pad=0.25)
    fig.savefig(file, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def plot_avg_regrets_per_iter(exp, file):
    """
        Plot the average regret in each iteration (not the best until here)
    """
    prefix = f'../../data/logs/{exp}_pc_int_'
    search_spaces = ['multi']
    search_space_regrets = {}
    for space in search_spaces:
        path = prefix + space
        dfs = read_csvs(path)

        # for each df, compute regrets at each sample
        regrets = []
        for df in dfs:
            rgr = []
            for i in np.unique(df['iter']):
                r = df[df['iter'] == i]['test_performance'].to_numpy().mean() / 100.
                rgr.append(BEST_PERFS[exp] - r)
            regrets.append(np.array(rgr))
        regrets = np.column_stack(regrets)
        regrets = np.mean(regrets, axis=1)
    
        search_space_regrets[space] = regrets
    
    colors = {
        's1': '#3492eb',
        's2': '#34ebe5',
        's3': '#8a4000',
        'multi': '#3492eb'
    }
    fig = plt.figure()
    for space, rgr in search_space_regrets.items():
        plt.plot(np.arange(len(rgr)), rgr, label=space, linewidth=2, c=colors[space])

    plt.xlabel('Iteration')
    plt.ylabel('Test error')

    plt.legend()
    plt.grid()
    plt.savefig(file)

def plot_cost_vs_optim_cost(file):
    base_path = '../../data/logs_v2/'
    exps = ['nas101', 'nas201', 'jahs_cifar10', 'jahs_co', 'jahs_fm']
    eval_perc_runtimes_mean, optim_perc_runtimes_mean = [], []
    eval_runtimes_stds, optim_runtimes_std = [], []
    eval_runtime_mean, optim_runtime_mean = [], []
    labels = []
    for e in exps:
        path = f'{base_path}{e}_pc_runtime/'
        dfs = read_csvs(path)
        eval_rts, optim_rts = [], []
        for df in dfs:
            eval_rts.append(df['cost'].sum())
            optim_rts.append(df['optim_cost'][0])
        
        mean_eval_rt = np.mean(eval_rts)
        mean_optim_rt = np.mean(optim_rts)
        eval_runtime_mean.append(mean_eval_rt)
        optim_runtime_mean.append(mean_optim_rt)
        eval_perc = mean_eval_rt / (mean_eval_rt + mean_optim_rt)
        optim_perc = mean_optim_rt / (mean_eval_rt + mean_optim_rt)
        eval_perc_runtimes_mean.append(eval_perc)
        optim_perc_runtimes_mean.append(optim_perc)
        eval_runtimes_stds.append(np.std(eval_rts))
        optim_runtimes_std.append(np.std(optim_rts))
        labels.append(BENCHMARK_LABELS[e])

    plt.rcParams.update({'font.size': 30, 'font.family':'serif'})
    fig = plt.figure(figsize=(13, 10))
    
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(labels))
    r2 = [x + bar_width for x in r1]

    # Create the bar plot
    plt.bar(r1, eval_perc_runtimes_mean, color='#51829B', width=bar_width, edgecolor='grey', label='Evaluation time (%)')
    plt.bar(r2, optim_perc_runtimes_mean, color='#C1E1C1', width=bar_width, edgecolor='grey', label='Optimization time (%)')
    print(f"Eval Runtime: {eval_runtime_mean} +/- {eval_runtimes_stds}")
    print(f"Optim Runtime: {optim_runtime_mean} +/- {optim_runtimes_std}")

    # Add labels and title
    plt.xlabel('Benchmark')
    plt.ylabel('Training vs. Search time (in %)')
    #plt.title('Bar Plot with Two Bars per Label')
    plt.xticks([r + bar_width / 2 for r in range(len(labels))], ['NAS-101 \n (C-10)', 'NAS-201 \n (C-10)', 'JAHS \n (C-10)', 'JAHS \n (FM)', 'JAHS \n(CH)'])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '20', '40', '60', '80', '100'])

    for i in range(len(eval_perc_runtimes_mean)):
        num1 = eval_perc_runtimes_mean[i] 
        num2 = optim_perc_runtimes_mean[i]
        plt.text(r1[i] - .15, num1 + .01, s=f'{num1 * 100:.2f}')
        plt.text(r2[i] - .15, num2 + .01, s=f'{num2 * 100:.2f}')

    # Add a legend
    plt.legend(loc='center right')

    plt.savefig(file, bbox_inches='tight')

def plot_runtime_comparison(file, algos):
    base_path = '../../data/logs_v2/'
    exps = ['nas201', 'nas101', 'jahs_cifar10', 'jahs_co', 'jahs_fm']
    pc_runtimes_mean, smac_runtimes_mean = [], []
    pc_rt_stds, smac_rt_stds = [], []
    labels = []
    for e in exps:
        pc_path = f'{base_path}{e}_pc_runtime/'
        smac_path = f'{base_path}{e}_smac_runtime/'
        pc_dfs = read_csvs(pc_path, 20)
        smac_dfs = read_csvs(smac_path, 20)
        pc_cost, smac_cost = [], []
        for pc_df, smac_df in zip(pc_dfs, smac_dfs):
            pc_cost.append(pc_df['optim_cost'][0])
            smac_cost.append(smac_df['optim_cost'][0])

        
        min_factor = min(min(pc_cost), min(smac_cost))
        max_factor = max(max(pc_cost), max(smac_cost))
        pc_cost, smac_cost = np.array(pc_cost), np.array(smac_cost)
        pc_cost_norm = (pc_cost - min_factor) / (max_factor - min_factor)
        smac_cost_norm = (smac_cost - min_factor) / (max_factor - min_factor)
        
        cum_pc_cost = np.mean(pc_cost_norm)
        cum_smac_cost = np.mean(smac_cost_norm)
        pc_runtimes_mean.append(cum_pc_cost)
        smac_runtimes_mean.append(cum_smac_cost)
        pc_rt_stds.append(np.std(pc_cost_norm))
        smac_rt_stds.append(np.std(smac_cost_norm))
        labels.append(BENCHMARK_LABELS[e])

    plt.rcParams.update({'font.size': 30, 'font.family':'serif'})
    fig = plt.figure(figsize=(12, 10))
    
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(labels))
    r2 = [x + bar_width for x in r1]

    # Create the bar plot
    plt.bar(r1, pc_runtimes_mean, yerr=pc_rt_stds, color='#51829B', width=bar_width, edgecolor='grey', label='IBO-HPC runtime (%)')
    plt.bar(r2, smac_runtimes_mean, yerr=smac_rt_stds, color='#C1E1C1', width=bar_width, edgecolor='grey', label='SMAC runtime (%)')
    print(f"IBO Runtime: {pc_runtimes_mean} +/- {pc_rt_stds}")
    print(f"SMAC Runtime: {smac_runtimes_mean} +/- {smac_rt_stds}")

    # Add labels and title
    plt.xlabel('Benchmark (ordered by #hyperparameters)')
    plt.ylabel('Relative Runtime IBO-HPC vs. SMAC')
    #plt.title('Bar Plot with Two Bars per Label')
    plt.xticks([r + bar_width / 2 for r in range(len(labels))], ['NAS-201 \n (C-10)', 'NAS-101 \n (C-10)', 'JAHS \n (C-10)', 'JAHS \n (FM)', 'JAHS \n(CH)'])

    # Add a legend
    plt.legend()

    plt.savefig(file, bbox_inches='tight')

def plot_interaction_gain(file, version):
    plt.rcParams.update({'font.size': 30, 'font.family':'serif'})
    search_spaces = ['s1', 's2', 's3']

    benchmarks = ['jahs_cifar10', 'jahs_co', 'jahs_fm']

    search_space_sizes = {
        'v2': {
            's1': (1 - 1e-3) * (1e-2 - 1e-5) * 3 * 2 * 5**6 * 16 * 32 * 200 * 1,
            's2': (1 - 1e-3) * (1e-2 - 1e-5) * 3 * 2 * 5**6 * 8 * 14 * 200 * 1,
            's3': (1 - 1e-3) * (1e-2 - 1e-5) * 3 * 2 * 5**6 * 3 * 3 * 200 * 1,
        },
        'v3': {
            's1': (1 - 1e-3) * (1e-2 - 1e-5) * 3 * 2 * 5**6 * 16 * 32 * 200 * 1,
            's2': (1 - 1e-3) * (1e-2 - 1e-5) * 3 * 2 * 5**6 * 8 * 14 * 200 * 1,
            's3': (1 - 1e-3) * (1e-2 - 1e-5) * 3 * 2 * 5**6 * 3 * 3 * 200 * 1
        }
    }

    colors = {
        'nas101': '#32a852',
        'nas201': '#c98b2e',
        'jahs_cifar10': '#c9462e',
        'jahs_co': '#2e6fc9',
        'jahs_fm': '#a82ec9'
    }
    diffs = {}
    for b in benchmarks:
        prefix = f'../../data/search-space-size-{version}/{b}_pc_'
        space_diffs = []
        space_stds = []
        for space in search_spaces:
            vanilla_run_pth = prefix + space
            interaction_run_pth = prefix + 'int_' + space
            vanilla_dfs = read_csvs(vanilla_run_pth, 50)
            interaction_dfs = read_csvs(interaction_run_pth, 50)

            v_best = np.array([df['test_performance'].max() / 100. for df in vanilla_dfs])
            i_best = np.array([df['test_performance'].max() / 100. for df in interaction_dfs])
            diff = i_best[:46] - v_best[:46]
            avg_diff = np.average(diff)
            #avg_diff = - np.log(avg_diff / search_space_sizes[version][space])
            std_diff = np.std(diff)
            space_diffs.append(avg_diff)
            space_stds.append(std_diff)
        diffs[b] = (space_diffs, space_stds)

    bar_width = 0.2
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111)
    x = np.arange(3)
    y_max = 0
    for i, b in enumerate(benchmarks):
        space_diffs, space_std = diffs[b]
        y_max = max(y_max, max(space_diffs))
        c = colors[b]
        x_ = x + (i * bar_width)
        ax.bar(x_, space_diffs, color=c, label=BENCHMARK_LABELS[b], width=bar_width, yerr=space_std)

    ax.set_xticks([0.2, 1.2, 2.2], ['Large', 'Medium', 'Small'])

    plt.xlabel('Search Space Size')
    plt.ylabel('Accuracy Gain by Interaction')
    plt.ylim(-5e-3, y_max + 0.03)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(file, bbox_inches='tight')

def plot_speedups(file):
    plt.rcParams.update({'font.size': 30, 'font.family':'serif'})
    #plt.rc('axes', titlesize=20)
    exps = ['nas101', 'nas201', 'jahs_cifar10', 'jahs_fm', 'jahs_co']

    #pc_int_means, pc_int_early_means = [], []
    #pc_int_stds, pc_int_early_stds = [], []
    #for exp in exps:
    #    speedups = compute_speedups(exp)
    #    pc_int_means.append(np.mean(speedups['pc_int']))
    #    pc_int_early_means.append(np.mean(speedups['pc_int_early']))
    #    pc_int_early_stds.append(np.std(speedups['pc_int_early']))
    #    pc_int_stds.append(np.std(speedups['pc_int']))

    pc_int, pc_int_early = [], []
    for exp in exps:
        speedups = compute_speedups(exp)
        pc_int.append(speedups['pc_int'])
        pc_int_early.append(speedups['pc_int_early'])
    
    br1 = np.arange(len(pc_int)) 
    br2 = [x - (0.25+0.1) for x in br1]

    fig, ax = plt.subplots(figsize=(13, 10))

    #plt.bar(br1, pc_int_means, width=0.25, color='r', label='Interaction@5', yerr=pc_int_stds)
    #plt.bar(br2, pc_int_early_means, width=0.25, color='b', label='Interaction@10', yerr=pc_int_early_stds)

    bp1 = ax.boxplot(pc_int, positions=br1, widths=0.25, showfliers=False, boxprops={'linewidth': 4},)
    bp2 = ax.boxplot(pc_int_early, positions=br2, widths=0.25, showfliers=False, boxprops={'linewidth': 4})

    for k, v in bp1.items():
        if k not in ['whiskers', 'fliers', 'caps']:
            plt.setp(bp1.get(k), color='#FA8072')

    for k, v in bp2.items():
        if k not in ['whiskers', 'fliers', 'caps']:
            plt.setp(bp2.get(k), color='#7B3F00')
    #for p in bp1['boxes']:
    #    p.set_markerfacecolor('cyan')
#
    #for p in bp2['boxes']:
    #    p.set_markerfacecolor('tan')
    tick_pos = [(br1[i] + br2[i]) / 2 for i in range(len(br1))]
    plt.xticks(tick_pos, ['NAS-101 \n (C-10)', 'NAS-201 \n (C-10)', 'JAHS \n (C-10)', 'JAHS \n (FM)', 'JAHS \n(CH)'], size=25)

    plt.ylabel('Speedup')
    plt.xlabel('Benchmark')
    plt.ylim(-2, 20)
    plt.grid(axis='y', alpha=0.5)
    plt.ylim(0.0, 18.)

    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Interaction@10', 'Interaction@5'], loc='upper left')

    plt.savefig(file, bbox_inches='tight')

def plot_slowdown(file):
    plt.rcParams.update({'font.size': 30, 'font.family':'serif'})
    #plt.rc('axes', titlesize=20)
    exps = ['nas101', 'nas201', 'jahs_cifar10', 'jahs_fm', 'jahs_co']

    results_slowdown = []
    stds_slowdown = []
    results_speedup = []
    stds_speedup = []
    for exp in exps:
        slowdown_mean, slowdown_std = compute_slowdown(exp, 'pc_int_recover')
        #speedup_mean, speedup_std = compute_slowdown(exp, 'pc_int_early')
        #results_speedup.append(speedup_mean)
        #stds_speedup.append(speedup_std)
        results_slowdown.append(slowdown_mean)
        stds_slowdown.append(slowdown_std)

    br1 = np.arange(len(results_slowdown))
    br2 = np.array([x + 0.25 for x in br1])

    fig, ax = plt.subplots(figsize=(12, 10))


    #plt.bar(br1, pc_int_means, width=0.25, color='r', label='Interaction@5', yerr=pc_int_stds)
    #plt.bar(br2, pc_int_early_means, width=0.25, color='b', label='Interaction@10', yerr=pc_int_early_stds)

    ax.bar(br1, results_slowdown, yerr=stds_slowdown, width=0.25, color='#51829B', label='IBO-HPC w/ harm. feedback')
    #ax.bar(br2, results_speedup, yerr=stds_speedup, width=0.25, color='#F6995C', label='IBO-HPC w/ ben. feedback')
    #ax.hlines(1.0, xmax=len(results_slowdown) - .5, xmin=-.5, color='#8D493A', label='IBO-HPC w/o feedback')

    #for k, v in bp1.items():
    #    if k not in ['whiskers', 'fliers', 'caps']:
    #        plt.setp(bp1.get(k), color='#6b7ff2')

    tick_pos = [br1[i] for i in range(len(br1))]
    plt.xticks(tick_pos, ['NAS-101 \n (C-10)', 'NAS-201 \n (C-10)', 'JAHS \n (C-10)', 'JAHS \n (FM)', 'JAHS \n(CH)'], size=25)
    lbls = ax.get_yticklabels()
    lbls = [float(l.get_text()) + 1 for l in lbls]
    lbls[0] = ''
    ax.set_yticklabels(lbls)

    plt.ylabel('Relative RT w.r.t. std. IBO-HPC')
    plt.xlabel('Benchmark')
    #plt.ylim(2, -10)
    plt.grid(axis='y', alpha=0.5)
    #plt.ylim(0.0, 18.)

    #ax.legend([bp1["boxes"][0]], ['Interaction@5'], loc='lower right')
    plt.legend()
    plt.savefig(file, bbox_inches='tight')

def plot_no_catchup_diffs_on_neg_feedback(file):
    plt.rcParams.update({'font.size': 30, 'font.family':'serif'})
    #plt.rc('axes', titlesize=20)
    exps = ['nas101', 'nas201', 'jahs_cifar10', 'jahs_fm', 'jahs_co']
    exps = ['jahs_cifar10', 'jahs_co']

    non_recover_mean_perfs, mean_perfs = [], []
    non_recover_perf_stds, perfs_stds = [], []

    for exp in exps:
        non_recovered_perfs, perfs = compute_no_catch_up_diff(exp)
        non_recover_mean_perfs.append(np.mean(non_recovered_perfs))
        mean_perfs.append(np.mean(perfs))
        non_recover_perf_stds.append(np.std(non_recovered_perfs))
        perfs_stds.append(np.std(perfs))

    br1 = np.arange(len(non_recover_mean_perfs))
    br2 = np.array([x + 0.25 for x in br1])

    fig, ax = plt.subplots(figsize=(12, 10))

    #plt.bar(br1, pc_int_means, width=0.25, color='r', label='Interaction@5', yerr=pc_int_stds)
    #plt.bar(br2, pc_int_early_means, width=0.25, color='b', label='Interaction@10', yerr=pc_int_early_stds)

    ax.bar(br1, non_recover_mean_perfs, yerr=non_recover_perf_stds, width=0.25, color='#51829B', label='neg. feedback, no recover')
    ax.bar(br2, mean_perfs, yerr=perfs_stds, width=0.25, color='#F6995C', label='no feedback')
    #ax.hlines(0.0, xmax=len(results), xmin=0, color='#8D493A')

    #for k, v in bp1.items():
    #    if k not in ['whiskers', 'fliers', 'caps']:
    #        plt.setp(bp1.get(k), color='#6b7ff2')

    tick_pos = [(br1[i] + br2[i]) / 2 for i in range(len(br1))]
    #plt.xticks(tick_pos, ['NAS-101 \n (C-10)', 'NAS-201 \n (C-10)', 'JAHS \n (C-10)', 'JAHS \n (FM)', 'JAHS \n(CH)'], size=25)
    plt.xticks(tick_pos, ['JAHS \n (C-10)', 'JAHS \n(CH)'], size=25)

    plt.ylabel('Test Accuracy (%)')
    plt.xlabel('Benchmark')
    #plt.ylim(2, -10)
    plt.grid(axis='y', alpha=0.5)
    #plt.ylim(0.0, 18.)

    ax.legend(loc='lower right')

    plt.savefig(file, bbox_inches='tight')

def plot_cdf(exp, file, max_reads=50):
    plt.rcParams.update({'font.size': 40, 'font.family':'serif'})
    prefix = f'../../data/logs_v2/{exp}_'
    #algos = [ 'pc', 'pc_int', 'pc_int_early', 'rs', 'pibo', 'hyperband', 'ls', 'smac']
    #algos = [ 'pc', 'pc_int_autorecover', 'pc_int_multi_new', 'pc_int', 'pc_int_early', 'pc_int_dist', 'rs', 'pibo', 'bopro', 'ls', 'smac']
    algos = [ 'pc', 'pc_int_recover', 'pc_int_many', 'pc_int_late', 'pc_int_early', 'pc_int_dist', 'skoptbo', 'pibo', 'bopro', 'ls', 'smac']
    fig, ax = plt.subplots(figsize=(12, 9.2))
    for a in algos:
        dfs = read_csvs(prefix + a, max_reads=max_reads)
        df = pd.concat(dfs)
        #for df in dfs:
        test_perf = df['test_performance'].to_numpy()
        print(f"MAX: {test_perf.max()}")
        regret = BEST_PERFS[exp] - test_perf
        c, line_type = ALGO_COLORS[a]
        label = ALGO_NAME_MAPPING[a]
            #ax.hist(test_perf, cumulative=True, density=True, color=c)
        #sns.kdeplot(data=test_perf, cumulative=True, ax=ax, color=c, label=label)
        sns.ecdfplot(data=test_perf, ax=ax, color=c, label=label, linestyle=line_type, linewidth=3.5)

    #plt.legend(fontsize=18)
    ax.tick_params(length=12., width=3., direction='inout')
    plt.ylim(-0.05, 1.05)
    plt.xlim(10, 100)
    plt.ylabel('CDF')
    plt.xlabel('Test Accuracy (%)')
    plt.xticks(np.arange(0, 100, 10), np.arange(0, 100, 10))
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(file, bbox_inches='tight')

# =====================================================================
# Intervention-classification ROC plots
#
# Shared helpers for the good-vs-bad intervention-classification
# experiments. Used by analyze_intervention_classification.py for the
# full set of figures, and by plot_all.py's --intervention-classification
# mode for just the ROC PDFs.
# =====================================================================

INTERVENTION_METHOD_COLOR = {"pval": "#931147", "score": "#16501B"}
INTERVENTION_METHOD_LINESTYLE = {"pval": "solid", "score": (0, (5, 2))}
INTERVENTION_METHOD_DISPLAY = {"pval": "DisCoMBO", "score": "DynaBO"}
# Direction of the natural discriminator: "good" probability rises with
# `direction * metric`. MC pvals: higher pval -> good (+1). DynaBO LCB-diff
# scores: lower score -> good (-1).
INTERVENTION_NATURAL_DIRECTION = {"pval": +1, "score": -1}
INTERVENTION_BENCHMARK_DISPLAY = {
    ("jahs", "cifar10"):              "JAHS (CIFAR-10)",
    ("jahs", "fashion_mnist"):        "JAHS (FM)",
    ("jahs", "colorectal_histology"): "JAHS (CO)",
    ("floris", ""):                   "FLORIS",
    ("pv", "crossed_barrel"):         "PV (crossed-barrel)",
}


def _intervention_key_label(benchmark, task):
    return benchmark if not task else f"{benchmark}_{task}"


def _intervention_apply_rcparams():
    """rcParams for the intervention-classification ROC plots."""
    plt.rcParams.update({
        "font.size": 20,
        "font.family": "serif",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })


def _intervention_load_method(directory, suffix, metric_col, method):
    rows = []
    paths = sorted(p for p in listdir(directory) if p.endswith(suffix))
    for fname in paths:
        df = pd.read_csv(join(directory, fname))
        if "benchmark" not in df.columns:
            df["benchmark"] = "jahs"
        if "task" not in df.columns:
            df["task"] = ""
        df["task"] = df["task"].fillna("").astype(str)
        df["method"] = method
        df["metric"] = df[metric_col]
        rows.append(df[[
            "method", "benchmark", "task", "seed_idx", "seed",
            "intervention_idx", "label", "kind", "trigger_iter",
            "metric", "final_best_val_perf",
        ]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_intervention_classification_data(pval_dir, score_dir):
    """Load the long-form (method, benchmark, task, ..., metric) dataframe
    from the canonical pval and score directories.
    """
    pvals = _intervention_load_method(str(pval_dir), "__pvals.csv", "pval_mean", "pval")
    scores = _intervention_load_method(str(score_dir), "__scores.csv", "score", "score")
    return pd.concat([pvals, scores], ignore_index=True)


def compute_intervention_classification_summary(all_df):
    """Per (method, benchmark, task): AUC (natural / best), KS, label balance."""
    from scipy.stats import ks_2samp
    from sklearn.metrics import roc_auc_score

    rows = []
    for keys, sub in all_df.groupby(["method", "benchmark", "task"], dropna=False):
        method, benchmark, task = keys
        sub = sub.dropna(subset=["metric"])
        if sub.empty:
            continue
        labels = (sub["label"] == "good").astype(int).values
        signed = INTERVENTION_NATURAL_DIRECTION[method] * sub["metric"].values

        if len(np.unique(labels)) < 2 or len(signed) == 0:
            auc_natural = float("nan")
        else:
            auc_natural = float(roc_auc_score(labels, signed))
        auc_best = (max(auc_natural, 1.0 - auc_natural)
                    if not np.isnan(auc_natural) else float("nan"))

        good = sub.loc[sub["label"] == "good", "metric"].values
        bad = sub.loc[sub["label"] == "bad", "metric"].values
        ks_stat = (ks_2samp(good, bad).statistic
                   if len(good) > 0 and len(bad) > 0 else float("nan"))

        rows.append({
            "method": method, "benchmark": benchmark, "task": task,
            "n_total": int(len(sub)),
            "n_good": int(len(good)), "n_bad": int(len(bad)),
            "n_intervention_idx": int(sub["intervention_idx"].nunique()),
            "n_seeds": int(sub["seed_idx"].nunique()),
            "good_mean_metric": float(np.nanmean(good)) if len(good) else float("nan"),
            "bad_mean_metric": float(np.nanmean(bad)) if len(bad) else float("nan"),
            "ks_statistic": ks_stat,
            "auc_natural": auc_natural, "auc_best": auc_best,
        })
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["separation_gap"] = (
        summary["good_mean_metric"] - summary["bad_mean_metric"]
    )
    return summary.sort_values(["benchmark", "task", "method"]).reset_index(drop=True)


def plot_intervention_classification_combined_roc(all_df, summary, out_dir, exts=("pdf",)):
    """Per (benchmark, task): one ROC plot with DisCoMBO + DynaBO overlaid."""
    from sklearn.metrics import roc_curve

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _intervention_apply_rcparams()

    written = []
    for keys, sub in all_df.groupby(["benchmark", "task"], dropna=False):
        benchmark, task = keys
        sub = sub.dropna(subset=["metric"])
        if sub.empty or sub["label"].nunique() < 2:
            continue
        title = INTERVENTION_BENCHMARK_DISPLAY.get(
            (benchmark, task or ""), _intervention_key_label(benchmark, task))

        fig, ax = plt.subplots(figsize=(5.25, 4.9))
        ax.plot([0, 1], [0, 1], color="#888888", linestyle=":", lw=1.2, zorder=0)

        any_curve = False
        for method in ("pval", "score"):
            method_df = sub[sub["method"] == method]
            if method_df.empty or method_df["label"].nunique() < 2:
                continue
            labels = (method_df["label"] == "good").astype(int).values
            signed = INTERVENTION_NATURAL_DIRECTION[method] * method_df["metric"].values
            try:
                fpr, tpr, _ = roc_curve(labels, signed)
            except ValueError:
                continue
            ax.plot(fpr, tpr,
                    color=INTERVENTION_METHOD_COLOR[method], lw=3.0,
                    linestyle=INTERVENTION_METHOD_LINESTYLE[method], zorder=2)
            any_curve = True

        if not any_curve:
            plt.close(fig)
            continue

        ax.set_xlabel("false positive rate", labelpad=4)
        ax.set_ylabel("true positive rate")
        ax.set_title(title, pad=6, fontsize=19)
        ax.set_xlim(-0.025, 1.025)
        ax.set_ylim(-0.025, 1.025)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="-", alpha=0.6, linewidth=1.1)
        for ext in exts:
            target = out_dir / f"roc_combined__{_intervention_key_label(benchmark, task)}.{ext}"
            fig.savefig(target, dpi=160, bbox_inches="tight")
            written.append(target)
        plt.close(fig)
    return written


def plot_intervention_classification_roc_pair(all_df, summary, out_dir, exts=("pdf",),
                                              pair=(("jahs", "colorectal_histology"),
                                                    ("pv", "crossed_barrel"))):
    """Two-panel ROC for the (jahs CO, pv crossed_barrel) pair, shared legend below."""
    from sklearn.metrics import roc_curve

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _intervention_apply_rcparams()

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.7))
    handles_for_legend = {}

    for i, (ax, (benchmark, task)) in enumerate(zip(axes, pair)):
        sub = all_df[(all_df["benchmark"] == benchmark)
                     & (all_df["task"].fillna("") == task)]
        title = INTERVENTION_BENCHMARK_DISPLAY.get(
            (benchmark, task), _intervention_key_label(benchmark, task))

        ax.plot([0, 1], [0, 1], color="#888888", linestyle=":", lw=1.5, zorder=0)
        for method in ("pval", "score"):
            method_df = sub[sub["method"] == method].dropna(subset=["metric"])
            if method_df.empty or method_df["label"].nunique() < 2:
                continue
            labels = (method_df["label"] == "good").astype(int).values
            signed = INTERVENTION_NATURAL_DIRECTION[method] * method_df["metric"].values
            try:
                fpr, tpr, _ = roc_curve(labels, signed)
            except ValueError:
                continue
            line, = ax.plot(fpr, tpr,
                            color=INTERVENTION_METHOD_COLOR[method], lw=3.0,
                            linestyle=INTERVENTION_METHOD_LINESTYLE[method],
                            label=INTERVENTION_METHOD_DISPLAY[method], zorder=2)
            handles_for_legend.setdefault(method, line)

        ax.set_xlabel("false positive rate", labelpad=4)
        if i == 0:
            ax.set_ylabel("true positive rate")
        ax.set_title(title, pad=6, fontsize=19)
        ax.set_xlim(-0.025, 1.025)
        ax.set_ylim(-0.025, 1.025)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="-", alpha=0.6, linewidth=1.1)

    ordered = [handles_for_legend[m] for m in ("pval", "score")
               if m in handles_for_legend]
    ordered_labels = [INTERVENTION_METHOD_DISPLAY[m] for m in ("pval", "score")
                      if m in handles_for_legend]
    fig.legend(ordered, ordered_labels, loc="lower center", ncol=len(ordered),
               frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.8, columnspacing=1.4)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.27, wspace=0.10)

    written = []
    for ext in exts:
        target = out_dir / f"roc_pair__jahs_co__pv_crossed_barrel.{ext}"
        fig.savefig(target, dpi=160, bbox_inches="tight")
        written.append(target)
    plt.close(fig)
    return written


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', type=str, help='Output file plot is stored to.')
    parser.add_argument('--exp', type=str, help='Experiment to plot. This name applies to the log-directory names, not the experiment names from main.py.')
    parser.add_argument('--plot', default='regret_vs_cost', help='Plot to produce. regret_vs_cost will plot the average incumbent regret against the search cost, \
                        regret_per_it will plot the average test error per iteration (not incumben!), \
                        interaction_gain will plot the gain an interaction achieves compared to no interaction, \
                        speedups generates a boxplot of the speedups across runs and \
                        cdf plots the cumulative distribution function of the accuracy achieved over time.')
    parser.add_argument('--with-std', action='store_true', help='Whether to plot std or not')
    parser.add_argument('--plot-all', action='store_true', help='Plot all in one go')
    parser.add_argument('--version', default='v1') # only relevant for interaction_gain plots
    parser.add_argument('--uncertainty', default='SE', help='Type of uncertainty plotted. SE=Standard Error, STD=Standard Deviation')
    parser.add_argument('--num-exps', default=500, type=int, help='Number of experiments to use. Will be randomly sampled from the log-directory given by the --exp option.')
    parser.add_argument('--cd-benchmarks', type=str, default='nas101,nas201,jahs_cifar10,jahs_co,jahs_fm', help='Comma-separated benchmark IDs for critical difference plot.')
    parser.add_argument('--cd-algos', type=str, default='pc,pc_int_late,pc_int_early,pc_int_dist,mc_stratified_annealed_sine_c20,pibo,smac,optunabo,bopro,skoptbo,priorband,ls', help='Comma-separated algorithm IDs for critical difference plot.')
    parser.add_argument('--cd-alpha', type=float, default=0.05, help='Alpha for Wilcoxon significance in critical difference plot.')
    parser.add_argument('--cd-base-path', type=str, default='../../data/new-exp', help='Base path that contains benchmark_algo log directories.')
    parser.add_argument('--cd-algo-fontsize', type=float, default=19, help='Font size of algorithm names in critical difference plot.')

    args = parser.parse_args()

    if args.plot == 'regret_vs_cost':
        if args.plot_all:
            exps = ['jahs_cifar10', 'jahs_co', 'jahs_fm', 'nas101', 'nas201']
            base_path = f'./plots/interactive-hpo/tpm/'
            out_files = [base_path + f'{e}.pdf' for e in exps]
            for exp, of in zip(exps, out_files):
                res = collect_results(exp, args.num_exps, args.uncertainty)
                plot_regrets(res, exp, of, args.with_std)
        else:
            res = collect_results(args.exp, args.num_exps, args.uncertainty)
            plot_regrets(res, args.exp, args.out_file, args.with_std)
    elif args.plot == 'regret_per_it':
        plot_avg_regrets_per_iter(args.exp, args.out_file)
    elif args.plot == 'interaction_gain':
        plot_interaction_gain(args.out_file, args.version)
    elif args.plot == 'speedups':
        plot_speedups(args.out_file)
    elif args.plot == 'cdf':
        plot_cdf(args.exp, args.out_file, args.num_exps)
    elif args.plot == 'slowdown':
        plot_slowdown(args.out_file)
    elif args.plot == 'runtime_split':
        plot_cost_vs_optim_cost(args.out_file)
    elif args.plot == 'runtime_pc_vs_smac':
        plot_runtime_comparison(args.out_file, None)
    elif args.plot == 'non-recover-diff':
        plot_no_catchup_diffs_on_neg_feedback(args.out_file)
    elif args.plot == 'critical_difference':
        cd_algos = _resolve_algo_names(args.cd_algos)
        cd_benchmarks = _resolve_benchmarks(args.cd_benchmarks)
        plot_critical_difference(
            file=args.out_file,
            benchmarks=cd_benchmarks,
            algos=cd_algos,
            alpha=args.cd_alpha,
            base_path=args.cd_base_path,
            max_reads=args.num_exps,
            algo_fontsize=args.cd_algo_fontsize,
        )
    else:
        raise ValueError(f'No such plot type: {args.plot}')