"""
Learning Curve Plot Generator
==============================
CONFIG accepts either:
  - A folder path  → auto-discovers all .csv files, uses filename as method name
  - A dict         → {method_name: 'path/to/file.csv' or None to skip}

Run: python generate_plots.py
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — set folder path OR dict of {method: csv_path}
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = './plots'

# Option 1: point to a folder — all CSVs in it will be loaded automatically
# Option 2: point to specific files with method names
# You can mix: use folder for some tasks, specific files for others
DATA_ROOT=os.path.expanduser('~/Downloads/reflexion-res')

HOTPOT = {
    'ReAct':       f'{DATA_ROOT}/hotpot/react/',
    'CoT+GT':      f'{DATA_ROOT}/hotpot/cot/',
    'Reflexion':   f'{DATA_ROOT}/hotpot/reflexion/',
    'RAR (Ours)':  f'{DATA_ROOT}/hotpot/retrieval/',
}

ALFWORLD = {
    'ReAct':       f'{DATA_ROOT}/alf/react/',
    'Reflexion':   f'{DATA_ROOT}/alf/reflexion/',
    'RAR (Ours)':  f'{DATA_ROOT}/alf/retrieval/',
}

HUMANEVAL = {
    'Simple':      f'{DATA_ROOT}/prog/simple/',
    'CoT+GT':      f'{DATA_ROOT}/prog/cot_gt/',
    'Reflexion':   f'{DATA_ROOT}/prog/reflexion/',
    'RAR (Ours)':  f'{DATA_ROOT}/prog/retrieval/',
}

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'ReAct':       '#2196F3',
    'Simple':      '#4CAF50',
    'CoT+GT':      '#4CAF50',
    'Reflexion':   '#FF9800',
    'RAR (Ours)':  '#E91E63',
}
MARKERS = {
    'ReAct':       'o',
    'Simple':      's',
    'CoT+GT':      's',
    'Reflexion':   '^',
    'RAR (Ours)':  'D',
}
LINESTYLES = {
    'ReAct':       '-',
    'Simple':      '--',
    'CoT+GT':      '--',
    'Reflexion':   '-.',
    'RAR (Ours)':  '-',
}
DEFAULT_COLOR     = '#607D8B'
DEFAULT_MARKER    = 'o'
DEFAULT_LINESTYLE = '-'

plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       11,
    'axes.titlesize':  12,
    'axes.labelsize':  11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi':      150,
    'axes.grid':       True,
    'grid.linestyle':  '--',
    'grid.alpha':      0.4,
})

# ─────────────────────────────────────────────────────────────────────────────
# SMART DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> dict:
    """Load a single CSV file into a data dict."""
    df = pd.read_csv(path)
    x_col = df.columns[0]
    return {
        'x':       df[x_col].tolist(),
        'success': df['SuccessRate'].tolist(),
        'fail':    df['FailRate'].tolist(),
        'halted':  df['HaltedRate'].tolist(),
        'steps':   df['AvgSteps'].tolist(),
    }


def _method_name_from_path(path: str) -> str:
    """
    Derive a display name from filename.
    e.g. hotpot_react.csv     -> ReAct
         alf_retrieval.csv    -> RAR (Ours)
         he_cot_gt.csv        -> CoT+GT
    """
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    # strip task prefix (hotpot_, alf_, he_)
    for prefix in ('hotpot_', 'alf_', 'he_', 'humaneval_', 'alfworld_'):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    mapping = {
        'react':      'ReAct',
        'reflexion':  'Reflexion',
        'retrieval':  'RAR (Ours)',
        'cot_gt':     'CoT+GT',
        'cot':        'CoT+GT',
        'simple':     'Simple',
    }
    return mapping.get(stem, stem.replace('_', ' ').title())


def load_data(config) -> dict:
    """
    config can be:
      - str  : folder path → load all .csv files in that folder
      - dict : {method_name: csv_path or None}
    Returns {method_name: data_dict}
    """
    data = {}

    if isinstance(config, str):
        # Folder mode — auto-discover all CSVs
        if not os.path.isdir(config):
            print(f"WARNING: folder '{config}' not found.")
            return data
        csv_files = sorted(glob.glob(os.path.join(config, '*.csv')))
        if not csv_files:
            print(f"WARNING: no CSV files found in '{config}'.")
            return data
        for path in csv_files:
            method = _method_name_from_path(path)
            try:
                data[method] = _load_csv(path)
                print(f"  Loaded '{method}' from {path}")
            except Exception as e:
                print(f"  ERROR loading {path}: {e}")

    elif isinstance(config, dict):
        for method, path in config.items():
            if path is None:
                continue
            # If path is a folder, find the single CSV in it
            if os.path.isdir(path):
                csv_files = sorted(glob.glob(os.path.join(path, '*.csv')))
                if not csv_files:
                    print(f"WARNING: no CSV in folder '{path}', skipping {method}")
                    continue
                if len(csv_files) > 1:
                    print(f"WARNING: multiple CSVs in '{path}', using first: {csv_files[0]}")
                path = csv_files[0]
            if not os.path.exists(path):
                print(f"WARNING: '{path}' not found, skipping {method}")
                continue
            try:
                data[method] = _load_csv(path)
                print(f"  Loaded '{method}' from {path}")
            except Exception as e:
                print(f"  ERROR loading {path}: {e}")
    else:
        print(f"WARNING: unknown config type {type(config)}")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _plot_line(ax, data, metric):
    for method, d in data.items():
        ax.plot(d['x'], d[metric],
                color=COLORS.get(method, DEFAULT_COLOR),
                marker=MARKERS.get(method, DEFAULT_MARKER),
                linestyle=LINESTYLES.get(method, DEFAULT_LINESTYLE),
                linewidth=2, markersize=5, label=method)


def _style_ax(ax, xlabel, ylabel, title, ylim=None, xticks=None, pct=True):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', pad=8)
    if ylim:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.legend(loc='best', framealpha=0.9)
    if pct:
        ax.yaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=1, decimals=0))


def make_task_figure(data, task_name, xlabel, output_path,
                     ylim_success=(0, 1.05), ylim_steps=None):
    if not data:
        print(f"No data for {task_name}, skipping.")
        return
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle(f'{task_name} — Learning Curves',
                 fontsize=13, fontweight='bold', y=1.02)
    xticks = list(data.values())[0]['x']

    _plot_line(axes[0], data, 'success')
    _style_ax(axes[0], xlabel, 'Success Rate',
              '(a) Success Rate', ylim_success, xticks)

    _plot_line(axes[1], data, 'fail')
    _style_ax(axes[1], xlabel, 'Fail Rate',
              '(b) Fail Rate', (0, 1), xticks)

    _plot_line(axes[2], data, 'halted')
    _style_ax(axes[2], xlabel, 'Halted Rate',
              '(c) Halted Rate', (0, 0.5), xticks)

    _plot_line(axes[3], data, 'steps')
    max_steps = max(v for d in data.values() for v in d['steps'])
    _style_ax(axes[3], xlabel, 'Avg Steps',
              '(d) Avg Steps per Trial',
              ylim_steps or (0, max_steps * 1.2), xticks, pct=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def make_combined_success_figure(all_data, output_path):
    tasks = [
        ('HotPotQA',  'hotpot',    'Trial Number',     (0.0, 0.75)),
        ('ALFWorld',  'alfworld',   'Trial Number',     (0.0, 1.05)),
        ('HumanEval', 'humaneval',  'Iteration Number', (0.65, 1.05)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Success Rate per Trial/Iteration — All Tasks',
                 fontsize=13, fontweight='bold', y=1.02)
    for ax, (name, key, xlabel, ylim) in zip(axes, tasks):
        data = all_data.get(key, {})
        if not data:
            ax.set_title(f'{name} (no data)'); continue
        xticks = list(data.values())[0]['x']
        for method, d in data.items():
            ax.plot(d['x'], d['success'],
                    color=COLORS.get(method, DEFAULT_COLOR),
                    marker=MARKERS.get(method, DEFAULT_MARKER),
                    linestyle=LINESTYLES.get(method, DEFAULT_LINESTYLE),
                    linewidth=2, markersize=5, label=method)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Success Rate')
        ax.set_ylim(ylim)
        ax.set_xticks(xticks)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.yaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=1, decimals=0))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def make_fail_halted_figure(all_data, output_path):
    tasks = [
        ('HotPotQA',  'hotpot',    'Trial Number'),
        ('ALFWorld',  'alfworld',   'Trial Number'),
        ('HumanEval', 'humaneval',  'Iteration Number'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Fail Rate and Halted Rate — All Tasks',
                 fontsize=13, fontweight='bold', y=1.02)
    for col, (name, key, xlabel) in enumerate(tasks):
        data = all_data.get(key, {})
        if not data:
            continue
        xticks = list(data.values())[0]['x']
        for method, d in data.items():
            kw = dict(color=COLORS.get(method, DEFAULT_COLOR),
                      marker=MARKERS.get(method, DEFAULT_MARKER),
                      linestyle=LINESTYLES.get(method, DEFAULT_LINESTYLE),
                      linewidth=2, markersize=5, label=method)
            axes[0][col].plot(d['x'], d['fail'],   **kw)
            axes[1][col].plot(d['x'], d['halted'], **kw)
        for row, (ylabel, ylim) in enumerate([('Fail Rate', (0,1)),
                                               ('Halted Rate', (0, 0.5))]):
            axes[row][col].set_title(name if row == 0 else '', fontweight='bold')
            axes[row][col].set_ylabel(ylabel)
            axes[row][col].set_ylim(ylim)
            axes[row][col].set_xticks(xticks)
            axes[row][col].legend(loc='best', framealpha=0.9)
            axes[row][col].yaxis.set_major_formatter(
                mticker.PercentFormatter(xmax=1, decimals=0))
            if row == 1:
                axes[row][col].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def make_avg_steps_figure(all_data, output_path):
    tasks = [
        ('HotPotQA',  'hotpot',    'Trial Number'),
        ('ALFWorld',  'alfworld',   'Trial Number'),
        ('HumanEval', 'humaneval',  'Iteration Number'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Average Steps per Trial/Iteration — All Tasks',
                 fontsize=13, fontweight='bold', y=1.02)
    for ax, (name, key, xlabel) in zip(axes, tasks):
        data = all_data.get(key, {})
        if not data:
            ax.set_title(f'{name} (no data)'); continue
        xticks = list(data.values())[0]['x']
        for method, d in data.items():
            ax.plot(d['x'], d['steps'],
                    color=COLORS.get(method, DEFAULT_COLOR),
                    marker=MARKERS.get(method, DEFAULT_MARKER),
                    linestyle=LINESTYLES.get(method, DEFAULT_LINESTYLE),
                    linewidth=2, markersize=5, label=method)
        max_steps = max(v for d in data.values() for v in d['steps'])
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Avg Steps / Iterations')
        ax.set_ylim(0, max_steps * 1.2)
        ax.set_xticks(xticks)
        ax.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading HotPotQA data...")
    hotpot_data = load_data(HOTPOT)

    print("Loading ALFWorld data...")
    alfworld_data = load_data(ALFWORLD)

    print("Loading HumanEval data...")
    humaneval_data = load_data(HUMANEVAL)

    all_data = {
        'hotpot':    hotpot_data,
        'alfworld':  alfworld_data,
        'humaneval': humaneval_data,
    }

    print("\nGenerating per-task figures...")
    make_task_figure(hotpot_data, 'HotPotQA', 'Trial Number',
                     f'{OUTPUT_DIR}/hotpotqa_learning_curves.png',
                     ylim_success=(0.0, 0.75), ylim_steps=(0, 8))

    make_task_figure(alfworld_data, 'ALFWorld', 'Trial Number',
                     f'{OUTPUT_DIR}/alfworld_learning_curves.png',
                     ylim_success=(0.0, 1.05), ylim_steps=(0, 30))

    make_task_figure(humaneval_data, 'HumanEval', 'Iteration Number',
                     f'{OUTPUT_DIR}/humaneval_learning_curves.png',
                     ylim_success=(0.65, 1.05), ylim_steps=(0, 5))

    print("\nGenerating combined figures...")
    make_combined_success_figure(all_data, f'{OUTPUT_DIR}/combined_success_rate.png')
    make_fail_halted_figure(all_data,      f'{OUTPUT_DIR}/combined_fail_halted.png')
    make_avg_steps_figure(all_data,        f'{OUTPUT_DIR}/combined_avg_steps.png')

    print(f"\nDone. All plots saved to {OUTPUT_DIR}/")