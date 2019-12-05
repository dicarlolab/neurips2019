from matplotlib import ticker
import logging
import sys
from pathlib import Path

import numpy as np
import seaborn
from matplotlib import pyplot
from numpy.random.mtrand import RandomState

from figures import read_common_data

seaborn.set(context='talk')
seaborn.set_style("whitegrid")
pyplot.rcParams['svg.fonttype'] = 'none'  # avoid individual text letters, https://stackoverflow.com/a/35734729/2225200

benchmark_color_mapping = {
    'movshon.FreemanZiemba2013.V1-pls': '#0D9E52',
    'movshon.FreemanZiemba2013.V2-pls': '#1A82C3',
    'V4': '#006A34',
    'IT': '#073066',
    'Behavior': '#5D5F61',
    'OST': '#F52F2F',
}
ceilings = {
    'movshon.FreemanZiemba2013.V1-pls': 0,
    'movshon.FreemanZiemba2013.V2-pls': 0,
    'V4': .9,  # TODO for old benchmark
    'IT': .8,  # TODO
    'Behavior': .485,
    'OST': .79,
}


def ceil(score, benchmark):
    ceiling = ceilings[benchmark]
    if benchmark == 'Behavior':
        return score / np.sqrt(ceiling)
    if benchmark == 'OST':
        return score / ceiling
    return explained_variance(score, ceiling)


def explained_variance(score, ceiling):
    # ro(X, Y)
    # = (r(X, Y) / sqrt(r(X, X) * r(Y, Y)))^2
    # = (r(X, Y) / sqrt(r(Y, Y) * r(Y, Y)))^2  # assuming that r(Y, Y) ~ r(X, X) following Yamins 2014
    # = (r(X, Y) / r(Y, Y))^2
    r_square = np.power(score / ceiling, 2)
    return r_square


def plot(benchmarks, annotated_models=(), hide_all_but=None, highlight=()):
    scores = read_common_data()
    scores = scores[[not model.startswith('Base') and not model == 'IamNN' for model in scores['Model']]]
    benchmarks_scores = {benchmark: scores[benchmark] for benchmark in benchmarks}

    fig, axes = pyplot.subplots(figsize=(5, 7), ncols=len(benchmarks))
    random = RandomState(0)
    for i, (ax, benchmark) in enumerate(zip(axes.flatten(), benchmarks)):
        benchmark_scores = benchmarks_scores[benchmark]
        models = scores['Model']
        x = random.uniform(-.3, +.3, size=len(benchmark_scores))
        y = [score for score in benchmark_scores]
        y = [score if not np.isnan(score) else 0 for score in y]
        y = [ceil(score, benchmark) for score in y]
        for _x, _y, model in zip(x, y, models):
            if model not in annotated_models:
                continue
            if hide_all_but and model not in hide_all_but:
                continue
            maxlen = 10
            if len(model) > maxlen:
                lenhalf = (maxlen - 3) // 2
                model = model[:lenhalf] + '...' + model[-lenhalf:]
            ax.text(_x, _y, model,
                    color='lightgray', alpha=0.7,
                    rotation=40, rotation_mode='anchor', horizontalalignment='left', verticalalignment='baseline')
        color = benchmark_color_mapping[benchmark]
        # set alpha to 0 to hide models
        if color.startswith('#'):
            color = color.lstrip('#')
            color = tuple(int(color[i:i + 2], 16) / 256 for i in (0, 2, 4))
        alphas = [1 if not hide_all_but or model in hide_all_but else 0 for model in models]
        colors = [color + (alpha,) for alpha in alphas]
        ax.scatter(x, y, color=np.array(colors))
        # highlight models
        if highlight:
            highlight_indices = [i for i, model in enumerate(models) if model in highlight]
            highlight_x, highlight_y = np.array(x)[highlight_indices], np.array(y)[highlight_indices]
            highlight_colors = np.array(colors)[highlight_indices]
            ax.scatter(highlight_x, highlight_y, color=highlight_colors, linewidth=2, edgecolor='orange')

        # label
        ax.set_xticks([0])
        ax.set_xticklabels([benchmark])
        if i == 0:
            ax.set_ylabel('Brain-Score Component Score (normalized)')
        ax.yaxis.set_major_locator(ticker.MultipleLocator(.095))
        ax.set_yticklabels([f"{tick:.1f}"[1:] for tick in ax.get_yticks()])
        ax.tick_params(axis='y', which='major', labelsize=10, direction="in", pad=1)
        ax.grid(False)

    pyplot.xticks(rotation=90)
    seaborn.despine()
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    benchmarks = ['V4', 'IT', 'Behavior',
                  # 'OST'
                  ]
    for show_only, highlight, filename in [
        (None, ['AlexNet'], 'jittered'),
        (['AlexNet'], ['AlexNet'], f'jittered-AlexNet_only'),
        (None, ['CORnet-S'], 'jittered-highlight_CORnet-S'),
    ]:
        fig = plot(benchmarks=benchmarks, hide_all_but=show_only,
                   highlight=highlight if show_only is None else [])
        pyplot.savefig(Path(__file__).parent / 'figures' / f"{filename}.png", bbox_inches='tight', dpi=600)
