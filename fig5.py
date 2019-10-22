import os
from collections import OrderedDict

import fire
import numpy as np
import scipy.stats
import seaborn
from matplotlib import pyplot

from brainscore.benchmarks.temporal import DicarloKar2019OST
from brainscore.metrics.ost import OSTCorrelation
from candidate_models import score_model, brain_translated_pool

seaborn.set()
seaborn.set_context('paper', font_scale=2)
seaborn.set_style('whitegrid', {'axes.grid': False})


def recurrence_vs_score():
    s_identifiers = OrderedDict([
        (222, '2,2,2'),
        ('', '2,4,2'),
        (444, '4,4,4'),
        (484, '4,8,4'),
        (10, '10,10,10'),
    ])

    scores = []
    for identifier in s_identifiers:
        model_identifier = f'CORnet-S{identifier}'
        model = brain_translated_pool[model_identifier]
        score = score_model(model_identifier=model_identifier, model=model, benchmark_identifier='dicarlo.Kar2019-ost')
        scores.append(score)

    # plot
    x = list(s_identifiers.values())
    y = [score.sel(aggregation='center') for score in scores]
    yerr = [score.sel(aggregation='error') for score in scores]
    pyplot.figure()
    gray, pink = '#808080', '#D4145A'
    pyplot.bar(x, y, yerr=yerr, width=.5, color=pink)
    pyplot.xticks(rotation=45)
    pyplot.xlabel('recurrent steps')
    pyplot.ylabel('OST score')

    pyplot.tight_layout()
    seaborn.despine(right=True, top=True)
    for extension in ['png', 'pdf', 'svg']:
        pyplot.savefig(os.path.join(os.path.dirname(__file__), '..', 'results', 'osts', f'ost-recurrence.{extension}'))


def prediction_vs_target():
    class MetricHook(OSTCorrelation):
        def __init__(self):
            super(MetricHook, self).__init__()
            self._predicted_osts, self._target_osts = [], []

        def correlate(self, predicted_osts, target_osts):
            self._predicted_osts = np.concatenate((self._predicted_osts, predicted_osts))
            self._target_osts = np.concatenate((self._target_osts, target_osts))
            return super(MetricHook, self).correlate(predicted_osts, target_osts)

    metric_hook = MetricHook()
    benchmark = DicarloKar2019OST()
    benchmark._similarity_metric = metric_hook
    model = brain_translated_pool['CORnet-S']
    score = benchmark(model)
    if hasattr(score, 'ceiling'):
        score = score.raw  # use unceiled score

    correlation = score.sel(aggregation='center')
    t, p = scipy.stats.ttest_ind(score.raw.values, [0] * len(score.raw.values))
    num_bins = 5
    predicted_osts, target_osts = metric_hook._predicted_osts, metric_hook._target_osts
    non_nan = np.logical_and(~np.isnan(predicted_osts), ~np.isnan(target_osts))
    predicted_osts, target_osts = predicted_osts[non_nan], target_osts[non_nan]
    min_x, max_x = predicted_osts.min(), predicted_osts.max()
    stepsize = (max_x - min_x) / num_bins
    bins = np.arange(min_x, max_x, stepsize)
    binned_values = OrderedDict()
    for bin1, bin2 in zip(bins, bins[1:].tolist() + [np.inf]):
        mask = np.array([bin1 <= x < bin2 for x in predicted_osts])
        y = target_osts[mask]
        binned_values[bin1 + stepsize] = y
    binned_x, binned_y = list(binned_values.keys()), list(binned_values.values())
    binned_y_means, binned_y_err = [np.mean(y) for y in binned_y], [scipy.stats.sem(y) for y in binned_y]
    _plot(binned_x, binned_y_means, yerr=binned_y_err, correlation_p=(correlation.values.tolist(), p),
          filename='fig5', plot_type='errorbar')


def _plot(x, y, yerr=None, filename='osts', plot_type='scatter',
          trend_line=True, correlation_p=None):
    x, y, yerr = np.array(x), np.array(y), np.array(yerr)
    seaborn.set()
    seaborn.set_context('paper', font_scale=2)
    seaborn.set_style('whitegrid', {'axes.grid': False})
    pyplot.figure()
    plot = getattr(pyplot, plot_type)
    if plot_type == 'errorbar':
        idx = x.argsort()
        plot(x[idx], y[idx], yerr=yerr[idx], markersize=7.5, elinewidth=.5, fmt='o', color='#808080')
    elif plot_type == 'boxplot':
        plot(y, positions=x)
    elif plot_type == 'violinplot':
        plot(y, positions=x, showmeans=True, widths=8)
    else:
        plot(x, y)

    if plot_type in ['bar', 'errorbar']:
        pyplot.ylim(min(y) - 10, pyplot.ylim()[1])

    if trend_line:
        if isinstance(y, list) and isinstance(y[0], list):
            import itertools
            x = list(itertools.chain(*[[_x] * len(_y) for _x, _y in zip(x, y)]))
            y = list(itertools.chain(*y))
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        print("trend line", p)
        trend_x = list(sorted(set(x)))
        pyplot.plot(trend_x, p(trend_x), linestyle='dashed', color='#D4145A', linewidth=4)

    if correlation_p:
        correlation, p = correlation_p
        p_magnitude = np.round(np.log10(p))
        print(f"magnitude of {p} is {p_magnitude}")
        pyplot.text(pyplot.xlim()[0] + 10, pyplot.ylim()[1] - 10, f"r={correlation:.2f} (p<{10 ** p_magnitude})")

    pyplot.xlabel('$IT_{COR}$ object solution times')
    pyplot.ylabel('$IT_{monkey}$ object solution times')

    pyplot.tight_layout()
    seaborn.despine(right=True, top=True)
    target_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'osts', filename)
    for extension in ['png', 'pdf', 'svg']:
        pyplot.savefig(target_path + "." + extension)
    print(f"saved to {target_path}")


if __name__ == '__main__':
    fire.Fire()
