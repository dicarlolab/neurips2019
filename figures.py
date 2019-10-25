import os

import fire
import numpy as np
import scipy.stats
import pandas
import matplotlib
import matplotlib.pyplot as plt
import itermplot

from matplotlib.ticker import FuncFormatter

OUTPUT = 'generated'
RNG = np.random.RandomState(0)

# reverse plot color is output is in the terminal
itermplot.THEME = 'rv' if OUTPUT is None else ''
matplotlib.use('module://itermplot')

matplotlib.rcParams['font.sans-serif'] = 'Source Sans Pro'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.weight'] = 'semibold'
matplotlib.rcParams['font.size'] = '10'
matplotlib.rcParams['pdf.fonttype'] = 42


def formatter(x, pos):
    if x != 0:
        if abs(x) > 1:  # hacky
            x = f'{int(x)}'
        else:
            x = ('%.2f' % x).lstrip('0').rstrip('0')
    else:
        x = '0'
    return x


def corr(df, x, y, report_p=False):

    sel = df[x].notnull() & df[y].notnull()
    try:
        r, p = scipy.stats.pearsonr(df.loc[sel, x], df.loc[sel, y])
    except:
        breakpoint()
    if p > .05:
        r = 'r: n.s.'
    else:
        r = 'r = ' + f'{r:.2f}'.lstrip('0')

    if report_p:
        if p < .001:
            p_level = f'{p:.0e}'
        elif p < .01:
            p_level = '.01'
        elif p < .05:
            p_level = '.05'
        else:
            p_level = ''

        if p_level != '':
            r += f' (p < {p_level})'
    
    return r


def output_paper_quality(ax, title=None, xlabel=None, ylabel=None):
    ax.set_title(title)
    ax.set_xlabel(xlabel, weight='semibold', size=12)
    ax.set_ylabel(ylabel, weight='semibold', size=12)

    ax.xaxis.set_major_formatter(FuncFormatter(formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(formatter))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(bottom=False, left=False)
    

def output(filename):
    plt.tight_layout()
    if OUTPUT is not None:
        plt.savefig(os.path.join(OUTPUT, f'{filename}.png'), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(OUTPUT, f'{filename}.pdf'), bbox_inches='tight', transparent=True)
    else:
        plt.show()


def scatterplot(ax, df, x, y, scale=30, annotate=False):
    dff = df[df.Model != 'CORnet-S']
    jitter = dff.jitter if 'jitter' in dff else 0
    ax.scatter(dff[x] + jitter, dff[y], s=scale, color=dff.color, alpha=.7, edgecolors='none')#linewidths=.1)
    
    dff = df[df.Model == 'CORnet-S']
    jitter = dff.jitter if 'jitter' in dff else 0
    ax.scatter(dff[x] + jitter, dff[y], s=2*scale, color=dff.color, alpha=.7, edgecolors='none')#, linewidths=.1)
    
    if annotate:
        for idx, row in df.iterrows():
            ax.text(row[x], row[y], row.Model)
    return ax


def read_common_data():
    df = pandas.read_csv('data/data.csv')

    for idx, row in df.iterrows():
        if row.Model.lower().startswith('basenet'):
            color = 'gray'
        elif row.Model.lower().startswith('cornet'):
            color = 'crimson'
        else:
            color = '#078930'
        df.loc[idx, 'color'] = color

    return df


def _fig1(df, scale=15, inset=False):
    ax = plt.subplot(111)
    scatterplot(ax, df, x='ImageNet', y='Brain-Score', scale=scale)
    if not inset:
        df = df[df.ImageNet < .7]
    r = corr(df, x='ImageNet', y='Brain-Score')
    
    ax.annotate(r, xy=(.75, .1),
                xycoords='axes fraction',
                # horizontalalignment='left', verticalalignment='top',
                fontsize=10)
    output_paper_quality(ax, 
            xlabel='ImageNet top-1 performance',
            ylabel='Brain-Score')


def fig1():
    df = read_common_data()
    plt.figure(figsize=(4, 4))
    _fig1(df, scale=15)
    output('fig1')
    plt.figure(figsize=(3, 3))
    _fig1(df[(df['ImageNet'] >= .7) & (df.Model != 'CORnet-S')], scale=45, inset=True)
    output('fig1_inset')


def _fig2(ax, df, x, y, title=None, xlabel=None, ylabel=None, index=0, r=None):
    scatterplot(ax, df, x, y)
    r = corr(df, x, y)
    ax.annotate(r, xy=(.75, .1), xycoords='axes fraction', fontsize=10)
    ax.annotate(f"({'abcd'[index]})", xy=(.05 + .24 * index, .9),
                xycoords='figure fraction', fontsize=20)
    output_paper_quality(ax, title=title,
        xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, va='top', pad=20)


def fig2():
    df = read_common_data()
    dff = df[~df.Model.str.startswith('BaseNet')]

    fig, axes = plt.subplots(ncols=4, figsize=(12,3))
    _fig2(axes[0], dff, x='IT', y='IT (new data)',
        xlabel='IT score (original neurons)', ylabel='IT score (new neurons)',
        title='New neural recordings,\nsame images', index=0, r=.93)
    _fig2(axes[1], dff, x='IT', y='IT (new images)',
        xlabel='IT score (original neurons)', ylabel='IT score (new neurons)',
        title='New neural recordings,\nnew images', index=1, r=.76)
    _fig2(axes[2], dff, x='Behavior', y='Behavior (new data)',
        xlabel='Behavioral score (original)', ylabel='Behavioral score (new)',
        title='New behavioral recordings,\nnew images', index=2, r=.83)
    _fig2(axes[3], dff, x='Brain-Score', y='CIFAR-100',
        xlabel='Brain-Score', ylabel='CIFAR-100 transfer',
        title='CIFAR-100 transfer', index=3, r=.69)
    output('fig2')


def _fig3(ax, df, y, ylabel=None, title=None):
    scatterplot(ax, df, x='Depth', y=y)
    ax.set_xscale('log')
    ax.set_xticks([10, 25, 50, 100, 200])
    output_paper_quality(ax, title=title,
        xlabel='Model Depth (log scale)', ylabel=ylabel)


def fig3():
    df = read_common_data()
    dff = df[~df.Model.str.startswith('BaseNet')].copy()

    fig, axes = plt.subplots(ncols=3, figsize=(9,3), sharex=True)
    rangex = .05 * (dff['Depth'].max() - dff['Depth'].min())

    jitter = []
    for idx, row in dff.iterrows():
        if row.Model.startswith('MobileNet'):
            j = RNG.uniform(-rangex, rangex)
        else:
            j = 0
        jitter.append(j)
    dff['jitter'] = jitter

    _fig3(axes[0], dff, y='Brain-Score', ylabel='Brain-Score')
    _fig3(axes[1], dff, y='ImageNet', ylabel='ImageNet top-1')
    _fig3(axes[2], dff, y='CIFAR-100', ylabel='CIFAR-100 transfer')
    output('fig3')


def fig_a1():
    df = read_common_data()
    dff = df[(df['V4 number of features'] < 20000) & (df['IT number of features'] < 20000)]

    fig, axes = plt.subplots(ncols=2, figsize=(6,3))
    scatterplot(axes[0], dff, x='V4 number of features', y='V4')
    output_paper_quality(axes[0], xlabel='Number of features', ylabel='V4 neural score')
    scatterplot(axes[1], dff, x='IT number of features', y='IT')
    output_paper_quality(axes[1], xlabel='Number of features', ylabel='IT neural score')
    output('fig_a1')


def fig_a2():
    df = pandas.read_csv('data/cornet_search.csv')

    plt.figure(figsize=(6, 4))
    ax = plt.subplot(111)
    ax.scatter(df['ImageNet'], df['Behavior'], s=10, color='gray', alpha=.7, edgecolors='none')
    r = corr(df, x='ImageNet', y='Behavior')
    ax.annotate(r, xy=(.75, .1), xycoords='axes fraction', fontsize=10)
    output_paper_quality(ax, xlabel='ImageNet top-1', ylabel='Behavioral score')
    output('fig_a2')


def _fig_a3(ax, df, region, time):
    y = f'{region} ({time})'
    ms = 100 if time == 'early' else 200
    scatterplot(ax, df, x='ImageNet', y=y)
    r = corr(df, x='ImageNet', y=y, report_p=True)
    ax.annotate(r, xy=(.5, .1), xycoords='axes fraction', fontsize=10)
    output_paper_quality(ax, xlabel='ImageNet top-1',
                         ylabel=f'{region} neural score (at {ms} ms)',
                         title=f'{region} {time}')


def fig_a3():
    df = read_common_data()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6,6))
    _fig_a3(axes[0,0], df, region='V4', time='early')
    _fig_a3(axes[0,1], df, region='V4', time='late')
    _fig_a3(axes[1,0], df, region='IT', time='early')
    _fig_a3(axes[1,1], df, region='IT', time='late')
    output('fig_a3')


def _highlight_max(data):
    if data.dtype.name == 'object':  # got a string (model)
        return data

    max_idx = data.idxmax()
    formatted_data = []
    for idx, value in data.iteritems():
        if value != 0:
            value = f'{value:.3f}'.lstrip('0').rstrip('0')
        else:
            value = '0'

        if idx == max_idx:
            value = '\\textbf{' + value + '}'
        formatted_data.append(value)

    return formatted_data


def table_a1():
    df = read_common_data()
    dff = df[df['Brain-Score'].notnull() & ~df.Model.str.startswith ('BaseNet') & ~df.Model.str.startswith('MobileNet')]

    idx = df.loc[df.Model.str.startswith ('BaseNet'), 'Brain-Score'].idxmax()
    best_basenet = df.loc[idx].copy()
    best_basenet['Model'] = 'Best BaseNet'
    dff = dff.append(best_basenet)

    idx = df.loc[df.Model.str.startswith ('MobileNet'), 'Brain-Score'].idxmax()
    best_mobilenet = df.loc[idx].copy()
    best_mobilenet['Model'] = 'Best MobileNet'
    dff = dff.append(best_mobilenet)

    dff = dff.sort_values(by='Brain-Score', ascending=False)
    dff = dff[['Model', 'Brain-Score', 'V4', 'IT', 'OST', 'Behavior']].apply(_highlight_max)
    
    dff.to_latex(os.path.join(OUTPUT, 'table_a1.tex'), escape=False, index=False)


def gen_all():
    fig1()
    fig2()
    fig3()
    # data for fig4 is not provided
    # fig5 is generated by fig5.py from scratch
    table_a1()
    fig_a1()
    fig_a2()
    fig_a3()


if __name__ == '__main__':
    fire.Fire()