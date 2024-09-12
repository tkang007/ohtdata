# filename: ohtgraph.py
# purpose: graph objects: function, etc

# packages
import pathlib
from collections.abc import Iterable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


import ohtconf as conf


def make_chartargs(
    kind: str,
    dfs: list[pd.DataFrame],
    labels: list[str],
    cols: list[str],
    grid: tuple[int, int],
    title: str | None = None,
) -> tuple[list[pd.DataFrame], list[str], list[str], tuple[int, int], str, tuple[int, int]]:
    """Check chart function args consitency and adjust dataframe slice and figsize"""

    assert isinstance(dfs, Iterable), "chart arg, dfs should be iterable"
    assert isinstance(labels, Iterable), "chart arg, labels should be iterable"
    assert isinstance(cols, Iterable), "chart arg, cols should be iterable"
    assert isinstance(grid, Iterable), "chart arg, grid should be iterable"

    assert len(dfs) == len(labels), "chart arg, dfs and labels should be same length"

    for i, df in enumerate(dfs):
        assert len(df) > 0, f"chart arg, {i+1} df should not be empty"

    if title is None:
        if kind == "line":
            title = "Data Trend with " + kind.capitalize() + " chart"
        else:
            title = "Data Distribution with " + kind.capitalize() + " chart"

    figsize = (conf.PLOTSIZE[0] * grid[1], conf.PLOTSIZE[1] * grid[0])

    return dfs, labels, cols, grid, title, figsize


def calc_lowhigh(dfs: list[pd.DataFrame], cols: list[str]) -> tuple[np.int16 | np.float32, np.int16 | np.float32]:
    """Calculate dataframe column values mix and max"""
    colranges: dict[str, list[np.float32 | np.int16]] = dict()
    for df in dfs:
        for col in cols:
            if col in colranges.keys():
                colranges[col] = [min(colranges[col][0], df[col].min()), max(colranges[col][1], df[col].max())]
            else:
                colranges[col] = [df[col].min(), df[col].max()]

    for col in cols:
        if col in conf.COLUMN_PMA + conf.COLUMN_COA:
            colranges[col] = [np.int16(colranges[col][0]) - 1, np.int16(colranges[col][1]) + 1]
        else:
            colranges[col] = [np.float32(np.floor(colranges[col][0]) - 1), np.float32(np.ceil(colranges[col][1])) + 1]

    # match between all columns
    low: np.int16 | np.float32 = np.int16(0)
    hig: np.int16 | np.float32 = np.int16(0)

    for k, v in colranges.items():
        if not low:
            low = v[0]
        else:
            low = min(low, v[0])  # type: ignore[call-overload]
        if not hig:
            hig = v[1]
        else:
            hig = max(hig, v[1])  # type: ignore[call-overload]
    return low, hig


def linechart(
    dfs: list[pd.DataFrame],
    labels: list[str],
    cols: list[str],
    grid: tuple[int, int],
    title: str | None = None,
    pngfile: str | None = None,
) -> None:
    """display and save line chart

    save chart file only when pngfile is not None

    Args:
        dfs list[DataFrame] - compared each column of the two dataframes
        labels list[str] - label for df
        cols list[str] - compared columns
        title str - title of the charts
        grid tuple[int,int] - row and column
        pngfile str - png filename only for saving
        xslice slice - charted df slice
    """
    dfs, labels, cols, grid, title, figsize = make_chartargs("line", dfs, labels, cols, grid, title)

    low, hig = calc_lowhigh(dfs, cols)

    fig, axs = plt.subplots(nrows=grid[0], ncols=grid[1], figsize=figsize)
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    for ic, (col, ax) in enumerate(zip(cols, axs)):
        for ik, df in enumerate(dfs):
            ax.plot(
                range(1, len(df) + 1), df[col], color=conf.COLORS[ik % len(conf.COLORS)], label=labels[ik] + "_" + col
            )
        ax.set_ylim(low, hig)
        ax.set_ylabel(col)
        ax.set_xlabel("0.1 sec")
        ax.set_title(col)
        ax.legend(loc="lower right", bbox_to_anchor=(1, 0))

        if col in conf.COLUMN_PMA + conf.COLUMN_COA:
            # Use FormatStrFormatter for y-axis to display as integers
            ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
        else:
            # Use FormatStrFormatter for y-axis to display as floats with 1 decimal places
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    fig.suptitle(title)

    plt.tight_layout()
    plt.show()  # fig.show not work at notebook as no event loop

    if pngfile is not None:
        filepath = pathlib.Path(conf.DIRCHART) / pngfile
        fig.savefig(
            filepath,
            dpi=conf.DPI_FILE,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format=None,
            transparent=False,
            bbox_inches=None,
            pad_inches=None,
        )


def histchart(
    dfs: list[pd.DataFrame],
    labels: list[str],
    cols: list[str],
    grid: tuple[int, int],
    title: str | None = None,
    pngfile: str | None = None,
) -> None:
    """display and save hist chart"""

    dfs, labels, cols, grid, title, figsize = make_chartargs("histogram", dfs, labels, cols, grid, title)

    low, hig = calc_lowhigh(dfs, cols)

    fig, axs = plt.subplots(nrows=grid[0], ncols=grid[1], figsize=figsize)
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    for ic, (col, ax) in enumerate(zip(cols, axs)):
        for ik, df in enumerate(dfs):
            ax.hist(df[col], color=conf.COLORS[ik % len(conf.COLORS)], label=labels[ik], bins=conf.BINS)
        ax.set_ylabel("Frequency")
        # ax.set_xlim(low,hig)
        ax.set_xlabel("Value")
        ax.set_title(col)
        ax.legend(loc="lower right", bbox_to_anchor=(1, 0))

        if col in conf.COLUMN_PMA + conf.COLUMN_COA:
            # Use FormatStrFormatter for y-axis to display as integers
            ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        else:
            # Use FormatStrFormatter for y-axis to display as floats with 1 decimal places
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    fig.suptitle(title)

    plt.tight_layout()
    plt.show()  # fig.show not work at notebook as no event loop

    if pngfile is not None:
        filepath = pathlib.Path(conf.DIRCHART) / pngfile
        fig.savefig(
            filepath,
            dpi=conf.DPI_FILE,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format=None,
            transparent=False,
            bbox_inches=None,
            pad_inches=None,
        )


def boxchart(
    dfs: list[pd.DataFrame],
    labels: list[str],
    cols: list[str],
    grid: tuple[int, int],
    title: str | None = None,
    pngfile: str | None = None,
) -> None:
    """display and save box chart"""

    dfs, labels, cols, grid, title, figsize = make_chartargs("boxplot", dfs, labels, cols, grid, title)

    low, hig = calc_lowhigh(dfs, cols)

    fig, axs = plt.subplots(nrows=grid[0], ncols=grid[1], figsize=figsize)
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    for ic, (col, ax) in enumerate(zip(cols, axs)):
        for ik, df in enumerate(dfs):
            ax.boxplot(df[col], positions=[ik + 1], widths=0.4, boxprops=dict(color=conf.COLORS[ik % len(conf.COLORS)]))

        ax.set_xticks([x + 1 for x in range(len(labels))])
        ax.set_xticklabels(labels)
        ax.set_title(col)

    fig.suptitle(title)

    plt.tight_layout()
    plt.show()  # fig.show not work at notebook as no event loop

    if pngfile is not None:
        filepath = pathlib.Path(conf.DIRCHART) / pngfile
        fig.savefig(
            filepath,
            dpi=conf.DPI_FILE,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format=None,
            transparent=False,
            bbox_inches=None,
            pad_inches=None,
        )


def violinchart(
    dfs: list[pd.DataFrame],
    labels: list[str],
    cols: list[str],
    grid: tuple[int, int],
    title: str | None = None,
    pngfile: str | None = None,
) -> None:
    """display and save box chart"""
    dfs, labels, cols, grid, title, figsize = make_chartargs("violinplot", dfs, labels, cols, grid, title)

    low, hig = calc_lowhigh(dfs, cols)

    fig, axs = plt.subplots(nrows=grid[0], ncols=grid[1], figsize=figsize)
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    for ic, (col, ax) in enumerate(zip(cols, axs)):
        for ik, df in enumerate(dfs):
            ax.violinplot(df[col], positions=[ik + 1])

        ax.set_xticks([x + 1 for x in range(len(labels))])
        ax.set_xticklabels(labels)
        ax.set_title(col)

    fig.suptitle(title)

    plt.tight_layout()
    plt.show()

    if pngfile is not None:
        filepath = pathlib.Path(conf.DIRCHART) / pngfile
        fig.savefig(
            filepath,
            dpi=conf.DPI_FILE,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format=None,
            transparent=False,
            bbox_inches=None,
            pad_inches=None,
        )
