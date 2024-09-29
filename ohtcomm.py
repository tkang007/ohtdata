# filename: ohtcomm.py
# purpose: common objects: function, etc

# packages
import os
import pathlib
import glob
import datetime
import shutil
import random
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

import duckdb

import ohtconf as conf

# pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 200)  # Adjust width to your preference (e.g., 200)
# pd.set_option('display.max_colwidth', None)  # Adjust maximum column width


# functions
def get_multifiles_indir(adir: str, pat: str) -> list[str]:
    """get csv filenames in a directory

    Arguemnt:
        adir str - directory of the csv files
        pat str - filename glob pattern

    Return:
        list[str] - file paths

    Note:
        reference global constant, FILENAME_PATTERN
    """
    return sorted(glob.glob(str(pathlib.Path(adir) / pat)))


def get_multifiles_size(files: list[str]) -> list[int]:
    """Get size of multiple files

    Argument:
        files list[str] - list of file path
    Return:
        list[int] - list of file size
    """
    return [os.path.getsize(file_path) for file_path in files]


def read_csvfile(csvfile: str) -> tuple[pd.DataFrame | None, list[tuple[int, str, str]]]:
    """Read a csv file

    Argument:
        csvfile  str - csv file path

    Return: (dataframe, elines)
        dataframe - None for not found or error
        elines - list of (line_no, line_data, line_message)

    Note:
        reference conf constant, COLUMN_NAMES, COLUMN_TYPES, DATE_FORMAT
        manual parsing as csv, pandas failed on the dtype convert error

        COLUMN_NAMES = ["DATETM",       "TEM",     "PM1",    "PM2_5",   "PM10",  "CO",     "NH3",    "CT1",      "CT2",      "CT3",       "CT4"]
        COLUMN_TYPES = [datetime.datetime, np.float32, np.int16, np.int16, np.int16, np.int16, np.int16, np.float32, np.float32, np.float32, np.float32]
        ex data:  2024-07-29 09:43:18:696, 40.8,       10,       12,       13,       161,      88,       0.8,        1,          0.5,        0.6
    """
    vrows: list[list[datetime.datetime | np.float32 | np.int16]] = []
    elines: list[tuple[int, str, str]] = []  # error lines
    lineno: int = 0
    try:
        with open(csvfile, "r") as reader:
            for line in reader:
                line = line.strip()
                cols = line.split(",")
                row: list[datetime.datetime | np.float32 | np.int16] = []
                lineno += 1
                if len(cols) != len(conf.COLUMN_NAMES):
                    elines.append((lineno, line, f"column count not equal to {len(conf.COLUMN_NAMES)}."))
                else:
                    try:
                        row.append(datetime.datetime.strptime(cols[0], conf.DATE_FORMAT))
                        row.append(np.float32(cols[1]))

                        row.append(np.int16(cols[2]))  # pm
                        row.append(np.int16(cols[3]))
                        row.append(np.int16(cols[4]))
                        row.append(np.int16(cols[5]))  # co
                        row.append(np.int16(cols[6]))

                        row.append(np.float32(cols[7]))  # ct
                        row.append(np.float32(cols[8]))
                        row.append(np.float32(cols[9]))
                        row.append(np.float32(cols[10]))
                        vrows.append(row)
                    except ValueError as e:
                        elines.append((lineno, line, str(e)))
            # end for
        # end with
        if len(vrows) < 1:
            elines.append((0, "", "no valid row"))  # no=0,data="",message
            return None, elines
        return pd.DataFrame(vrows, columns=conf.COLUMN_NAMES), elines
    except Exception as e:
        elines.append((0, "", str(e)))
    return None, elines


def read_multifiles(files: list[str], logstep: int = 10, verbose: bool = False) -> pd.DataFrame | None:
    """Read multiple csv files

    Argument:
        files list[str] - file paths
        logstep int - logging step per files
        verbose bool - verbose mode

    Return:
        pd.DataFrame - dataframe for the csv files
    """
    dfall: pd.DataFrame = None

    cur_file_count: int = 0
    cur_file_path: str | None = None

    dataframes: list[pd.DataFrame] = []
    errinfos: list[tuple[str, list[tuple[int, str, str]]]] = []  # use tuple for list

    for afile in files:
        cur_file_path = afile
        cur_file_count += 1

        if logstep and (cur_file_count - 1) % logstep == 0:
            if verbose:
                print(f"file reading {cur_file_count} file={pathlib.Path(cur_file_path).name}")

        df, elines = read_csvfile(cur_file_path)
        if df is not None:
            dataframes.append(df)
        if len(elines) > 0:
            errinfos.append((cur_file_path, elines))

        if verbose:
            # file level report
            if len(elines) > 0:
                print(f"CSV error lines in file={pathlib.Path(cur_file_path).name}:")
                for eline in elines:
                    print(f'no={eline[0]}, line="{eline[1]}", message="{eline[2]}"')

    # report error
    # if verbose:
    if False:
        # report error
        if len(errinfos) > 0:
            print(f"CSV error report in {cur_file_count} files: {len(errinfos)} errors")
            for einfo in errinfos:
                for eline in einfo[1]:
                    print(f'file={pathlib.Path(einfo[0]).name}, no={eline[0]}, line="{eline[1]}", message="{eline[2]}"')
        else:
            print(f"CSV error report in {cur_file_count} files: 0 error")

    # report data
    if len(dataframes) > 0:
        dfall = pd.concat(dataframes)
        dfall.sort_values(conf.COLUMN_NAMES[0], inplace=True)  # ordering
        dfall.reset_index(drop=True, inplace=True)  # reindex default index: 0,1,2,...
    else:
        dfall = None
    if verbose:
        if dfall is not None:
            print(f"dataframe prepared with (rows,columns)={dfall.shape} in {cur_file_count} files.")
        else:
            print(f"dataframe not prepared as no valid data in {cur_file_count} files.")
    return dfall


def save_dftab(df: pd.DataFrame, tabname: str) -> None:
    """Save a dataframe to a duckdb table

    Create duckdb file when not exist, recreate table when exist

    Argument:
        df pd.DataFrame - a dataframe
        tabname str - table name

    Return: None
    """
    try:
        con = duckdb.connect(conf.DBFILE)  # created when not exist
        con.execute(f"DROP TABLE IF EXISTS {tabname}")
        con.execute(f"CREATE TABLE {tabname} AS SELECT * FROM df")
        con.close()
    except Exception as e:
        print(f"duckdb table={tabname} create exception in dbfile={conf.DBFILE}: {e}")
        raise


def read_tabdf(tabname: str) -> pd.DataFrame | None:
    """Read a duckdb table and return a dataframe

    Argument:
        tabname str - table name

    Return:
        pd.DataFrame - a dataframe
    """
    if not pathlib.Path(conf.DBFILE).exists():
        raise Exception(f"dbfile={conf.DBFILE} not exist")

    try:
        con = duckdb.connect(conf.DBFILE)
        df = con.execute(f"SELECT * FROM {tabname}").df()
        con.close()
        return df
    except Exception as e:
        print(f"duckdb table={tabname} read exception in dbfile={conf.DBFILE}: {e}")
        raise


def remove_file(afile: str) -> None:
    """Remove afile when exist

    Argument:
    afile str: file path
    """
    file_path = pathlib.Path(afile)
    if file_path.exists():
        file_path.unlink()


def remove_directory(adir: str) -> None:
    """Removes adir when exist, even if it contains files.

    Args:
        dirpat str - The path to the directory to remove.
    """
    dir_path = pathlib.Path(adir)
    if dir_path.exists():
        shutil.rmtree(dir_path)


def save_csvfile(df: pd.DataFrame, filename: str, adir: str) -> None:
    """save dataframe to csvfile

    Args:
    filename str: only filename with extension
    adir str: parent directory of file's directory

    Note:
        reference conf constant, COLUMN_NAMES, DATE_FORMAT, FLOAT_FORMAT
        use basename and extension of filename to determin the directory and filename.
        created file path : adir/basename/basename-NNN.extention
    """

    # make output directory
    namepath = pathlib.Path(filename)
    filebase, fileext = namepath.name.split(".")
    if not fileext:
        fileext = "csv"
    dirpath = pathlib.Path(adir) / filebase

    # recreate output directory
    remove_directory(str(dirpath))
    dirpath.mkdir(parents=True, exist_ok=True)

    # create multiple files base to be opened in excell
    if df.shape[0] < 1:
        print("save_csvfile, dataframe has no rows to write, skip")
        return

    num_chunks, m = divmod(df.shape[0], conf.CSVFILE_LINES)
    if m:
        num_chunks = num_chunks + 1

    cnt = 0
    for i in range(num_chunks):
        start_index = i * conf.CSVFILE_LINES
        end_index = min(start_index + conf.CSVFILE_LINES, df.shape[0])
        chunk = df.iloc[start_index:end_index]
        output_file = str(dirpath / f"{filebase}-{i+1:03d}.{fileext}")

        with open(output_file, "w") as of:
            line = 0
            for index, row in chunk.iterrows():
                if line != 0:
                    of.write("\n")
                line += 1
                # for col in conf.COLUMN_NAMES:
                for col in df.columns:  # for FLAG column
                    if col == conf.COLUMN_NAMES[0]:
                        if conf.DATETM_INCLUDE:
                            of.write(row[col].strftime(conf.DATE_FORMAT)[:-3])
                        else:
                            pass
                    elif col in conf.COLUMN_TEM + conf.COLUMN_CTA:
                        if not conf.DATETM_INCLUDE and col == conf.COLUMN_TEM[0]:
                            of.write("{:.1f}".format(row[col]))
                        else:
                            of.write(",{:.1f}".format(row[col]))
                    elif col in conf.COLUMN_PMA + conf.COLUMN_COA + ["FLAG"]:
                        of.write(",{:d}".format(row[col]))

        cnt = cnt + 1
        if i % 10 == 0:
            print(f"saved {i+1}/{num_chunks} csvfile={output_file}")

    print(f"saved {cnt}/{num_chunks} files")


def dfinfo(df: pd.DataFrame) -> None:
    """Display dataframe information"""

    print(f"shape:\n{df.shape}\n")

    print(f"index:\n dtype={df.index.dtype}, unique={df.index.is_unique}\n")

    print("info:")
    print(df.info(verbose=conf.VERBOSE))

    string_output = df.loc[:, conf.COLUMN_GRAPH].describe().round(decimals=2).to_string(line_width=240)
    print(f"\nstatistics:\n{string_output}\n")

    print(f"head:\n{df.head().to_string(line_width=240)}\n")


def custom_sequence(start: np.int16 | np.float32, stop: np.int16 | np.float32, patkind: int) -> NDArray[Any]:
    """Renturn non-linear increasing sequence values

    Non-linear increasing sequence gernation fuctions:
    0. cumsum - flex increasing
    1. exponential - late increasing
    2. logarithmic - early increasing
    3. quadratic - smooth increasing, coefficient
    4. cubic - smooth increasing, coefficient
    """

    intervals = np.linspace(0.1, 1.1, conf.POINTS["PATTERN"])

    # cumsum, cumulate sum function
    np.random.shuffle(intervals)
    sequence: NDArray[Any] = np.cumsum(intervals)

    kind = 5
    match patkind % kind:
        case 0:  # cumsum, cumulate sum fuction
            pass

        case 1:  # exponential function
            sequence = np.exp(sequence)

        case 2:  # logarithmic
            sequence = np.log(sequence)

        case 3:  # quadratic, slop depend on coefficent (a,b,c)
            sequence = sequence**2

        case _:  # cubic
            sequence = sequence**3

    sequence = start + (stop - start) * (sequence - sequence.min()) / (sequence.max() - sequence.min())
    return sequence.astype(np.float32)


def update_row_outer(df: pd.DataFrame, usestd: bool = False) -> Any:
    """Return closure for update outlier row"""

    sequences: dict[str, NDArray[Any]] = dict()
    outl_cnt = 0

    des = df.describe()

    def update_row_inner(row: Any) -> Any:
        nonlocal sequences
        nonlocal outl_cnt
        # nonlocal des # not assign, but reference

        repidx, patidx = divmod(row.name, conf.POINTS["PATTERN"])  # default index, 50
        if patidx == 0 or len(sequences) == 0:  # as row.name is logical index
            patkind = random.randint(0, conf.POINTS["PATTERN"])  # random patten order
            for colidx, col in enumerate(conf.COLUMN_GRAPH):  # use moving avg,std for row
                stddev = row[conf.MVSTD + col] if row[conf.MVSTD + col] != 0 else des.loc["std", col]
                minval = row[conf.MVAVG + col] + stddev * conf.SIGMA_NOISE  # 2-sigma
                maxval = (
                    row[conf.MVAVG + col] + stddev * conf.SIGMA_OUTLIER if usestd else conf.MAXVALS[col]
                )  # 6-sigmal or maxval
                sequences[col] = custom_sequence(minval, maxval, patkind)

        # NOTE: when make all column outlier at the same row, their correlation coefficent will be broken.
        if conf.OUTLIER_DISCRETE:
            # try to keep correlation coefficient by updating one column outlier value, other with mean value
            colidx = repidx % len(conf.COLUMN_GRAPH)
            colnam = conf.COLUMN_GRAPH[colidx]

            for col in conf.COLUMN_GRAPH:
                if col != colnam:  # take max(val,mean) for non-outlier column
                    if col in conf.COLUMN_PMA + conf.COLUMN_COA:
                        row[col] = np.int16(np.around(np.random.uniform(des.loc["25%", col], des.loc["75%", col])))
                    else:
                        row[col] = np.float32(np.around(np.random.uniform(des.loc["25%", col], des.loc["75%", col]), 1))
                else:  # take outlier value for outlier column
                    if col in conf.COLUMN_PMA + conf.COLUMN_COA:
                        row[col] = np.int16(np.around(sequences[col][patidx]))
                    else:
                        row[col] = np.float32(np.around(sequences[col][patidx], 1))

            row[conf.COLUMN_FLAG] = colidx + 1  # for ML feature selection, use colidx in conf.COLUMN_GRAPH + 1
        else:
            # large impoct on the correlation heatmap, scatter
            for col in conf.COLUMN_GRAPH:
                if col in conf.COLUMN_PMA + conf.COLUMN_COA:
                    row[col] = np.int16(np.around(sequences[col][patidx]))
                else:
                    row[col] = np.float32(np.around(sequences[col][patidx], 1))

            row[conf.COLUMN_FLAG] = -1  # negate when ML

        outl_cnt += 1

        if (outl_cnt - 1) % 100_000 == 0:
            print(f"outlier count={outl_cnt} / {des.loc['count', conf.COLUMN_NAMES[0]]}")

        return row

    return update_row_inner


def gen_outlier(df: pd.DataFrame, usestd: bool = False) -> pd.DataFrame:
    """Generate outlie dataframe

    Args:
        df pd.DataFrame - input dataframe

    Return
        pd.DataFrame - result datafrmae
    """
    np.random.seed(0)

    update_row = update_row_outer(df)  # closure

    df = df.apply(update_row, axis=1)

    return df


# eof
