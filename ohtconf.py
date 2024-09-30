# filename: ohtconf.py
# purpose: configuration

import os
import datetime
import pathlib
import numpy as np

#
# basic config
#
_dirflag = os.getenv("DIRFLAG", "small")  # increase will be used when increase small dataset to large dataset at output
if _dirflag == "small":
    DIRRAW = r".\sample\dataraw"
    DIROUT = r".\sample\dataout"
else:  # prod
    DIRRAW = r"C:\projects\ohtdatafiles\dataraw"
    # DIROUT = r"C:\projects\ohtdatafiles\dataout"
    DIROUT = r".\report\dataout"

assert pathlib.Path(DIRRAW).exists(), f"conf, DIRRAW={DIRRAW} dir not exist"

FILENAME_PATTERN = "afpLog_????-??-??_??????.csv"  # afpLog_2024-05-29_091339.csv

COLUMN_TEM = ["TEM"]

COLUMN_PMA = ["PM1", "PM2_5", "PM10"]  # group of PM*, dtype=int

COLUMN_COA = ["CO", "NH3"]  # group of CO and NH3. dtype=int

COLUMN_CTA = ["CT1", "CT2", "CT3", "CT4"]  # group of CT*

COLUMN_GRAPH = COLUMN_TEM + COLUMN_PMA + COLUMN_COA + COLUMN_CTA

# ex data:2024-07-29 09:43:18:696, 40.8,  10,   12,     13,    161,  88,    0.8,   1,     0.5,   0.6
# COLUMN_NAMES = ["DATETM",   "TEM","PM1","PM2_5","PM10", "CO", "NH3", "CT1", "CT2", "CT3", "CT4"]
COLUMN_NAMES = ["DATETM"] + COLUMN_GRAPH

COLUMN_TYPES = [
    datetime.datetime,
    np.float32,
    np.int16,
    np.int16,
    np.int16,
    np.int16,
    np.int16,
    np.float32,
    np.float32,
    np.float32,
    np.float32,
]  # datqframe
COLUMN_DBTYPES = [
    "TIMESTAMP",
    "FLOAT",
    "INTEGER",
    "INTEGER",
    "INTEGER",
    "INTEGER",
    "INTEGER",
    "FLOAT",
    "FLOAT",
    "FLOAT",
    "FLOAT",
]  # db

DATE_FORMAT = "%Y-%m-%d %H:%M:%S:%f"  # 3 digits after second

FLOAT_FORMAT = "%.1f"  # 1 digit after period

# 6M sample data's statistics for max values
# 	   DATETM                         TEM          PM1        PM2_5      PM10       CO         NH3        CT1        CT2        CT3         CT4
# count  6095021                        6095021.0  6095021.0  6095021.0  6095021.0  6095021.0  6095021.0  6095021.0  6095021.0  6095021.0   6095021.0
# mean   2024-01-04 12:39:11.000005120  42.1       9.4        10.8       12.3       157.2      98.2        0.6       1.0        0.4         0.6
# min    2024-01-01 00:00:00            22.6       7.0         8.0        9.0        31.0      66.0       -0.5       0.2        0.2         0.3
# 25%    2024-01-02 18:19:35.500000     40.8       8.0         9.0       11.0       141.0      88.0        0.3       0.9        0.4         0.5
# 50%    2024-01-04 12:39:11            41.7       9.0        11.0       12.0       158.0      93.0        0.6       1.0        0.4         0.6
# 75%    2024-01-06 06:58:46.500000     43.5      10.0        12.0       13.0       170.0      110.0       0.9       1.0        0.5         0.6
# max    2024-01-08 01:18:22            45.5      27.0        29.0       42.0       253.0      140.0       3.0       1.5        0.8         1.0
# std    NaN                             1.7       1.6         1.6        1.5        28.8      12.4        0.3       0.1        0.1         0.1

MAXVALS: dict[str, np.int16 | np.float32] = dict()
for col in COLUMN_GRAPH:
    match col:
        case "TEM":
            MAXVALS[col] = np.float32(50.0)

        case "PM1":
            MAXVALS[col] = np.int16(50)
        case "PM2_5":
            MAXVALS[col] = np.int16(50)
        case "PM10":
            MAXVALS[col] = np.int16(50)

        case "CO":
            MAXVALS[col] = np.int16(300)
        case "NH3":
            MAXVALS[col] = np.int16(200)

        case "CT1":
            MAXVALS[col] = np.float32(5.0)
        case "CT2":
            MAXVALS[col] = np.float32(5.0)
        case "CT3":
            MAXVALS[col] = np.float32(5.0)
        case "CT4":
            MAXVALS[col] = np.float32(5.0)
        case _:
            assert False, f"conf, column={col} invalid"

#
#  work config
#
DBFILE = str(
    pathlib.Path(DIROUT).parent / (str(pathlib.Path(DIROUT).parent.name) + ".duckdb")
)  # high performance than csvfiles, avoid parsing in between notebooks

TABNAME_RAW = "ohtraw"  # raw
TABNAME_NORM = "ohtnorm"  # normal
TABNAME_NOISE = "ohtnoise"  # noise
TABNAME_OUTL = "ohtoutl"  # outlier
TABNAME_MIX = "ohtmix"  # normal + outlier for ML

# FILENAME_RAW = TABNAME_RAW + '.csv'  # raw
FILENAME_NORM = TABNAME_NORM + ".csv"  # normal
FILENAME_NOISE = TABNAME_NOISE + ".csv"  # noise
FILENAME_OUTL = TABNAME_OUTL + ".csv"  # outlier
FILENAME_MIX = TABNAME_MIX + ".csv"  # normal + outlier for ML

# parsing
DBFILE_RECREATE = True  # flag for recreating dbfile. True when change input data or its scope.

DATETM_INCLUDE = True  # include or exclude datetm column oon the output report csvfile.

if DATETM_INCLUDE:
    INPUT_MAXSIZE = 550 * 1024 * 1024  # parsed csvfile max size as expected normal data 400,
else:
    INPUT_MAXSIZE = 750 * 1024 * 1024  # parsed csvfile max size as expected normal data 400,

REPORT_MINSIZE = 500 * 1024 * 1024  # required output csvfiles (normal+outler) size.

SKIP_FILES = 0  # skip files in the dataset dir in filename order. 0 or None for no skip
LIMIT_FILES = 0  # limit files in the dataset dir in filename order. 0 or None for no limit
LOGSTEP_FILES = 10  # progressive logging step of files

# calculating
DATAPOINT_INTERVAL = 0.1  # data point inerval in second

OUTLIER_INTERVAL = 5  # outlier interval in second
assert (
    DATAPOINT_INTERVAL < OUTLIER_INTERVAL
), f"conf, OUTLIER_INTERVAL={OUTLIER_INTERVAL} should be greater than DATAPOINT_INTERVAL={DATAPOINT_INTERVAL}"

OUTLIER_RATIO = 2 / 8  # normal : outlier = 6 : 2 (3 : 1)

OUTLIER_DISCRETE = True  # each columns's outlier is on the different row. True for ML and False for not enough outlier displayed for mix data chart.

POINTS = {
    "PATTERN": round(1 / DATAPOINT_INTERVAL * OUTLIER_INTERVAL)  # ex,  50 = (1/0.1 * 5), outlier pattern range points
}
POINTS["MOVING"] = POINTS["PATTERN"] * 12  # ex, 1 min, 50 / sec * 12 = 600 points for moving avg.
# POINTS["MOVING"] = POINTS["PATTERN"] * 12 * 60  # ex, 1 hour, 50 / sec * 12 * 60 = 36000 points for moving avg. more flatten

assert (
    POINTS["PATTERN"] < POINTS["MOVING"]
), f"conf. MOVING={POINTS['MOVING']} points shoud greater than PATTERN={POINTS['PATTERN']}"

SIGMA_NOISE = 2  # max sigma value for noise detection. adjust value to increase or decrease noise range
assert 1 < SIGMA_NOISE < 6, f"conf, SIGMA_NOISE={SIGMA_NOISE} invalid"

SIGMA_OUTLIER = 6  # max sigma value for outerlier generation. adjust value to increase or decrease outlier value range
assert 2 < SIGMA_OUTLIER < 9, f"conf, SIGMA_NOISE={SIGMA_OUTLIER} invalid"

MVAVG = "MVAVG_"  # column name prefix for moving average
MVSTD = "MVSTD_"  # moving stddev
MVSIG = "MVSIG_"  # moving sigma of column  value

COLUMN_FLAG = "FLAG"  # flag column to mark noise and outler row

# output file

CSVFILE_LINES = 36_000  # as much as input file line count
assert CSVFILE_LINES > 0, f"conf, CSVFILE_LINES={CSVFILE_LINES} invalid"

# chart
DIRCHART = str(pathlib.Path(DIROUT) / "ohtchart")

CHARTSLICE = slice(
    POINTS["MOVING"],
    POINTS["MOVING"] + POINTS["PATTERN"] * 20,  # line chart x-axis count
    1,
)  # ex)  (600, 500, 1), adjust start,stop,step. exclude initial moving window. 5 patterns
assert CHARTSLICE.stop > CHARTSLICE.start, f"conf, CHARTSLICE={CHARTSLICE} invalid"

BINS = 30  # histogram bins

COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "black", "gray", "pink", "brown"]  # 10 colors in a plot

# PLOTSIZE = [10, 6]  # recommended plotsize on the notbook.  width, height in pixel
# DPI = 100  # default dpi of notebook

PLOTSIZE = [7, 5]  # recommended plotsize on the notbook.  width, height in pixel
DPI = 300  # recommended DPI, density per inch on the file for document.

SCATTER_INCLUDE = True  # flag for scatter chart generation

MIX_INCLUDE = False  # flag for mixed data charting

# KNN
N_NEIGHBORS = range(2, 12)  # time issue when extend
TRAIN_SIZE = 800_000

# etc
VERBOSE = True

# eof
