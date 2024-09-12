# filename: ohtconf.py
# purpose: configuration

import datetime
import pathlib
import numpy as np

#
# basic config
#
if 1 == 1:  # dev
    DIRRAW = r".\sample\dataraw"
    DIROUT = r".\sample\dataout"
else:  # prod
    DIRRAW = r"C:\projects\ohtdatafiles\dataraw"
    DIROUT = r"C:\projects\ohtdatafiles\dataout"

assert pathlib.Path(DIRRAW).exists(), f"conf, DIRRAW={DIRRAW} dir not exist"

FILENAME_PATTERN = "afpLog_????-??-??_??????.csv"  # afpLog_2024-05-29_091339.csv

COLUMN_TEM = ["TEMPER"]

COLUMN_PMA = ["PM1", "PM2_5", "PM10"]  # group of PM*, dtype=int

COLUMN_COA = ["CO", "NH3"]  # group of CO and NH3. dtype=int

COLUMN_CTA = ["CT1", "CT2", "CT3", "CT4"]  # group of CT*

COLUMN_GRAPH = COLUMN_TEM + COLUMN_PMA + COLUMN_COA + COLUMN_CTA

# ex data:2024-07-29 09:43:18:696, 40.8,  10,   12,     13,    161,  88,    0.8,   1,     0.5,   0.6
# COLUMN_NAMES = ["DATETM",   "TEMPER","PM1","PM2_5","PM10", "CO", "NH3", "CT1", "CT2", "CT3", "CT4"]
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

# """Sample data statics to determine column max value:

#         DATETM                        TEMPER    PM1       PM2_5     PM10      CO        NH3       CT1       CT2       CT3       CT4
# count    360000                       360000.0  360000.0  360000.0  360000.0  360000.0  360000.0  360000.0  360000.0  360000.0  360000.0
# mean    2024-07-29 14:43:18.669020    40.7      10.9      12.3      13.6      160.3     88.5      0.9       1.0       0.5       0.5
# min     2024-07-29 09:43:18.696000    40.3       9.0      11.0      12.0      112.0     70.0      0.6       0.8       0.3       0.3
# 25%     2024-07-29 12:13:18.694000    40.5      10.0      12.0      13.0      158.0     88.0      0.8       1.0       0.4       0.5
# 50%     2024-07-29 14:43:18.671500    40.6      10.0      12.0      13.0      161.0     89.0      0.9       1.0       0.5       0.5
# 75%     2024-07-29 17:13:18.644000    40.8      13.0      13.0      15.0      163.0     89.0      0.9       1.0       0.5       0.6
# max     2024-07-29 19:43:18.619000    41.8      15.0      21.0      26.0      170.0     96.0      1.1       1.2       0.8       0.7
# std     NaN                            0.2       1.4       0.5       1.0        3.6      1.7      0.1       0.1       0.1       0.1
# """

MAXVALS: dict[str, np.int16 | np.float32] = dict()
for col in COLUMN_GRAPH:
    match col:
        case "TEMPER":
            MAXVALS[col] = np.float32(50.0)

        case "PM1":
            MAXVALS[col] = np.int16(30)
        case "PM2_5":
            MAXVALS[col] = np.int16(30)
        case "PM10":
            MAXVALS[col] = np.int16(30)

        case "CO":
            MAXVALS[col] = np.int16(180)
        case "NH3":
            MAXVALS[col] = np.int16(100)

        case "CT1":
            MAXVALS[col] = np.float32(2.0)
        case "CT2":
            MAXVALS[col] = np.float32(2.0)
        case "CT3":
            MAXVALS[col] = np.float32(2.0)
        case "CT4":
            MAXVALS[col] = np.float32(2.0)
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

# FILENAME_RAW = TABNAME_RAW + '.csv'  # raw
FILENAME_NORM = TABNAME_NORM + ".csv"  # normal
FILENAME_NOISE = TABNAME_NOISE + ".csv"  # noise
FILENAME_OUTL = TABNAME_OUTL + ".csv"  # outlier

# parsing
DBFILE_RECREATE = True  # flag for recreating dbfile. True when change input data or its scope.

DATETM_INCLUDE = True  # include or exclude datetm column oon the output report csvfile.

if DATETM_INCLUDE:  #
    INPUT_MAXSIZE = 400 * 1024 * 1024  # parsed csvfile max size as expected normal data 400,
else:
    INPUT_MAXSIZE = 650 * 1024 * 1024  # parsed csvfile max size as expected normal data 400,

REPORT_MINSIZE = 500 * 1024 * 1024  # required output csvfiles (normal+outler) size.
# report files (normal, 374MB + outlier, 126 MBfor INPUT_MAXSIZE = 400MB, when include DATETM in output csvfile)
# report files (normal, 380MB + outlier, 128 MBfor INPUT_MAXSIZE = 6500MB, when exclude DATETM in output csvfile)

SKIP_FILES = 0  # skip files in the dataset dir in filename order. 0 or None for no skip
LIMIT_FILES = 0  # limit files in the dataset dir in filename order. 0 or None for no limit
LOGSTEP_FILES = 10  # progressive logging step of files

# calculating
DATAPOINT_INTERVAL = 0.1  # data point inerval in second
OUTLIER_INTERVAL = 5  # outlier interval in second
assert (
    DATAPOINT_INTERVAL < OUTLIER_INTERVAL
), f"conf, OUTLIER_INTERVAL={OUTLIER_INTERVAL} should be greater than DATAPOINT_INTERVAL={DATAPOINT_INTERVAL}"

OUTLIER_RATIO = 2 / 6  # normal : outlier = 6 : 2, 1/3

POINTS = {
    "PATTERN": round(1 / DATAPOINT_INTERVAL * OUTLIER_INTERVAL)  # ex,  50 = (1/0.1 * 5), outlier pattern range points
}
POINTS["MOVING"] = POINTS["PATTERN"] * 12  # ex, 600= 60 * 12, x times of pattern points for moving avg.

assert (
    POINTS["PATTERN"] < POINTS["MOVING"]
), f"conf. MOVING={POINTS['MOVING']} points shoud greater than PATTERN={POINTS['PATTERN']}"

SIGMA_NOISE = 2  # max sigma value for noise detection. adjust value to increase or decrease noise range
assert 1 < SIGMA_NOISE < 6, f"conf, SIGMA_SOISE={SIGMA_NOISE} invalid"

SIGMA_OUTLIER = 6  # max sigma value for outerlier generation. adjust value to increase or decrease outlier value range
assert 2 < SIGMA_OUTLIER < 9, f"conf, SIGMA_SOISE={SIGMA_OUTLIER} invalid"

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
    POINTS["MOVING"], POINTS["MOVING"] + POINTS["PATTERN"] * 10, 1
)  # ex)  (600, 500, 1), adjust start,stop,step. exclude initial moving window. 5 patterns
assert CHARTSLICE.stop > CHARTSLICE.start, f"conf, CHARTSLICE={CHARTSLICE} invalid"

BINS = 10  # histogram bins

COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "black", "gray", "pink", "brown"]  # 10 colors in a plot

# PLOTSIZE = [10, 6]  # recommended plotsize on the notbook.  width, height in pixel
# DPI = 100  # default dpi of notebook

PLOTSIZE = [7, 5]  # recommended plotsize on the notbook.  width, height in pixel
DPI = 300  # recommended DPI, density per inch on the file for document.

# etc
VERBOSE = True

# eof
