# oht data preparation

## objective
- find abnormal,outlier, noise data and report noise data file when found
- geneate normal, outlier data and charts for trend, distribution and correlation.

## raw data
- csv files in this folder: 

\\BlueServer\ntcsoft_ftp\AFP_Data(열화상x)\240819 AFP Data\apfLog*

## report files
```
.\report\dataout\ohtchart  # various chart files
.\report\dataout\ohtoutl   # outlier csv files. not include normal data
.\report\dataout\ohtmix    # normal + outlier mixed files. the last additional column flag INT to mark outlier. 0 for normal, 1~10 for column ordered outlier label
.\report\html              # notebook executed result html files for KNN reports
```

## developer guide

### python virtual environment 
Python environment with miniconda, anaconda or python.
Below explanation is based on minicona.

```
mkdir c:\projects && cd c:\projects
mkdir ohtdata && cd ohtdata

conda create --name ohtdata 
:: conda env remove --name ohtdata
conda activate ohtdata      
:: conda deactivate

conda install pip
pip intall -r requirements.txt
```

## prepare raw sample data files
```
mkdir sample && mkdir sample\dataraw

:: copy sample data from raw data location
:: \\BlueServer\ntcsoft_ftp\AFP_Data(열화상x)\240819 AFP Data\apfLog* sample\dataraw
```

## project files
```
.gitignore
codecheck.bat
coderun.bat
oht1parse.ipynb
oht2output.ipynb
oht3graph.ipynb
oht4knn.ipynb
ohtcomm.py
ohtconf.py
ohtgraph.py
pyproject.toml
README.md
requirements.txt
report/
sample/

```
## code check - lint, format and static type check

codecheck.bat  - use ruff, mypy with pyproject.toml config file.

## run notebook at browser
```
jupyter lab 

# check ohtconf.py  -  DIRRAW for raw csvfile location 

# run notebooks in this order

oht1parse.ipynb - parse raw csvfile and save them to duckdb ohtraw table 
oht2output.ipynb - read duckdb ohtraw table, generate noise,normal,outlier csvfiles and duckdb tables 
oht3graph.ipynb - read duckdb ohtraw, ohtnoise, ohtnorm, ohtoutl tables and generate chart image files 


# find notebook output files in DIROUT folder

sample/dataout/ohtnoise/ohtnoise-NNN.csv - noise data
sample/dataout/ohtnorm/ohtnorm-NNN.csv - normal data
sample/dataout/ohtoutl/ohtoutl-NNN.csv - outlier data

sample/dataout/ohtchart/*.png -  chart image files 

# duckdb file, shared between notebooks

sample/sample.duckdb 
```

## run notebook at command prompt, cmd.exe
result notebooks in .\tmp\notebook-HHMMSS.ipynb 
result data files is same with browser use case.
```
# run 3 notebooks
coderun.bat   

# run 3 notebooks with DIRFLAG=large variable for large datafiles, see ohtconf.py 
set DIRFLAG=large
coderun.bat   

# run 1 notebook, use arg1 with  parse or output or graph
coderun.bat parse
```

