# oht data preparation

## objective
- find abnormal,outlier, noise data and report noise data file when found
- geneate normal, outlier data 

## raw data
- csv files in this folder: 

\\BlueServer\ntcsoft_ftp\AFP_Data(열화상x)\240819 AFP Data\apfLog*

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
README.md
requirements.txt
codecheck.bat

ohtconf.py
ohtcomm.py
ohtgraph.py
oht1parse.ipynb 
oht2output.ipynb
oht3graph.ipynb 

sample/dataraw   
```
## code lint, format and static type check

codecheck.bat 

## start jupyter lab
jupyter lab 

## run jupyter notebooks 
- check ohtconf.py  -  DIRRAW for raw csvfile location 

- run notebooks in this order
```
oht1parse.ipynb - parse raw csvfile and save them to duckdb ohtraw table 
oht2output.ipynb - read duckdb ohtraw table, generate noise,normal,outlier csvfiles and duckdb tables 
oht3graph.ipynb - read duckdb ohtraw, ohtnoise, ohtnorm, ohtoutl tables and generate chart image files 
```

- find notebook output files in DIROUT foler
```
sample/dataout/ohtnoise/ohtnoise-NNN.csv - noise data
sample/dataout/ohtnorm/ohtnorm-NNN.csv - normal data
sample/dataout/ohtoutl/ohtoutl-NNN.csv - outlier data

sample/dataout/ohtchart/*.png -  chart image files 
```
- find duckdb file
```
sample/sample.duckdb 
```
