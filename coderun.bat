
::
:: run juptyer notebooks 
::
:: uage: coderun.bat [ {parse | output | graph | clear }]
::    each arg depend on the previos arg result, duckdb file.
::    clear arg for clear cell output before git add file
::
:: environment variables:
::   DIRFLAG={small | large}  used to control DIRRAW,DIROUT location. default=small
::     small for using .\sample folder which defined at ohtconf.py
::     large for using other folder which defined at ohtconf.py  
::     use "set DIRFLAG=large" command before run this script.
::     use "set DIRFLAG=" command to unset this flag.
::

echo Current DIRFLAG variable value %DIRFLAG%

@echo off
:: Prepare filename part
set HOUR=%TIME:~0,2%
set MINUTE=%TIME:~3,2%
set SECOND=%TIME:~6,2%
:: Remove leading spaces from HOUR
if "%HOUR:~0,1%" == " " set HOUR=0%HOUR:~1,1%
set NAMEPART=%HOUR%%MINUTE%%SECOND%

:: Create subdirectories for output files
if not exist ".\tmp" mkdir ".\tmp"
if not exist ".\report" mkdir ".\report"
if not exist ".\report\html" mkdir ".\report\html"

time /t
if "%1"=="" (
    papermill oht1parse.ipynb  .\tmp\oht1parse-%DIRFLAG%.ipynb
    jupyter nbconvert --to html .\tmp\oht1parse-%DIRFLAG%.ipynb --output=..\report\html\oht1parse-%DIRFLAG%.html

    time /t
    papermill oht2output.ipynb .\tmp\oht2output-%DIRFLAG%.ipynb
    jupyter nbconvert --to html .\tmp\oht2output-%DIRFLAG%.ipynb --output=..\report\html\oht2output-%DIRFLAG%.html

    time /t
    papermill oht3graph.ipynb  .\tmp\oht3graph-%DIRFLAG%.ipynb
    jupyter nbconvert --to html .\tmp\oht3graph-%DIRFLAG%.ipynb --output=..\report\html\oht3graph-%DIRFLAG%.html
    
    time /t
    papermill oht4knn.ipynb  .\tmp\oht4knn-%DIRFLAG%.ipynb
    jupyter nbconvert --to html .\tmp\oht4knn-%DIRFLAG%.ipynb --output=..\report\html\oht4knn-%DIRFLAG%.html
) else (
    if "%1"=="parse" (
        papermill oht1parse.ipynb  .\tmp\oht1parse-%DIRFLAG%.ipynb
        jupyter nbconvert --to html .\tmp\oht1parse-%DIRFLAG%.ipynb --output=..\report\html\oht1parse-%DIRFLAG%.html
    ) else (
        if "%1"=="output" (
            papermill oht2output.ipynb .\tmp\oht2output-%DIRFLAG%.ipynb
            jupyter nbconvert --to html .\tmp\oht2output-%DIRFLAG%.ipynb --output=..\report\html\oht2output-%DIRFLAG%.html
        ) else (
            if "%1"=="graph" (
                papermill oht3graph.ipynb  .\tmp\oht3graph-%DIRFLAG%.ipynb
                jupyter nbconvert --to html .\tmp\oht3graph-%DIRFLAG%.ipynb --output=..\report\html\oht3graph-%DIRFLAG%.html
            ) else (
                if "%1"=="knn" (
                    papermill oht4knn.ipynb  .\tmp\oht4knn-%DIRFLAG%.ipynb
                    jupyter nbconvert --to html .\tmp\oht4knn-%DIRFLAG%.ipynb --output=..\report\html\oht4knn-%DIRFLAG%.html
                ) else (
                    if "%1"=="clear" (
                        jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb
                    ) else (
                        echo "Invalud arg, support parse,output,graph,knn or clear"
                        exit /b 1
                    )
                )
            )
        )
    )
)
time /t

