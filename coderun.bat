

:: run jupyer notebooks 
::
:: DIRFLAG={small | large}  used to control DIRRAW,DIROUT location. default=small
:: Please set DIRFLAG=large at command prompt to use large dataset
:: set DIRFLAG=large
:: Please unset DIRFLAG at command prompt to not use large dataset
:: set DIRFLAG

echo Current DIRFLAG variable value %DIRFLAG%

@echo off
:: Prepare filename part
set HOUR=%TIME:~0,2%
set MINUTE=%TIME:~3,2%
set SECOND=%TIME:~6,2%
:: Remove leading spaces from HOUR
if "%HOUR:~0,1%" == " " set HOUR=0%HOUR:~1,1%
set NAMEPART=%HOUR%%MINUTE%%SECOND%

:: Create subdirectory ./tmp
if not exist ".\tmp" mkdir ".\tmp"

time /t
if "%1"=="" (
    :: jupyter nbconvert --to notebook --execute oht1parse.ipynb  --output .\tmp\oht1parse-%NAMEPART%.ipynb
    papermill oht1parse.ipynb  .\tmp\oht1parse-%NAMEPART%.ipynb
    
    time /t
    papermill oht2output.ipynb .\tmp\oht2output-%NAMEPART%.ipynb

    time /t
    papermill oht3graph.ipynb  .\tmp\oht3graph-%NAMEPART%.ipynb
    jupyter nbconvert --to html .\tmp\oht3graph-%NAMEPART%.ipynb
) else (
    if "%1"=="parse" (
        papermill oht1parse.ipynb  .\tmp\oht1parse-%NAMEPART%.ipynb
    ) else (
        if "%1"=="output" (
            papermill oht2output.ipynb .\tmp\oht2output-%NAMEPART%.ipynb
        ) else (
            if "%1"=="graph" (
                papermill oht3graph.ipynb  .\tmp\oht3graph-%NAMEPART%.ipynb
                jupyter nbconvert --to html .\tmp\oht3graph-%NAMEPART%.ipynb
            ) else (
                if "%1"=="clear" (
                    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb
                ) else (
                    echo "Invalud arg, support parse,output or graph"
                    exit /b 1
                )
            )
        )
    )
)
time /t

