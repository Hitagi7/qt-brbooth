@echo off
echo Testing compilation...
echo.

REM Try to find Qt installation
set QT_DIR=C:\Qt\6.9.1\msvc2022_64
if exist "%QT_DIR%\bin\qmake.exe" (
    echo Found Qt at %QT_DIR%
    set PATH=%QT_DIR%\bin;%PATH%
    qmake qt-brbooth.pro
    if exist Makefile (
        echo Makefile generated successfully
        nmake
    ) else (
        echo Failed to generate Makefile
    )
) else (
    echo Qt not found at %QT_DIR%
    echo Please install Qt or set correct path
)

pause 