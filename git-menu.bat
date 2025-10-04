@echo off
setlocal enabledelayedexpansion

:: --- helpers ---
for /f "delims=" %%i in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set CURR=%%i
where gh >nul 2>&1 && (set GH_OK=1) || (set GH_OK=0)

:menu
cls
echo =========================================
echo     AlgoTrade - Git Menu  (branch: %CURR%)
echo =========================================
echo [1] Status
echo [2] Pull (current branch)
echo [3] Add + Commit + Push (current branch)
echo [4] Create new branch from main
echo [5] Switch to main + Pull
echo [6] Delete local branch
echo [7] Create Pull Request  (gh) 
echo [8] Merge latest PR      (gh)
echo [9] Open repo in browser (gh)
echo [0] Exit
echo =========================================
if %GH_OK%==0 echo (Note: gh CLI not found - options 7/8/9 will be disabled)
set /p choice=Choose:

if "%choice%"=="1" goto status
if "%choice%"=="2" goto pull
if "%choice%"=="3" goto push
if "%choice%"=="4" goto newbranch
if "%choice%"=="5" goto mainpull
if "%choice%"=="6" goto delbranch
if "%choice%"=="7" goto prcreate
if "%choice%"=="8" goto prmerge
if "%choice%"=="9" goto openrepo
if "%choice%"=="0" goto end
goto menu

:status
git status
echo.
pause
goto menu

:pull
git pull
echo.
pause
goto menu

:push
set /p msg=Commit message:
if "%msg%"=="" set msg=update
git add -A
git commit -m "%msg%" || echo (Nothing to commit)
for /f "delims=" %%i in ('git rev-parse --abbrev-ref HEAD') do set CURR=%%i
git rev-parse --symbolic-full-name --abbrev-ref --quiet @{u} >nul 2>&1 || git push -u origin %CURR%
git push
echo.
pause
goto menu

:newbranch
git checkout main || goto menu
git pull origin main
set /p br=New branch name:
if "%br%"=="" goto menu
git checkout -b %br%
echo.
pause
goto menu

:mainpull
git checkout main
git pull origin main
echo.
pause
goto menu

:delbranch
set /p br=Local branch to delete:
if "%br%"=="" goto menu
git checkout main
git branch -d %br%
echo.
pause
goto menu

:prcreate
if %GH_OK%==0 (
  echo gh CLI not installed. Install from https://cli.github.com/ and run: gh auth login
  pause
  goto menu
)
for /f "delims=" %%i in ('git rev-parse --abbrev-ref HEAD') do set CURR=%%i
set base=main
echo Creating PR: base=%base%  head=%CURR%
set /p title=PR title [UI polish]:
if "%title%"=="" set title=UI polish
set /p body=PR body [auto]:
if "%body%"=="" set body=Sticky sidebar, CSS vars, table UX, a11y, top bar height
gh pr create --base %base% --head %CURR% --title "%title%" --body "%body%"
echo.
pause
goto menu

:prmerge
if %GH_OK%==0 (
  echo gh CLI not installed. Install from https://cli.github.com/ and run: gh auth login
  pause
  goto menu
)
echo Merge method? [1] Merge commit  [2] Squash  [3] Rebase
set /p mm=Select (1/2/3) [2]:
if "%mm%"=="1" set MFLAG=--merge
if "%mm%"=="3" set MFLAG=--rebase
if "%mm%"=="" set MFLAG=--squash
if "%mm%"=="2" set MFLAG=--squash
gh pr merge %MFLAG% --auto
echo.
pause
goto menu

:openrepo
if %GH_OK%==0 (
  echo gh CLI not installed.
  pause
  goto menu
)
gh repo view --web
goto menu

:end
endlocal
