@echo off
REM Setzt das Netzabdeckungs-Training vom gesicherten Stand fort (netcov_resume.pt).
REM Doppelklick zum manuellen Start, oder Claude startet es in einer neuen Session.
cd /d "%~dp0"
echo Starte Dashboard (http://localhost:7070) ...
start "" ".venv\Scripts\python.exe" dashboard.py
echo Setze Training fort von checkpoints\netcov_resume.pt ...
".venv\Scripts\python.exe" -u train_netcov.py --updates 4000 --envs 1024 --start-stage 1 --name netcov_run16 --resume "checkpoints\netcov_resume.pt" > "out\netcov_run16.log" 2>&1
echo Fertig. Log: out\netcov_run16.log
pause
