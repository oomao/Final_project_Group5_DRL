@echo off
REM ============================================================================
REM Overnight run: Acrobot-v1 (6 conditions x 5 seeds)
REM Auto-retry up to 3 times. Orchestrator is resumable.
REM ETA: ~3-4h on first pass; retries only do leftover work.
REM ============================================================================

cd /d "C:\Users\Mao\Desktop\DRL\Final Project"

echo.
echo ============================================================================
echo ACROBOT OVERNIGHT RUN STARTED AT %DATE% %TIME%
echo ============================================================================
echo.

if exist runs\final_acr (
    echo [preflight] Found existing runs\final_acr. Will resume.
) else (
    echo [preflight] Fresh run.
)
echo.

echo ============================================================================
echo ATTEMPT 1/3 STARTED AT %TIME%
echo ============================================================================
python scripts\run_full_experiment.py --exp final_acr --env-id Acrobot-v1 --episodes 500 --workers 3
echo.
echo [attempt 1/3] finished at %TIME% with exit code %errorlevel%
echo.

echo [pause 30s before retry attempt 2]
timeout /t 30 /nobreak >nul

echo ============================================================================
echo ATTEMPT 2/3 (RESUME) STARTED AT %TIME%
echo ============================================================================
python scripts\run_full_experiment.py --exp final_acr --env-id Acrobot-v1 --episodes 500 --workers 3
echo.
echo [attempt 2/3] finished at %TIME% with exit code %errorlevel%
echo.

echo [pause 30s before retry attempt 3]
timeout /t 30 /nobreak >nul

echo ============================================================================
echo ATTEMPT 3/3 (FINAL RESUME) STARTED AT %TIME%
echo ============================================================================
python scripts\run_full_experiment.py --exp final_acr --env-id Acrobot-v1 --episodes 500 --workers 3
echo.
echo [attempt 3/3] finished at %TIME% with exit code %errorlevel%
echo.

echo ============================================================================
echo ACROBOT OVERNIGHT RUN COMPLETED AT %DATE% %TIME%
echo ============================================================================
echo.
echo Morning checklist:
echo   1. Tell Claude that Acrobot finished
echo   2. Or run this to see Table 1 directly:
echo      python tools\compare_conditions.py --exp final_acr --conditions B0-env-native,B1-handcrafted,B2-gemma-oneshot,B3-hermes-full,B3-no-memory,B3-no-AST
echo.
echo Press any key to close this window...
pause >nul
