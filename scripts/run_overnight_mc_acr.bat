@echo off
REM ============================================================================
REM Overnight run: MountainCar-v0 then Acrobot-v1
REM ETA: ~3-4h MountainCar + ~4-5h Acrobot = ~7-9h total
REM Resumable: re-run this .bat to pick up after any crash
REM ============================================================================

cd /d "C:\Users\Mao\Desktop\DRL\Final Project"

echo.
echo ============================================================================
echo OVERNIGHT RUN STARTED AT %DATE% %TIME%
echo ============================================================================
echo.

REM --- Phase 1: MountainCar-v0 ---
echo [phase 1/2] Starting MountainCar-v0 at %TIME%
echo.
python scripts\run_full_experiment.py --exp final_mc --env-id MountainCar-v0 --episodes 300 --workers 3
echo.
echo [phase 1/2] MountainCar-v0 finished at %TIME% (exit code %errorlevel%)
echo.

REM --- Phase 2: Acrobot-v1 ---
echo [phase 2/2] Starting Acrobot-v1 at %TIME%
echo.
python scripts\run_full_experiment.py --exp final_acr --env-id Acrobot-v1 --episodes 500 --workers 3
echo.
echo [phase 2/2] Acrobot-v1 finished at %TIME% (exit code %errorlevel%)
echo.

REM --- Final summary ---
echo ============================================================================
echo OVERNIGHT RUN COMPLETED AT %DATE% %TIME%
echo ============================================================================
echo.
echo Next steps:
echo   1. Verify data: tools\compare_conditions.py --exp final_mc --conditions B0-env-native,B1-handcrafted,B2-gemma-oneshot,B3-hermes-full,B3-no-memory,B3-no-AST
echo   2. Verify data: tools\compare_conditions.py --exp final_acr --conditions B0-env-native,B1-handcrafted,B2-gemma-oneshot,B3-hermes-full,B3-no-memory,B3-no-AST
echo.
echo Press any key to close this window...
pause >nul
