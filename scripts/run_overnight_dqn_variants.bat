@echo off
REM ============================================================================
REM Part 2 overnight: DQN-variant supplementary study
REM   2 NEW variants (Double DQN, Dueling DQN)
REM   x 2 conditions (B0-env-native, B3-hermes-full)
REM   x 4 envs (LunarLander-v3, CartPole-v1, MountainCar-v0, Acrobot-v1)
REM   x n=5 seeds
REM   = 80 new runs
REM
REM Vanilla DQN data already exists in runs/final*/ from Part 1.
REM Output layout: runs/part2_<variant>_<env>/<cond>/seed_NN/
REM
REM Each (variant, env) chunk gets 3 retry attempts (orchestrator skips done).
REM ETA: ~7-10 hours with workers=5.
REM ============================================================================

cd /d "C:\Users\Mao\Desktop\DRL\Final Project"

echo.
echo ============================================================================
echo PART 2 OVERNIGHT STARTED AT %DATE% %TIME%
echo ============================================================================

REM -------------------------------------------------------------------------
REM Helper: each block runs the same orchestrator command 3 times.
REM Orchestrator detects completed (cond, seed) pairs and skips them, so
REM the 2nd/3rd attempts only cover whatever the 1st missed.
REM -------------------------------------------------------------------------

REM Order: cheap envs first (CartPole, MountainCar, Acrobot) so partial
REM completion gives broad coverage; expensive LunarLander last per variant.

REM ============================================================================
REM VARIANT 1/2: Double DQN
REM ============================================================================

echo.
echo === Double DQN x CartPole-v1 ===
for /l %%i in (1,1,3) do (
    echo --- attempt %%i ---
    python scripts\run_full_experiment.py --exp part2_double_cp --env-id CartPole-v1 --dqn-variant double --conditions B0-env-native,B3-hermes-full --episodes 500 --workers 5
)

echo.
echo === Double DQN x MountainCar-v0 ===
for /l %%i in (1,1,3) do (
    echo --- attempt %%i ---
    python scripts\run_full_experiment.py --exp part2_double_mc --env-id MountainCar-v0 --dqn-variant double --conditions B0-env-native,B3-hermes-full --episodes 300 --workers 5
)

echo.
echo === Double DQN x Acrobot-v1 ===
for /l %%i in (1,1,3) do (
    echo --- attempt %%i ---
    python scripts\run_full_experiment.py --exp part2_double_acr --env-id Acrobot-v1 --dqn-variant double --conditions B0-env-native,B3-hermes-full --episodes 500 --workers 5
)

echo.
echo === Double DQN x LunarLander-v3 ===
for /l %%i in (1,1,3) do (
    echo --- attempt %%i ---
    python scripts\run_full_experiment.py --exp part2_double_ll --env-id LunarLander-v3 --dqn-variant double --conditions B0-env-native,B3-hermes-full --episodes 1500 --workers 5
)

REM ============================================================================
REM VARIANT 2/2: Dueling DQN
REM ============================================================================

echo.
echo === Dueling DQN x CartPole-v1 ===
for /l %%i in (1,1,3) do (
    echo --- attempt %%i ---
    python scripts\run_full_experiment.py --exp part2_dueling_cp --env-id CartPole-v1 --dqn-variant dueling --conditions B0-env-native,B3-hermes-full --episodes 500 --workers 5
)

echo.
echo === Dueling DQN x MountainCar-v0 ===
for /l %%i in (1,1,3) do (
    echo --- attempt %%i ---
    python scripts\run_full_experiment.py --exp part2_dueling_mc --env-id MountainCar-v0 --dqn-variant dueling --conditions B0-env-native,B3-hermes-full --episodes 300 --workers 5
)

echo.
echo === Dueling DQN x Acrobot-v1 ===
for /l %%i in (1,1,3) do (
    echo --- attempt %%i ---
    python scripts\run_full_experiment.py --exp part2_dueling_acr --env-id Acrobot-v1 --dqn-variant dueling --conditions B0-env-native,B3-hermes-full --episodes 500 --workers 5
)

echo.
echo === Dueling DQN x LunarLander-v3 ===
for /l %%i in (1,1,3) do (
    echo --- attempt %%i ---
    python scripts\run_full_experiment.py --exp part2_dueling_ll --env-id LunarLander-v3 --dqn-variant dueling --conditions B0-env-native,B3-hermes-full --episodes 1500 --workers 5
)

echo.
echo ============================================================================
echo PART 2 OVERNIGHT COMPLETED AT %DATE% %TIME%
echo ============================================================================
echo.
echo Morning checklist:
echo   1. Tell Claude that Part 2 finished
echo   2. Or inspect data manually:
echo      dir runs\part2_*
echo.
echo Press any key to close this window...
pause >nul
