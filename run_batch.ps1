# run_batch.ps1 - train + evaluate PPO seeds. Run from D:\edge_drl_scheduler:
#
#   powershell -ExecutionPolicy Bypass -File .\run_batch.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_batch.ps1 -Seeds "4,5"
#   powershell -ExecutionPolicy Bypass -File .\run_batch.ps1 -Setup alltrain -Seeds "1,2,3"
#
# -Setup focused  : trains on workloads normal+variable (both networks)   -> evaluation_final_s<seed>
# -Setup alltrain : trains on normal+variable+heavy+burst (both networks) -> evaluation_alltrain_s<seed>
# -Setup alltrainnonorm : same as alltrain but WITHOUT reward normalization -> evaluation_alltrainnonorm_s<seed>
#
# Full output goes to results\logs\<run>.log, only key lines are printed.
# Finished work is skipped, so re-running after an interruption is safe.
# To use several CPU cores, open one terminal per seed list (e.g. -Seeds "4,5" and -Seeds "6,7,8").

param(
    [string]$Setup = "focused",
    [string]$Seeds = "4,5,6,7,8"
)

$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv python not found - run this from D:\edge_drl_scheduler"; exit 1 }
New-Item -ItemType Directory -Force -Path "results\logs" | Out-Null

if ($Setup -eq "focused") {
    $trainArgs = @("--train-workloads", "normal", "variable"); $evalPrefix = "evaluation_final"
} elseif ($Setup -eq "alltrain") {
    $trainArgs = @("--train-workloads", "normal", "variable", "heavy", "burst"); $evalPrefix = "evaluation_alltrain"
} elseif ($Setup -eq "alltrainnonorm") {
    $trainArgs = @("--train-workloads", "normal", "variable", "heavy", "burst", "--no-reward-norm"); $evalPrefix = "evaluation_alltrainnonorm"
} else {
    Write-Host "-Setup must be 'focused', 'alltrain' or 'alltrainnonorm'"; exit 1
}

function Run-Seed {
    param($run, $seed, $evalOut)

    if (Test-Path "models\ppo_checkpoints\$run\final_model.zip") {
        Write-Host "[skip] training $run (final_model.zip exists)"
    } else {
        Write-Host "[train] $run  seed $seed  $(Get-Date -Format HH:mm:ss)"
        & $py -m training.train_ppo --run-name $run --seed $seed @trainArgs *> "results\logs\$run.log"
        if ($LASTEXITCODE -ne 0) { Write-Host "[FAILED] training $run - see results\logs\$run.log"; return }
        Select-String -Path "results\logs\$run.log" -Pattern "eval @" | Select-Object -Last 2 | ForEach-Object { $_.Line }
    }

    if (Test-Path "results\$evalOut\episodes.csv") {
        Write-Host "[skip] evaluation $evalOut (episodes.csv exists)"
    } else {
        Write-Host "[eval] $evalOut  $(Get-Date -Format HH:mm:ss)"
        & $py -m experiments.evaluate_agents --include-unseen --model "models/ppo_checkpoints/$run/best_model" --seed-base 90000 --episodes-per-condition 20 --out "results/$evalOut" *> "results\logs\$evalOut.log"
        if ($LASTEXITCODE -ne 0) { Write-Host "[FAILED] evaluation $evalOut - see results\logs\$evalOut.log"; return }
        Select-String -Path "results\logs\$evalOut.log" -Pattern "seen: PPO" | ForEach-Object { $_.Line }
    }
}

foreach ($s in ($Seeds -split ",")) {
    $seed = [int]$s
    Run-Seed "${Setup}_s$seed" $seed "${evalPrefix}_s$seed"
}
Write-Host "Done. Summarise with:  .\venv\Scripts\python.exe -m experiments.seed_summary --setups final focusedlast alltrain allload --seeds 1 2 3 4 5 6 7 8"
