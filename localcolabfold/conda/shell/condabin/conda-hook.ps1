$Env:CONDA_EXE = "/global/home/users/dmstev/mamp_prediction_ml/localcolabfold/conda/bin/conda"
$Env:_CE_M = $null
$Env:_CE_CONDA = $null
$Env:_CONDA_ROOT = "/global/home/users/dmstev/mamp_prediction_ml/localcolabfold/conda"
$Env:_CONDA_EXE = "/global/home/users/dmstev/mamp_prediction_ml/localcolabfold/conda/bin/conda"
$CondaModuleArgs = @{ChangePs1 = $True}
Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs

Remove-Variable CondaModuleArgs