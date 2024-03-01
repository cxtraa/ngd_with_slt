
@echo off

REM Define the kernel sizes to use for the convolutions
SET kernels_sizes=1 2 3 4
SET out_channels=64

REM Loop over each hidden node value
FOR %%k IN (%kernels_sizes%) DO (
    python train.py --model CM --kernel_size %%k --out_channels %out_channels% --optimiser sgd --lr 1e-03
)