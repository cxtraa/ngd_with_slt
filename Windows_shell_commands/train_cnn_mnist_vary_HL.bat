
@echo off

REM Define the kernel sizes to use for the convolutions
SET hidden_conv_layers=1 2 3 4

REM Loop over each hidden node value
FOR %%h IN (%hidden_conv_layers%) DO (
    python train.py --model CM --CMHL %%h --optimiser sgd --lr 1e-03
)