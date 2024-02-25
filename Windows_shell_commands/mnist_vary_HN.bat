
@echo off

REM Define the sets of hidden nodes to iterate over
SET hidden_nodes=2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
SET hidden_layers=2

REM Loop over each hidden node value
FOR %%h IN (%hidden_nodes%) DO (
    python train.py --model LM --LMHN %%h --LMHL %hidden_layers%
)