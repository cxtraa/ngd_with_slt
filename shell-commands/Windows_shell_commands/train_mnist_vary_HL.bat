
@echo off

REM Define the sets of hidden nodes to iterate over
SET hidden_nodes = 256
SET hidden_layers = 1 2

REM Loop over each hidden node value
FOR %%h IN (%hidden_layers%) DO (
    python train.py --model LM --HN %hidden_nodes% --HL %%h
)