
@echo off

REM Define the sets of hidden nodes to iterate over
SET hidden_nodes=2 4 8 16 32 64 128 256
SET hidden_layers=4

REM Loop over each hidden node value
FOR %%h IN (%hidden_nodes%) DO (
    python train.py --model LM --LMHN %%h --LMHL %hidden_layers% --num_workers 12 --optimiser sgd --lr 1e-03
)