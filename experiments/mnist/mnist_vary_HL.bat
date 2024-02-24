
@echo off

REM Define the sets of hidden nodes to iterate over
SET hidden_nodes = 256
SET hidden_layers = 1 2

REM Loop over each hidden node value
FOR %%h IN (%hidden_layers%) DO (
    python mnist_vary_architecture.py --model LM --LMHN %hidden_nodes% --LMHL %%h
)