
@echo off

REM Define the optimisers to produce models over and HN/HL
SET optimisers=sgd;rmsprop;adam;ngd
SET hidden_nodes=128
SET hidden_layers=2

REM Loop over each hidden node value
FOR %%o IN (%optimisers%) DO (
    IF "%%o"=="sgd" (
        python train.py --model LM --optimiser %%o --lr 1e-03 --LMHN %hidden_nodes% --LMHL %hidden_layers%
    ) ELSE (
        python train.py --model LM --optimiser %%o --LMHN %hidden_nodes% --LMHL %hidden_layers%
    )    
)