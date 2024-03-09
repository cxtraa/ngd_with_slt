
@echo off

REM Define the optimisers to produce models over and HN/HL
SET optimisers=sgd;ngd
SET hidden_nodes=1024
SET hidden_layers=2

REM Loop over each hidden node value
FOR %%o IN (%optimisers%) DO (
    python train.py --model LM --lr 9e-03 --num_epochs 10 --optimiser %%o --HN %hidden_nodes% --HL %hidden_layers%
)