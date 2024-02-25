
@echo off

REM Define the sets of hidden nodes to iterate over
<<<<<<< HEAD:experiments/mnist/mnist_vary_HN.bat
SET hidden_nodes=2 4 8 16 32 64 128 256 512 1024 2048
SET hidden_layers=4
=======
SET hidden_nodes=2 4
SET hidden_layers=2
>>>>>>> 12fc82edb519de5277ef6ebed1f66faff7b22875:Windows_shell_commands/mnist_vary_HN.bat

REM Loop over each hidden node value
FOR %%h IN (%hidden_nodes%) DO (
    python train.py --model LM --LMHN %%h --LMHL %hidden_layers% --num_workers 8
)