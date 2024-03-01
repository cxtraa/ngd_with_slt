
optimizers=(sgd ngd)

for optim in "${optimizers[@]}"
    do 
        python train.py --model LM --optimiser $optim  --num_workers 12 --batch_size 128 --num_epochs 5 --LMHN 8 --LMHL 2
    done
