
optimizers=(sgd ngd)

for optim in "${optimizers[@]}"
    do 
        python train.py --model LM --optimiser $optim  --num_workers 12 --batch_size 512 --num_epochs 5 --LMHN 8 --LMHL 2 --lr 1e-3
    done
