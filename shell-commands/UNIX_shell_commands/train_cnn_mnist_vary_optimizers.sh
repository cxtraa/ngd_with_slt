
optimizers=(sgd ngd)

for optim in "${optimizers[@]}"
    do 
        python train.py --model CM --optimiser $optim  --num_workers 12 --batch_size 1024 --num_epochs 100
    done
