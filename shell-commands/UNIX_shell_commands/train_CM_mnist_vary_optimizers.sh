
optimizers=(sgd ngd)

for optim in "${optimizers[@]}"
    do 
        python train.py --model CM --optimiser $optim  --num_workers 12 --batch_size 128 --num_epochs 5 --CMKS 5 --CMHL 10
    done
