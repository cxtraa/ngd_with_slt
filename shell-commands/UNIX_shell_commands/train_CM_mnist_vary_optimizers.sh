optimizers=(sgd ngd)

for optim in "${optimizers[@]}"
do 
    #specify different lr for sgd or ngd
    if [ "$optim" = "sgd" ]; then
        lr=1e-3
    elif [ "$optim" = "ngd" ]; then
        lr=1e-2
    fi

    python train.py --model CM --optimiser $optim --lr $lr --num_workers 12 --batch_size 1024 --num_epochs 5 --CMKS 5 --CMHL 10
done
