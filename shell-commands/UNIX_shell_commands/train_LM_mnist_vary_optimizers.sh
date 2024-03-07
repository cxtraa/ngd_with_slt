optimizers=(sgd ngd)

for optim in "${optimizers[@]}"
do 
    #specify different lr for sgd or ngd
    if [ "$optim" = "sgd" ]; then
        lr=1e-2
    elif [ "$optim" = "ngd" ]; then
        lr=1e-2
    fi

    python train.py --model CM --optimiser $optim --lr $lr --num_workers 32 --batch_size 4086 --num_epochs 30 --CMKS 4 --CMHL 8
done

