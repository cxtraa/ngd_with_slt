optimizers=(sgd ngd)

for optim in "${optimizers[@]}"
do 
    #specify different lr for sgd or ngd
    if [ "$optim" = "sgd" ]; then
        lr=8e-2
    elif [ "$optim" = "ngd" ]; then
        lr=6e-2
    fi

    python train.py --model CM --optimiser $optim --lr $lr --num_workers 12 --batch_size 128 --num_epochs 5 --CMHL 0
done

