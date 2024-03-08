optimizers=(sgd ngd)

for optim in "${optimizers[@]}"
do 
    #specify different lr for sgd or ngd
    if [ "$optim" = "sgd" ]; then
        lr=5e-2
    elif [ "$optim" = "ngd" ]; then
        lr=1e-1
    fi

    python train.py --model CM --optimiser $optim --lr $lr --num_workers 12 --batch_size 128 --num_epochs 30 --CMHL 0
done

