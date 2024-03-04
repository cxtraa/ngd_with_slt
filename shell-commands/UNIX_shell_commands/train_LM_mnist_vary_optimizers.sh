optimizers=(sgd ngd)

for optim in "${optimizers[@]}"
do 
    #specify different lr for sgd or ngd
    if [ "$optim" = "sgd" ]; then
        lr=1e-2
    elif [ "$optim" = "ngd" ]; then
        lr=1e-2
    fi

    python train.py --model LM --optimiser $optim --lr $lr --num_workers 12 --batch_size 128 --num_epochs 20 --LMHN 8 --LMHL 12
done

