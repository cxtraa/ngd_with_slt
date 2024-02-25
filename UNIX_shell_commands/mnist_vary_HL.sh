hidden_nodes=(2 4 8 )
hidden_layers=2

for HN in "${hidden_nodes[@]}"
    do 
        python train.py --model LM --LMHN $HN --LMHL $hidden_layers
        #echo $HL
    done