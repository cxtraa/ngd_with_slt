hidden_nodes=16
hidden_layers=(1 2)

for HL in "${hidden_layers[@]}"
    do 
        python train.py --model LM --HN $hidden_nodes --HL $HL
        #echo $HL
    done