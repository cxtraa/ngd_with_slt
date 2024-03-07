# Define new selection criteria
model="CM"
optimisers=("sgd" "ngd")  # Corrected variable name
cmks=2
cmhl=4
num_epochs=30
freq=5

# Convert Bash array to a JSON-like string for optimisers
optimisers_str=$(printf '"%s",' "${optimisers[@]}")
optimisers_str="[${optimisers_str%,}]"  # Remove trailing comma and wrap in brackets

# Define other hyperparameters
num_draws=200
num_chains=1
epsilon=1e-5
gamma=100
#hessian_batch_size=24 remember to add this below too
batch_size=128
num_workers=12

# Call the Python script with the new parameters
python experiments/mnist/eval_mnist_optimisers.py \
    --criteria "{\"model\":\"$model\", \"optimiser\":$optimisers_str, \"CMKS\":$cmks, \"CMHL\":$cmhl, \"num_epochs\":$num_epochs} " \
    --num_draws $num_draws \
    --num_chains $num_chains \
    --epsilon $epsilon \
    --gamma $gamma \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --freq $freq \



