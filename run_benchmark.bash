device=cuda:1
for env in AssaultNoFrameskip-v4 SeaquestNoFrameskip-v4  # AsterixNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 BeamRiderNoFrameskip-v4 
    do 
    for seed in 0 1 2
        do
    tsp python train.py --config config/cvi.yaml --env_id $env --device $device --seed $seed
    done
done
