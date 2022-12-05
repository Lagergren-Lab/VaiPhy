# VaiPhy

### Simulate the synthetic datasets:
```
python simulate_datasets.py
```

### Run VaiPhy with default parameters: 
```
python main.py --dataset data_seed_1_taxa_10_pos_300
```

### Prepare contents for post-sampling: 
```
python post_analysis_prep.py --dataset data_seed_1_taxa_10_pos_300
```

### Run post-sampling CSMC:
```
python post_sampling/csmc.py --dataset data_seed_1_taxa_10_pos_300
```
or
```
python post_sampling/csmc.py --dataset data_seed_1_taxa_10_pos_300 --csmc_n_particles 2048 --csmc_distortion_poisson_rate 4
```
