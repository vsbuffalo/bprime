## SLiM Simulations

Contains JSON configuration files for each SLiM run, the SLiM scripts, and
Snakefiles to run the simulations described in the JSON files. These are the
main types of sims:

 - bgs: BGS forward simulations for B score evaluations, etc.
 - training: `segment.slim` for training the DNN

 This also contains shared code Eidos module of commonly used functions across
 scripts, `utils.slim`.
