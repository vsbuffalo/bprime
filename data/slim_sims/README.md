# SLiM Simulation Results and Associated Downstream Data and Trained Models

This directory contains for each model run the simulation results:

  -  `<model>/sims/`: simulation result tree sequences, organized in directories based on seed 
  -  `<model>/<model>.npz`: Numpy `.npz` file of processed tree sequences.
  -  `<model>/<model>_data.pkl`: Pickled `LearnedFunction:` object, not yet trained.
  -  `<model>/fits/<model_arch>.pkl` and `<model>/fits/<model_arch>.h5`: Pickled `LearnedFunction` object with HDF5-serialized TensorFlow model.
