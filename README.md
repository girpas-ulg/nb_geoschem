nb_geoschem
===========

Notebooks for various GEOS-Chem pre/post processing.


Notes
-----

For advanced processing tasks, it is recommended to create
several notebooks, each having its own role:

- The notebook with the suffix `_dev` is the development/detailled version
  (it generally includes all source code plus basic testing and plotting)

- The notebooks with the suffix `_run-xxx` should be used to run several
  processing tasks in production (each of these notebooks should be a copy
  for each independent run, with `xxx` replaced by a specific run name).

- Any additional notebooks can be created with suffixes like `_analysis-xxx`,
  `_plot-xxx`, etc... which depend on specific post-processing or visualization
  related to a particular run. 


Additional folders:

- `run_scripts` contains the script version of advanced processing tasks
  (usually called from the `_run` notebooks).
- `data` contains several datasets used in run examples.