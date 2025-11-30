# AGILE  
**AGILE – Open Global Glacier Data Assimilation Framework**

This project is an adaptation and extension of [OGGM](https://github.com/OGGM/oggm). It uses OGGM’s dynamical glacier model together with the backward-mode automatic differentiation features of [PyTorch](https://pytorch.org/). This makes it possible to run cost-function–based data assimilation for glacier evolution.

A preprint describing AGILE v0.1 is available [here](https://doi.org/10.5194/egusphere-2025-3401).

---

## Reproducing the results of the [preprint](https://doi.org/10.5194/egusphere-2025-3401)

If you want to reproduce the experiments shown in the preprint, follow these steps:

1. **Download the example experiment script** (Aletsch retreat case):  
   [run_example_experiment.sh](https://raw.githubusercontent.com/OGGM/AGILE/refs/heads/master/agile1d/sandbox/paper_v01_code/minimal_run_example/run_example_experiment.sh)

2. **Make the script executable**:  
       chmod +x run_example_experiment.sh

3. **Run the script**:  
       ./run_example_experiment.sh

   Before running the script, make sure you have **[git](https://git-scm.com/)** and **[docker](https://www.docker.com/)** installed.  
   The script will:
   - clone the needed repositories,  
   - download all required data (~2 GB),  
   - and run the full experiment.

4. **Running other experiments**  
   If you want to run more experiments, you can change the file  
   `mini_experiment_file_fg_oggm.py`  
   located in (after the first execution of the script):  
   `agile_workdir/AGILE/agile1d/sandbox/paper_v01_code/minimal_run_example/`

   For running **all** experiments from the publication, you can use the example settings here:  
   <https://github.com/OGGM/AGILE/tree/master/agile1d/sandbox/paper_v01_code/run_scripts>

5. **Create example plots**  
   After running the experiment(s), you can create an example plot by running:  
   [create_example_plot.sh](https://raw.githubusercontent.com/OGGM/AGILE/refs/heads/master/agile1d/sandbox/paper_v01_code/minimal_run_example/create_example_plot.sh)

   All plotting scripts used for the figures in the publication are available here:  
   <https://github.com/OGGM/AGILE/tree/master/agile1d/sandbox/paper_v01_code/plotting_scripts>  
   Note: You need to run **all** experiments if you want to recreate **all** figures from the publication.

---
## Previous work

**agile2D** (previously *combine2d*) is based on a 2D Shallow-Ice-Approximation model. It uses glacier outlines, surface topography, surface mass-balance time series, and optional ice-thickness measurements for ice caps.  
More information:  
- Master thesis by @phigre: <https://diglib.uibk.ac.at/ulbtirolhs/content/titleinfo/3086935/full.pdf>  
- Repository state: <https://github.com/OGGM/agile/tree/04aa57353f72f272a264be5a4c683ffa7dc5bf0f>

**agile1D** (previously *combine1d*) is based on a 1D flowline model. It uses flowline surface heights, widths, and surface mass-balance time series for valley glaciers.  
More information:  
- Master thesis by @pat-schmitt: <https://diglib.uibk.ac.at/ulbtirolhs/content/titleinfo/6139027/full.pdf>  
- Repository state: <https://github.com/OGGM/agile/tree/bf2f7372787adf3e4f31ba5fd56e8968b9fb3347>


