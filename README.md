# MetModeling_Solfisburg_EAC_2024

Info and code to reproduce the metabolic modeling analysis and Figure 5 of [Solfisburg et al. 2024](https://doi.org/10.1158/1055-9965.EPI-23-0652).

Microbiome metabolic modeling simulations were performed using the mgPipe module of the Microbiome Modeling Toolbox (COBRA toolbox commit: 71c117305231f77a0292856e292b95ab32040711) and the AGORA metabolic models (AGORA 1.02). All computations were performed in MATLAB version 2019a (Mathworks, Inc.), using the IBM CPLEX (IBM, Inc.) solver.

The script *StartMgPipe.m* was used to start the pipeline for microbiome metabolic models creation and simulation (mgPipe) including the reiqured information.

The script *analysisPredictions.py* was used to parse simulation results, conduct statistical analysis, and to produce all the plots (PCoA, Volcano plot, boxplots) present in Figure 5.

