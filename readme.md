# FaultPredictionOnGPGPUs

This repository includes the experiment materials and results for the soft error vulnerability study for GPGPU applications. 
 - **BenchmarkGPGPUApplications** folder includes the benchmark applications used in the experiments. These applications belong to the PolyBench benchmark suite.
 - **gpgpu-sim** folder includes the simulation results of benchmark applications.
 - **with_GPGPUsim** and **with-Nsight** folders include prediction results for the fault rates obtained with GPGPU-Sim 4.0 metrics and Nsight Compute Tool metrics, respectively. 
 - **plots** folder includes resultant plots for correlation results among features, between features and fault rates and prediction results.
 - **data_metrics_GPUsim.xls** and **data_metrics_NSC.xls** files include profiling metrics obtained from GPGPU-Sim and Nsight Compute tool.

**Required Python Libraries:** One can use the last versions of each library.
 - pandas
 - numpy
 - csv
 - sklearn
 - xlsxwriter  
 - openpyxl  
 - matplotlib
 - seaborn

**To run the experiments:**
**--**$ python3 plotsCorrelatorFaults.py --arg
The arg can be:
 1) --fault_rates_plot -> this will plot fault rates for each of the fault type
 2) --corr_results_heat_map -> this will plot the correlation results (Spearman and Pearson) among the features and faults 
 3) --corr_results_features_and_faults -> this plot the same correlation results between faults and features.

**$** python3 plotsCorrelatorFaults.py --arg1 --arg2 --arg3
For the classification experiments: 
 - **--arg1:**
 1) --gpgpu-sim -> prediction experiments obtained with gpgpu-sim metrics
 2) --nsight-compute -> prediction experiments obtained with nsight-compute metrics
 - **--arg2:**
 1) --crash -> crash classification results
 2) --sdc -> sdc classification results
 - **--arg3:**
 1) --all_features -> prediction study with all features
 2) --sel_features -> prediction study with selected features

**$** python3 plotsCorrelatorFaults.py --arg1 --arg2
For the regression experiments experiments: 
 - **--arg1:** 
 1) --gpgpu-sim -> prediction experiments obtained with gpgpu-sim metrics
 2) --nsight-compute -> prediction experiments obtained with nsight compute tool metrics
 - **--arg2:** 
 1) --masked -> prediction experiment for masked faults
 2) --others -> prediction experiment for SDCs and crashes


There are three types of faults examined in this study:
  1 - Crash Faults
  2 - Silent Data Corruptions (SDCs)
  3 - Masked Faults
  
To predict the crash and masked fault rates, different classification methods is used, while regression method is used to predict masked fault rates. 
Further details for the approaches used in the prediction studies, collected metrics, and feature selection stages are examined in our paper published soon. 



