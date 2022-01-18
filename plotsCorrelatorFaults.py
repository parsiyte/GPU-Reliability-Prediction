import pandas as pd
import numpy as np
import csv

from pathlib import Path  # path
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplotlib.patches as mpatches
import getopt, sys # for taking argument from the command line 


##############################
###### FAULT RATES PLOT ######
##############################

def FAULT_RATES_PLOT(data):

    applications = list(data['Applications'])
    nsight_masked = data['Masked'].values
    nsight_crash = data['Crash'].values
    nsight_sdc = data['SDC'].values

    np.arange(0,25,1)

    tot = [i+j+k for i,j,k in zip(data['SDC'], data['Crash'], data['Masked'])]
    sdc_bar = [i / j * 100 for i,j in zip(data['SDC'], tot)]
    crash_bar = [i / j * 100 for i,j in zip(data['Crash'], tot)]
    masked_bar = [i / j * 100 for i,j in zip(data['Masked'], tot)]

    barWidth = 0.85

    # SDC faults
    plt.bar(np.arange(0,25,1), sdc_bar, color='#b5ffb9', edgecolor='white', width=barWidth, label="SDCs")
    # Crash faults
    plt.bar(np.arange(0,25,1), crash_bar, bottom=sdc_bar, color='#f9bc86', edgecolor='white', width=barWidth, label="Crashes")
    # Masked faults
    plt.bar(np.arange(0,25,1), masked_bar, bottom=[i+j for i,j in zip(sdc_bar, crash_bar)], color='#a3acff', edgecolor='white', width=barWidth, label="Masked Faults")
     
    # X axis details, 
    plt.xticks(np.arange(0,25,1), applications, rotation = 75, fontsize=15)
    plt.yticks(fontsize=15)
     
    # Legends
    plt.legend(loc='upper right', fontsize=13)
    plt.show()

##########################################################
########## Correlation Results Heat Map Plots ############
##########################################################

def Corr_Results_Heat_Map(sp_corr_GPUsim, pr_corr_GPUsim, sp_corr_Nsight, pr_corr_Nsight):
    x_GPUsim = ["Load_instruction", "Store_instruction", "Param_Mem_instruction", "Total_instruction", "IPC", "Sim_Rate",
                "Global_Mem_Read", "Global_Mem_Write", "BW_Utilization", "Warp_Occupancy", "Control_Flow_Inst_Intensity", "Data_Mov_Inst_Intensity",
                "Float_Point_Inst_Intensity", "Integer_Arithmetic_Inst_Intensity", "Logical_Inst_Intensity", "Load_Inst_Intensity", 
                "Predicate_Inst_Intensity", "SDC", "Crash", "Masked"]

    x_Nsight = ["SOL_SM", "SOL_MEM", "SOL_L1_Tex_Cache", "SOL_L2_Cache", "SOL_DRAM", "Duration", "Elapsed_Cycle", "IPC", 
                "Mem_Throughput", "L2_Hit_Rate", "Mem_Busy", "Max_Band", "Active_Warp_Per_Sch", "Warp_Cyc_Inst", 
                "Executed_Inst", "Reg_Per_Thread", "Achieved_Occupancy", "Achieved_Active_Warp", "SDC", "Crash", "Masked"]


    for i in range(0, len(x_GPUsim)):
        for j in range (i, len(x_GPUsim)):
            sp_corr_GPUsim[x_GPUsim[i]][x_GPUsim[j]] = pr_corr_GPUsim[x_GPUsim[i]][x_GPUsim[j]]

    for i in range (0, len(x_Nsight)):
        for j in  range (i, len(x_Nsight)):
            sp_corr_Nsight[x_Nsight[i]][x_Nsight[j]] = pr_corr_Nsight[x_Nsight[i]][x_Nsight[j]]

    _, ax3 = plt.subplots(figsize=(11, 15))
    heatmap = sns.heatmap(sp_corr_GPUsim,
                          square = True,
                          linewidths = .1,
                          cmap = 'coolwarm',
                          cbar_kws = {'shrink': .4,
                                    'ticks' : [-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]},
                          vmin = -1,
                          vmax = 1,
                          annot = True,
                          annot_kws = {'size': 9})

    ax3.set_yticklabels(sp_corr_GPUsim.columns, rotation = 0, fontsize= 15)
    ax3.set_xticklabels(sp_corr_GPUsim.columns, rotation = 80, fontsize= 15)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    heatmap.get_figure().savefig('Corr_GPGPU_sim.png', bbox_inches='tight')

    _, ax4 = plt.subplots(figsize=(11, 15))

    heatmap = sns.heatmap(sp_corr_Nsight,
                          square = True,
                          linewidths = .1,
                          cmap = 'coolwarm',
                          cbar_kws = {'shrink': .4,
                                    'ticks' : [-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]},
                          vmin = -1,
                          vmax = 1,
                          annot = True,
                          annot_kws = {'size': 9})

    ax4.set_yticklabels(sp_corr_Nsight.columns, rotation = 0, fontsize = 15)
    ax4.set_xticklabels(sp_corr_Nsight.columns, rotation = 80, fontsize = 15)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    heatmap.get_figure().savefig('Corr_Nsight.png', bbox_inches='tight')


##s##################################################################
########## Correlation Results Between Faults and Features #########
####################################################################

def CorrRes_BetweenFeaturesAndFaults(sp_corr_GPUsim, pr_corr_GPUsim, sp_corr_Nsight, pr_corr_Nsight):
    x_axis_GPGPUSim = ['Load_instruction', 'Store_instruction', 'Param_Mem_instruction', 'Total_instruction', 'IPC', 'Sim_Rate',
                'Global_Mem_Read', 'Global_Mem_Write', 'BW_Utilization', 'Warp_Occupancy', 'Control_Flow_Inst_Intensity', 'Data_Mov_Inst_Intensity',
                'Float_Point_Inst_Intensity', 'Integer_Arithmetic_Inst_Intensity', 'Logical_Inst_Intensity',  'Load_Inst_Intensity', 
                'Predicate_Inst_Intensity']

    x_axis_Nsight = ["SOL_SM", "SOL_MEM", "SOL_L1_Tex_Cache", "SOL_L2_Cache", "SOL_DRAM", "Duration", "Elapsed_Cycle", "IPC", 
                     "Mem_Throughput", "L2_Hit_Rate", "Mem_Busy", "Max_Band", "Active_Warp_Per_Sch", "Warp_Cyc_Inst", 
                     "Executed_Inst", "Reg_Per_Thread", "Achieved_Occupancy", "Achieved_Active_Warp"]

    SpearmanCorr_GPGPU_sim = [sp_corr_GPUsim['Masked'].values[0:17], sp_corr_GPUsim['Crash'].values[0:17], sp_corr_GPUsim['SDC'].values[0:17]]
    PearsonCorr_GPGPU_sim = [pr_corr_GPUsim['Masked'].values[0:17], pr_corr_GPUsim['Crash'].values[0:17], pr_corr_GPUsim['SDC'].values[0:17]]

    SpearmanCorr_Nsight = [sp_corr_Nsight['Masked'].values[0:18], sp_corr_Nsight['Crash'].values[0:18], sp_corr_Nsight['SDC'].values[0:18]]
    PearsonCorr_Nsight = [pr_corr_Nsight['Masked'].values[0:18], pr_corr_Nsight['Crash'].values[0:18], pr_corr_Nsight['SDC'].values[0:18]]

    fig_GPGPUSim, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(x_axis_GPGPUSim, SpearmanCorr_GPGPU_sim[0], label="Maked")
    ax1.plot(x_axis_GPGPUSim, SpearmanCorr_GPGPU_sim[1], label="Crash")
    ax1.plot(x_axis_GPGPUSim, SpearmanCorr_GPGPU_sim[2], label="SDC")
    ax1.set_xticklabels(x_axis_GPGPUSim, rotation=60)
    ax1.set_ylabel('Spearman Correlation Results',fontsize=15)
    ax1.tick_params(axis='x', which='major', labelsize=15)
    ax1.legend(loc='upper right')

    ax2.plot(x_axis_GPGPUSim, PearsonCorr_GPGPU_sim[0], label="Maked")
    ax2.plot(x_axis_GPGPUSim, PearsonCorr_GPGPU_sim[1], label="Crash")
    ax2.plot(x_axis_GPGPUSim, PearsonCorr_GPGPU_sim[2], label="SDC")
    ax2.set_xticklabels(x_axis_GPGPUSim, rotation=60)
    ax2.set_ylabel('Pearson Correlation Results',fontsize=15)
    ax2.tick_params(axis='x', which='major', labelsize=15)
    ax2.legend(loc='upper right')
    #fig_GPGPUSim.savefig('GPGPUSim_CorrResults_WithFaults.png')

    fig_Nsight, (ax3, ax4) = plt.subplots(2, sharex=True)
    ax3.plot(x_axis_Nsight, SpearmanCorr_Nsight[0], label="Maked")
    ax3.plot(x_axis_Nsight, SpearmanCorr_Nsight[1], label="Crash")
    ax3.plot(x_axis_Nsight, SpearmanCorr_Nsight[2], label="SDC")
    ax3.set_ylabel('Spearman Correlation Results',fontsize=15)
    ax3.tick_params(axis='x', which='major', labelsize=15)
    ax3.set_xticklabels(x_axis_Nsight, rotation=75)
    ax3.legend(loc='upper right')

    ax4.plot(x_axis_Nsight, PearsonCorr_Nsight[0], label="Maked")
    ax4.plot(x_axis_Nsight, PearsonCorr_Nsight[1], label="Crash")
    ax4.plot(x_axis_Nsight, PearsonCorr_Nsight[2], label="SDC")
    ax4.set_ylabel('Pearson Correlation Results',fontsize=15)
    ax4.tick_params(axis='x', which='major', labelsize=15)
    ax4.set_xticklabels(x_axis_Nsight, rotation=60)
    ax4.legend(loc='upper right')
    #fig_Nsight.savefig('Nsight_CorrResults_WithFaults.png')
    plt.show()


metrics_GPUsim = pd.read_excel('data_metrics_GPUsim.xls')
data_GPUsim = pd.DataFrame(metrics_GPUsim,
                    columns=['Applications', 'Load_instruction', 'Store_instruction', 'Param_Mem_instruction', 'Total_instruction', 'IPC', 'Sim_Rate',
                    'Global_Mem_Read', 'Global_Mem_Write', 'BW_Utilization', 'Warp_Occupancy', 'Control_Flow_Inst_Intensity', 'Data_Mov_Inst_Intensity',
                    'Float_Point_Inst_Intensity',	'Integer_Arithmetic_Inst_Intensity', 'Logical_Inst_Intensity',	'Load_Inst_Intensity', 
                    'Predicate_Inst_Intensity', 'SDC', 'Crash', 'Masked'])

sp_corr_GPUsim = data_GPUsim.corr(method = 'spearman')
pr_corr_GPUsim = data_GPUsim.corr(method = 'pearson')

metrics_Nsight = pd.read_excel('data_metrics_NSC.xls')
data_Nsight = pd.DataFrame(metrics_Nsight,
          columns=["Applications", "SOL_SM", "SOL_MEM", "SOL_L1_Tex_Cache", "SOL_L2_Cache", "SOL_DRAM", "Duration", "Elapsed_Cycle", "IPC", 
                   "Mem_Throughput", "L2_Hit_Rate", "Mem_Busy", "Max_Band", "Active_Warp_Per_Sch", "Warp_Cyc_Inst", 
                   "Executed_Inst", "Reg_Per_Thread", "Achieved_Occupancy", "Achieved_Active_Warp", "SDC", "Crash", "Masked"])

sp_corr_Nsight = data_Nsight.corr(method = 'spearman')
pr_corr_Nsight = data_Nsight.corr(method = 'pearson')

argumentList = sys.argv[1:]

if argumentList[0] == '--fault_rates_plot':
    FAULT_RATES_PLOT(data_Nsight)

elif argumentList[0] == '--corr_results_heat_map':
    Corr_Results_Heat_Map(sp_corr_GPUsim, pr_corr_GPUsim, sp_corr_Nsight, pr_corr_Nsight)

elif argumentList[0] == '--corr_results_features_and_faults':
    CorrRes_BetweenFeaturesAndFaults(sp_corr_GPUsim, pr_corr_GPUsim, sp_corr_Nsight, pr_corr_Nsight)


