
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

best_results_df = pd.read_csv('./results_data/best_results.csv', sep=',')
early_fusion_results_df = pd.read_csv('./results_data/early_fusion_results.csv', sep=',')
early_fusion_with_issc_results_df = pd.read_csv('./results_data/early_fusion_with_issc_results.csv', sep=',')
hs_issc_results_df = pd.read_csv('./results_data/hs_issc_results.csv', sep=',')
hs_manual_results_df = pd.read_csv('./results_data/hs_manual_results.csv', sep=',')
hs_pca_results_df = pd.read_csv('./results_data/hs_pca_results.csv', sep=',')
late_fusion_hs_issc_results_df = pd.read_csv('./results_data/late_fusion_hs_issc_results.csv', sep=',')
single_modality_results_df = pd.read_csv('./results_data/single_modality_results.csv', sep=',')
three_branch_combo_fusion_results_df = pd.read_csv('./results_data/three_branch_combo_fusion_results.csv', sep=',')
three_branch_late_fusion_results_df = pd.read_csv('./results_data/three_branch_late_fusion_results.csv', sep=',')
two_branch_combo_fusion_results_df = pd.read_csv('./results_data/two_branch_combo_fusion_results.csv', sep=',')
two_branch_late_fusion_results_df = pd.read_csv('./results_data/two_branch_late_fusion_results.csv', sep=',')

# plt.rc(usetex=True)
# plt.rc('pgf', texsystem='pdflatex')

plt.figure(figsize=(12,9))
plt.plot(best_results_df['Experiment'], best_results_df['OA'], label='OA', marker='o')
plt.plot(best_results_df['Experiment'], best_results_df['AA'], label='AA', marker='o')
plt.plot(best_results_df['Experiment'], best_results_df['Kappa'], label='Kappa', marker='o')
plt.xticks(rotation=90)
plt.xlabel('Experiment')
plt.ylabel('Accuracy & Kappa Score')
plt.legend()
plt.tight_layout()
plt.savefig('./results_data/best_results_plot.png', bbox_inches='tight')

# Clear plot data for next plot
plt.clf()
plt.close()



plt.figure(figsize=(12,9))
plt.plot(early_fusion_results_df['Modality'], early_fusion_results_df['OA'], label='OA', marker='o')
plt.plot(early_fusion_results_df['Modality'], early_fusion_results_df['AA'], label='AA', marker='o')
plt.plot(early_fusion_results_df['Modality'], early_fusion_results_df['Kappa'], label='Kappa', marker='o')
plt.xticks(rotation=90)
plt.xlabel('Modality')
plt.ylabel('Accuracy & Kappa Score')
plt.legend()
plt.tight_layout()
plt.savefig('./results_data/early_fusion_results_plot.png', bbox_inches='tight')

# Clear plot data for next plot
plt.clf()
plt.close()

plt.figure(figsize=(12,9))
plt.plot(early_fusion_with_issc_results_df['Number of Selected Bands'], early_fusion_with_issc_results_df['OA'], label='OA', marker='o')
plt.plot(early_fusion_with_issc_results_df['Number of Selected Bands'], early_fusion_with_issc_results_df['AA'], label='AA', marker='o')
plt.plot(early_fusion_with_issc_results_df['Number of Selected Bands'], early_fusion_with_issc_results_df['Kappa'], label='Kappa', marker='o')
plt.xticks(rotation=90)
plt.xlabel('Number of Selected Bands')
plt.ylabel('Accuracy & Kappa Score')
plt.legend()
plt.tight_layout()
plt.savefig('./results_data/early_fusion_with_issc_results_plot.png', bbox_inches='tight')

# Clear plot data for next plot
plt.clf()
plt.close()



plt.figure(figsize=(12,9))
plt.plot(hs_issc_results_df['Number of Selected Bands'], hs_issc_results_df['OA'], label='OA', marker='o')
plt.plot(hs_issc_results_df['Number of Selected Bands'], hs_issc_results_df['AA'], label='AA', marker='o')
plt.plot(hs_issc_results_df['Number of Selected Bands'], hs_issc_results_df['Kappa'], label='Kappa', marker='o')
plt.xticks(rotation=90)
plt.xlabel('Number of Selected Bands')
plt.ylabel('Accuracy & Kappa Score')
plt.legend()
plt.tight_layout()
plt.savefig('./results_data/hs_issc_results_plot.png', bbox_inches='tight')

# Clear plot data for next plot
plt.clf()
plt.close()


plt.figure(figsize=(12,9))
plt.plot(hs_manual_results_df['Experiment'], hs_manual_results_df['OA'], label='OA', marker='o')
plt.plot(hs_manual_results_df['Experiment'], hs_manual_results_df['AA'], label='AA', marker='o')
plt.plot(hs_manual_results_df['Experiment'], hs_manual_results_df['Kappa'], label='Kappa', marker='o')
plt.xticks(rotation=90)
plt.xlabel('Experiment')
plt.ylabel('Accuracy & Kappa Score')
plt.legend()
plt.tight_layout()
plt.savefig('./results_data/hs_manual_results_plot.png', bbox_inches='tight')

# Clear plot data for next plot
plt.clf()
plt.close()


plt.figure(figsize=(12,9))
plt.plot(hs_pca_results_df['Number of Principle Components'], hs_pca_results_df['OA'], label='OA', marker='o')
plt.plot(hs_pca_results_df['Number of Principle Components'], hs_pca_results_df['AA'], label='AA', marker='o')
plt.plot(hs_pca_results_df['Number of Principle Components'], hs_pca_results_df['Kappa'], label='Kappa', marker='o')
plt.xticks(rotation=90)
plt.xlabel('Number of Principle Components')
plt.ylabel('Accuracy & Kappa Score')
plt.legend()
plt.tight_layout()
plt.savefig('./results_data/hs_pca_results_plot.png', bbox_inches='tight')

# Clear plot data for next plot
plt.clf()
plt.close()


plt.figure(figsize=(12,9))
plt.plot(late_fusion_hs_issc_results_df['Number of Selected Bands'], late_fusion_hs_issc_results_df['OA'], label='OA', marker='o')
plt.plot(late_fusion_hs_issc_results_df['Number of Selected Bands'], late_fusion_hs_issc_results_df['AA'], label='AA', marker='o')
plt.plot(late_fusion_hs_issc_results_df['Number of Selected Bands'], late_fusion_hs_issc_results_df['Kappa'], label='Kappa', marker='o')
plt.xticks(rotation=90)
plt.xlabel('Number of Selected Bands')
plt.ylabel('Accuracy & Kappa Score')
plt.legend()
plt.tight_layout()
plt.savefig('./results_data/late_fusion_hs_issc_results_plot.png', bbox_inches='tight')

# Clear plot data for next plot
plt.clf()
plt.close()


plt.figure(figsize=(12,9))
plt.plot(single_modality_results_df['Modality'], single_modality_results_df['OA'], label='OA', marker='o')
plt.plot(single_modality_results_df['Modality'], single_modality_results_df['AA'], label='AA', marker='o')
plt.plot(single_modality_results_df['Modality'], single_modality_results_df['Kappa'], label='Kappa', marker='o')
plt.xticks(rotation=90)
plt.xlabel('Modality')
plt.ylabel('Accuracy & Kappa Score')
plt.legend()
plt.tight_layout()
plt.savefig('./results_data/single_modality_results_plot.png', bbox_inches='tight')

# Clear plot data for next plot
plt.clf()
plt.close()


three_branch_combo_fusion_results_df['label'] = three_branch_combo_fusion_results_df.apply(lambda row: f"{row['Branch 1']} + {row['Branch 2']} + {row['Branch 3']}", axis=1)

plt.figure(figsize=(12,9))
plt.plot(three_branch_combo_fusion_results_df['label'], three_branch_combo_fusion_results_df['OA'], label='OA', marker='o')
plt.plot(three_branch_combo_fusion_results_df['label'], three_branch_combo_fusion_results_df['AA'], label='AA', marker='o')
plt.plot(three_branch_combo_fusion_results_df['label'], three_branch_combo_fusion_results_df['Kappa'], label='Kappa', marker='o')
plt.xticks(rotation=90)
plt.xlabel('Branches')
plt.ylabel('Accuracy & Kappa Score')
plt.legend()
plt.tight_layout()
plt.savefig('./results_data/three_branch_combo_fusion_results_plot.png', bbox_inches='tight')

# Clear plot data for next plot
plt.clf()
plt.close()


three_branch_late_fusion_results_df['label'] = three_branch_late_fusion_results_df.apply(lambda row: f"{row['Branch 1']} + {row['Branch 2']} + {row['Branch 3']}", axis=1)

plt.figure(figsize=(12,9))
plt.plot(three_branch_late_fusion_results_df['label'], three_branch_late_fusion_results_df['OA'], label='OA', marker='o')
plt.plot(three_branch_late_fusion_results_df['label'], three_branch_late_fusion_results_df['AA'], label='AA', marker='o')
plt.plot(three_branch_late_fusion_results_df['label'], three_branch_late_fusion_results_df['Kappa'], label='Kappa', marker='o')
plt.xticks(rotation=90)
plt.xlabel('Branches')
plt.ylabel('Accuracy & Kappa Score')
plt.legend()
plt.tight_layout()
plt.savefig('./results_data/three_branch_late_fusion_results_plot.png', bbox_inches='tight')

# Clear plot data for next plot
plt.clf()
plt.close()


two_branch_combo_fusion_results_df['label'] = two_branch_combo_fusion_results_df.apply(lambda row: f"{row['Branch 1']} + {row['Branch 2']}", axis=1)

plt.figure(figsize=(12,9))
plt.plot(two_branch_combo_fusion_results_df['label'], two_branch_combo_fusion_results_df['OA'], label='OA', marker='o')
plt.plot(two_branch_combo_fusion_results_df['label'], two_branch_combo_fusion_results_df['AA'], label='AA', marker='o')
plt.plot(two_branch_combo_fusion_results_df['label'], two_branch_combo_fusion_results_df['Kappa'], label='Kappa', marker='o')
plt.xticks(rotation=90)
plt.xlabel('Branches')
plt.ylabel('Accuracy & Kappa Score')
plt.legend()
plt.tight_layout()
plt.savefig('./results_data/two_branch_combo_fusion_results_plot.png', bbox_inches='tight')

# Clear plot data for next plot
plt.clf()
plt.close()


two_branch_late_fusion_results_df['label'] = two_branch_late_fusion_results_df.apply(lambda row: f"{row['Branch 1']} + {row['Branch 2']}", axis=1)

plt.figure(figsize=(12,9))
plt.plot(two_branch_late_fusion_results_df['label'], two_branch_late_fusion_results_df['OA'], label='OA', marker='o')
plt.plot(two_branch_late_fusion_results_df['label'], two_branch_late_fusion_results_df['AA'], label='AA', marker='o')
plt.plot(two_branch_late_fusion_results_df['label'], two_branch_late_fusion_results_df['Kappa'], label='Kappa', marker='o')
plt.xticks(rotation=90)
plt.xlabel('Branches')
plt.ylabel('Accuracy & Kappa Score')
plt.legend()
plt.tight_layout()
plt.savefig('./results_data/two_branch_late_fusion_results_plot.png', bbox_inches='tight')

# Clear plot data for next plot
plt.clf()
plt.close()


