import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "amiri"
plt.rcParams["font.size"] = 14

 
# set width of bar
barWidth = 0.20
 
# set height of bar
bars1 = [0.67, 0.47, 0.75, 0.70, 0.55, 0.51, 0.76, 0.68, 0.88, 0.69]
bars2 = [0.76, 0.30, 0.81, 0.72, 0.75, 0.69, 0.81, 0.72, 0.86, 0.69]
bars3 = [0.76, 0.77, 0.81, 0.72, 0.78, 0.69, 0.82, 0.72, 0.93, 0.74]
 
# Set position of bar on X axis
r0 = np.arange(len(bars1))
r1 = [x - barWidth*1.0 for x in r0]
r2 = [x                for x in r0]
r3 = [x + barWidth*1.0 for x in r0]
 
# Make the plot
plt.figure(figsize=(9,5))
plt.bar(r1, bars1, color='#003f5c', width=barWidth, edgecolor='white', label='GNN')
# plt.bar(r2, bars2, color='#bc5090', width=barWidth, edgecolor='white', label='GNN/RP')
# plt.bar(r3, bars3, color='#ffa600', width=barWidth, edgecolor='white', label='C-GNN')
 
# Add xticks on the middle of the group bars
plt.xlim([-0.5, len(bars1)-0.5])
plt.xticks(r0, ['Ising(+)', 'Ising(-)', 'Income', 'Edu', 'Emp', 'Vote', 'Anaheim', 'Chicago', 'Sexual', 'Twitch'])
plt.ylabel('Accuracy / R$^{2}$')
plt.ylim([0.2, 1.0])
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
 
# Create legend & Show graphic
plt.legend(ncol=3, loc=2, fontsize=15)
plt.tight_layout()
plt.savefig("results.svg", bbox_inches='tight')
