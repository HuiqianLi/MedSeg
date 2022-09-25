import matplotlib.pyplot as plt
import re
from mpl_toolkits.axes_grid1 import host_subplot
 
# read the log file
fp = open('/data/models/lhq/PNSNet/snapshot/log.log', 'r')
 
train_iterations = []
train_loss = []
 
for ln in fp:
    # get train_iterations and train_loss
    if 'Epoch [' in ln and 'Loss_AVG: ' in ln:
        arr1 = re.findall(r'\d+',ln)
        arr2 = re.findall(r'\d\b.\d+',ln)
        train_iterations.append(int(arr1[-4]))
        train_loss.append(float(arr2[-1]))
fp.close()
 
host = host_subplot(111)
plt.subplots_adjust(right=0.8) # ajust the right boundary of the plot window
par1 = host.twinx()
# set labels
host.set_xlabel("iterations")
host.set_ylabel("log loss")
 
# plot curves
p1, = host.plot(train_iterations, train_loss, label="training log loss")
 
# set location of the legend, 
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
host.legend(loc=5)
 
# set label color
host.axis["left"].label.set_color(p1.get_color())
# set the range of x axis of host and y axis of par1
host.set_xlim([0, 100])
 
plt.draw()
# plt.show()
plt.savefig('/data/models/lhq/PNSNet/squares_plot.png', bbox_inches='tight')