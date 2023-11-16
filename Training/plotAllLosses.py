import time
import matplotlib
import matplotlib.pyplot as plt

import torch

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

def plotLosses(title, cur_epoch, total_epochs, values, labels, startTime):
    plt.figure(2)
    plt.clf()
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    for v in values:
        plt.plot(v)  
    plt.pause(0.001)

    lower_label = "[Epoch %d/%d] \n" % (cur_epoch, total_epochs)

    for i in range(len(labels)):
        lower_label += "[%s: %f] " % (labels[i], values[i][-1])
        if i % 3 == 0:
            lower_label += "\n"
    print(lower_label)

    secondsElapsed = time.time() - startTime
    minutesElapsed = secondsElapsed // 60
    hoursElapsed   = minutesElapsed // 60
    secondsElapsed = secondsElapsed % 60
    minutesElapsed = minutesElapsed % 60
    print("Elapsed Time %d:%d:%d" % (hoursElapsed, minutesElapsed, secondsElapsed))
    if is_ipython: display.clear_output(wait=True)

def avg_loss(losses, episodes=-1):
    if episodes < 0:
        episodes = len(losses)
    total = 0
    for i in range(0, episodes):
        total += losses[episodes - (i+1)]
    total /= episodes
    return total

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()
    
def setUpperBound(a, bound):
    for i in range(len(a)):
        if a[i] > bound:
            a[i] = bound
    return a
def checkBound(a):
    avg = sum(a) / len(a)
    a = setUpperBound(a, avg * 5)
    return a