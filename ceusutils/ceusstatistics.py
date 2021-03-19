import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
object = pd.read_pickle(r'Output\Sequential12\11\B113-S1-Asus I5-9400 GPU RTX 2060\output-09-Feb-2021_0537X.pkl')

# transform dict to pd.dataframe
#pdf = pd.DataFrame.from_dict(object, orient='index')
#pdf.iloc[0, 2][1]
#print("FNH patients: ", len(object['0']['FNH'][1]), "\n")

allexpmeanperlessions = []
for key1 in object:
    print("=========================")
    print("Experiment no.:", int(key1)+1)
    j=0
    lmean = []
    plt.figure()
    plt.suptitle("Experiment " + key1)
    plt.subplots_adjust( hspace=1)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show(block=False)   
    for key in object[key1]:
        lmax = np.max(object[key1][key][1])
        lmean.append(np.mean(object[key1][key][1]))
        lmin = np.min(object[key1][key][1])
        lstd = np.std(object[key1][key][1]) 
        print('%9s' % key, "||", "max:", "%.2f ||" % lmax, "mean: ", "%.2f ||" % lmean[-1], "min: ", "%.2f ||" % lmin, "std: ", "%.2f ||" % lstd)
        #textstr = "max:", "%.2f" % lmax, "mean: ", "%.2f" % lmean[-1], "min: ", "%.2f" % lmin, "std: ", "%.2f" % lstd
        textstr = '\n'.join((
            r'$max=%.2f$' % (lmax, ),
            r'$mean=%.2f$' % (lmean[-1], ),
            r'$min=%.2f$' % (lmin, ),
            r'$std=%.2f$' % (lstd, )))
        print(object[key1][key][0], object[key1][key][1])
        j = j + 1
        plt.subplot(5, 1, j)
        x_pos = [i for i, _ in enumerate(object[key1][key][1])]
        plt.ylim(0, 1)
        plt.bar(x_pos, object[key1][key][1])
        plt.title(key, fontsize=10)
        plt.xticks(x_pos, object[key1][key][0])
        plt.xlabel("ID patient")
        plt.ylabel("Accuracy")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0, 1.1, textstr, horizontalalignment='left', transform=plt.gca().transAxes, fontsize=8, bbox=props)
    print("Accuracy per experiment", "%d is" % (int(key1)+1), "%.2f" % np.mean(lmean))
    allexpmeanperlessions.append(lmean)
print("==================================================")
print("Maxs: ", np.max(allexpmeanperlessions, axis=0))
print("Means: ", np.mean(allexpmeanperlessions, axis=0))
print("Mins: ", np.min(allexpmeanperlessions, axis=0))
print("Stds: ", np.std(allexpmeanperlessions, axis=0))
print("Accuracy over all experiments is: " "%.2f" %np.mean(allexpmeanperlessions))
print("===========================================================================")
xx = np.asarray(allexpmeanperlessions).T
labels = ['FNH', 'HCC', 'HMG', 'MHIPER', 'MHIPO']
plt.figure()
plt.show(block=False)  
plt.ylim(0, 1)
plt.boxplot(xx.tolist(), labels = labels, meanline=True)
plt.show()