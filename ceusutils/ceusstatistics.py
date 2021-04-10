import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import decimal
import seaborn as sns
import time

def ceusstatistics(file, plots):

    np.set_printoptions(precision=2)
    object, all_cm = pd.read_pickle(file+'.pkl')

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
        if plots:
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
            if plots:
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

    if __name__ != '__main__':
        f = open(file+'.txt','a+')
        f.write("\n")
        f.write("Maxs: " + str(np.max(allexpmeanperlessions, axis=0)))
        f.write("\n")
        f.write("Means: " + str(np.mean(allexpmeanperlessions, axis=0)))
        f.write("\n")
        f.write("Mins: " + str(np.min(allexpmeanperlessions, axis=0)))
        f.write("\n")
        f.write("Stds: " + str(np.std(allexpmeanperlessions, axis=0)))
        f.write("\n")
        f.write("Accuracy over all experiments is: " + "%.2f" %np.mean(allexpmeanperlessions))
        f.write("\n")
        f.write("Confusion matrix:" + str(all_cm))
        f.close()

    if plots:
        xx = np.asarray(allexpmeanperlessions).T
        labels = ['FNH', 'HCC', 'HMG', 'MHIPER', 'MHIPO']
        plt.figure()
        plt.show(block=False)  
        plt.ylim(0, 1)
        plt.boxplot(xx.tolist(), labels = labels, meanline=True)
        for key2 in object:
            plt.figure(figsize=(10, 8))
            sns.heatmap(all_cm[key2], xticklabels=labels, yticklabels=labels, annot=True, fmt='g')
            plt.xlabel('Prediction')
            plt.ylabel('Label')
        plt.show()

if __name__ == '__main__':

    saved = './Output/output-10-Apr-2021_2237hard-vote'
    ceusstatistics(saved, True)