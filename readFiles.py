"""
cb 14.11.18

general info:


function output:
"""

import numpy as np
import matplotlib.pyplot as plt
from io import open


def readFile(fileName, dir="/home/bonenberger/dataRecordings/", numOfSensors=24):
    data = []
    fileId = dir + fileName
    file = open(fileId, mode="rt", encoding="utf-8", newline="\n")
    content = file.read()
    file.close()
    content = content.split("\n")
    print("Start...")
    for i in range(len(content)):
        line = content[i]
        if not line.startswith("%"):
            line = line.split(",")
            for j in range(numOfSensors+1):
                try:
                    line[j] = line[j]
                    try:
                        line[j] = float(line[j])
                    except (IndexError, ValueError):
                        line[j] = 0.0
                except (IndexError, ValueError):
                    line = []
            if len(line) == numOfSensors+1:
                data.append(line)
        else:
            if "Anomaly" in line:
                print("Anomaly")
                print(line)
        #print(np.shape(line))
    data = np.array(data, dtype=float)
    print(data)
    data[::, 0] = (data[::, 0] - data[0, 0])*10**(-9)
    return data


if __name__ == '__main__':
    filenames = ["2019_4_26_16_57"]
    begin = 0
    end = 2000000
    for i in range(len(filenames)):
        data = readFile(fileName=filenames[i] + ".txt")
        t_in_min = data[::, 0]#/60

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        ax2.plot(data[begin:end, 0]/60, np.zeros(np.size(data[begin:end, 1])), alpha=0.01)
        ax1.plot(data[begin:end, 0], data[begin:end, 1], label="ir0")
        ax1.plot(data[begin:end, 0], data[begin:end, 2], label="ir1")
        ax1.plot(data[begin:end, 0], data[begin:end, 3], label="ir2")
        ax1.plot(data[begin:end, 0], data[begin:end, 4], label="ir3")
        ax1.plot(data[begin:end, 0], data[begin:end, 5], label="ir4")
        ax1.plot(data[begin:end, 0], data[begin:end, 6], label="ir5")
        ax1.legend(loc="upper right")
        ax1.set_xlabel(r"t in sec")
        ax2.set_xlabel(r"t in min")
        plt.title("IR-data")

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        ax2.plot(data[begin:end, 0]/60, np.zeros(np.size(data[begin:end, 1])), alpha=0.01)
        ax1.plot(data[begin:end, 0], data[begin:end, 7], label="fr_1")
        ax1.plot(data[begin:end, 0], data[begin:end, 8], label="fr_2")
#        ax1.plot(data[begin:end, 0], data[begin:end, 22], label="odo_1")
#        ax1.plot(data[begin:end, 0], data[begin:end, 23], label="odo_2")
        ax1.legend(loc="upper right")
        ax1.set_xlabel(r"t in sec")
        ax2.set_xlabel(r"t in min")
        plt.title("force and odo-data")


        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        ax2.plot(data[begin:end, 0]/60, np.zeros(np.size(data[begin:end, 1])), alpha=0.01)
        ax1.plot(data[begin:end, 0], data[begin:end, 9], label="quat_w")
        ax1.plot(data[begin:end, 0], data[begin:end, 10], label="quat_x")
        ax1.plot(data[begin:end, 0], data[begin:end, 11], label="quat_y")
        ax1.plot(data[begin:end, 0], data[begin:end, 12], label="quat_z")
        ax1.legend(loc="upper right")
        ax1.set_xlabel(r"t in sec")
        ax2.set_xlabel(r"t in min")
        plt.title("quat-data")


        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        ax2.plot(data[begin:end, 0]/60, np.zeros(np.size(data[begin:end, 1])), alpha=0.01)
        ax1.plot(data[begin:end, 0], data[begin:end, 13], label="ang_vec_x")
        ax1.plot(data[begin:end, 0], data[begin:end, 14], label="ang_vec_y")
        ax1.plot(data[begin:end, 0], data[begin:end, 15], label="ang_vec_z")
        ax1.legend(loc="upper right")
        ax1.set_xlabel(r"t in sec")
        ax2.set_xlabel(r"t in min")
        plt.title("angular_velocity-data")


        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        ax2.plot(data[begin:end, 0]/60, np.zeros(np.size(data[begin:end, 1])), alpha=0.01)
        ax1.plot(data[begin:end, 0], data[begin:end, 16], label="lin_acc_x")
        ax1.plot(data[begin:end, 0], data[begin:end, 17], label="lin_acc_y")
        ax1.plot(data[begin:end, 0], data[begin:end, 18], label="lin_acc_z")
        ax1.legend(loc="upper right")
        ax1.set_xlabel(r"t in sec")
        ax2.set_xlabel(r"t in min")
        plt.title("linear_acceleration-data")


        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        ax2.plot(data[begin:end, 0]/60, np.zeros(np.size(data[begin:end, 1])), alpha=0.01)
        ax1.plot(data[begin:end, 0], data[begin:end, 19], label="mag_x")
        ax1.plot(data[begin:end, 0], data[begin:end, 20], label="mag_y")
        ax1.plot(data[begin:end, 0], data[begin:end, 21], label="mag_z")
        ax1.legend(loc="upper right")
        ax1.set_xlabel(r"t in sec")
        ax2.set_xlabel(r"t in min")
        plt.title("mag-data")


        plt.title(filenames[i])

    plt.show()

