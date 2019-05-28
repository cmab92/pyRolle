import  numpy as np
import pywt
import matplotlib.pyplot as plt
from statsmodels.robust import mad
from io import open

data = []
total_num_columns = 22
file_name = "2019_1_15_18_4_Outdoor.txt"
file_id = "/home/bonenberger/dataRecordings/" + file_name
file = open(file_id, mode="rt", encoding="utf-8", newline="\n")
content = file.read()
file.close()
content = content.split("\n")
for i in range(len(content)):
    line = content[i]
    if not line.startswith("%"):
        line = line.split(",")
        for j in range(total_num_columns):
            try:
                # line[j] = line[j]
                try:
                    line[j] = float(line[j])
                except ValueError:
                    line[j] = 0
            except IndexError:
                line = []
        if len(line) >= 1:
            data.append(line)
    else:
        if "Anomaly" in line:
            # print(line)
            pass
data = np.array(data, dtype=float)
data[::, 0] = (data[::, 0] - data[0, 0]) * 10 ** (-9)
time = data[0:2**12, 0]
data = data[0:2**12, 1]
data = (data - np.median(data))/mad(data)
wavelet = "bior6.8"
wavelet = "coif1"
#wavelet = "db2"
#wavelet = "db20"
wavelet = "haar"
print(np.shape(data))
n_lvl = int(np.log2(len(data)))
n_subplots = n_lvl + 2
threshold = 3*mad(data)

coefficients = pywt.wavedec(data=data, wavelet=wavelet, mode='symmetric', level=n_lvl)

plt.subplot(n_subplots, 2, 1)
plt.plot(time, data, label="Source Signal")

# first lvl coeffs thresholding:
plt.subplot(n_subplots, 2, 3)
plt.plot(coefficients[0], 'b')
coefficients[0][np.abs(coefficients[0]) < threshold] = 0
#coefficients[0][np.abs(coefficients[0]) > 0] = 0
plt.plot(coefficients[0], 'r')

# second lvl coeffs thresholding:
plt.subplot(n_subplots, 2, 5)
plt.plot(coefficients[1], 'b')
coefficients[1][np.abs(coefficients[1]) < threshold] = 0
plt.plot(coefficients[1], 'r')

# plot base fct first lvl:
p_cA = np.zeros(np.size(coefficients[0]))
p_cD = np.zeros(np.size(coefficients[0]))
p_cA[int(len(coefficients[0])/2)] = 1
plt.subplot(n_subplots, 2, 4)
plt.plot(pywt.idwt(p_cA, p_cD, wavelet))

# plot base fct first lvl:
p_cA = np.zeros(np.size(coefficients[1]))
p_cD = np.zeros(np.size(coefficients[1]))
p_cD[int(len(coefficients[1])/2)] = 1
plt.subplot(n_subplots, 2, 6)
plt.plot(pywt.idwt(p_cA, p_cD, wavelet))
for i in range(n_lvl-2):
    plt.subplot(n_subplots, 2, 2*(i+4)-1)
    plt.plot(np.linspace(0, np.max(time), len(coefficients[i+2])), coefficients[i+2], 'b')
    # thresholding
    coefficients[i+2][np.abs(coefficients[i+2]) < threshold] = 0
    # plot
    plt.plot(np.linspace(0, np.max(time), len(coefficients[i+2])), coefficients[i+2], 'r')        #np.linspace(0, 1, len(coefficients[i+2])),
    plt.subplot(n_subplots, 2, 2*(i+4))
    p_cA = np.zeros(np.size(coefficients[i+2]))
    p_cD = np.zeros(np.size(coefficients[i+2]))
    p_cD[int(len(coefficients[i+2])/2)] = 1
    plt.plot(pywt.idwt(p_cA, p_cD, wavelet))
signal_rec = pywt.waverec(coeffs=coefficients, wavelet=wavelet, mode='symmetric')
plt.subplot(n_subplots, 2, 1)
plt.plot(time, signal_rec, label="Signal reconstruction")

plt.show()
