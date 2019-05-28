

import numpy as np
import sklearn as sk
import scipy.signal
import pywt
import matplotlib.pyplot as plt
from statsmodels.robust import mad
from sklearn import svm
from io import open


class data_pool:
    def __init__(self):
        self.__working_dir = "/home/bonenberger/dataRecordings/"

        self.__num_ir_sensors = 6
        self.__num_fr_sensors = 2
        self.__num_odo_sensors = 2
        self.__num_imu_sensors = 13
        self.__ana_norm_fact = 1023
        self.__num_ana_sensors = self.__num_ir_sensors + self.__num_fr_sensors
        self.__total_num_sensor_vars = self.__num_ir_sensors + self.__num_fr_sensors + self.__num_odo_sensors + \
                                       self.__num_imu_sensors
        self.__total_num_columns = self.__total_num_sensor_vars + 1
        print(self.__total_num_columns)
        self.__selected_variables = np.linspace(1, self.__total_num_sensor_vars)
        self.__num_of_used_vars = len(self.__selected_variables)

    def read_file(self, file_name):

        """
        Add a source file to the dataset. Given inputs are stored in arrays.
        :param file_name: String specifying the file name.
        :param start_index: Time (in samples) from which the data in the file is read.
        :param stop_index:Time (in samples) until which the data in the file is read.
        :param label: Label of the file.
        :param class_name: Class name displayed by some functions (self.__liveClassification).
        :return: writes to:
         - self.__file_source_name
         - self.__file_source_start_index
         - self.__file_source_stop_index
         - self.__file_labels
         - self.__number_of_classes
         - self.__class_name
         - self.__number_windows_per_class (initialization only)
        """
        data = []
        file_id = self.__working_dir + file_name
        file = open(file_id, mode="rt", encoding="utf-8", newline="\n")
        content = file.read()
        file.close()
        content = content.split("\n")
        for i in range(len(content)):
            line = content[i]
            if not line.startswith("%"):
                line = line.split(",")
                for j in range(self.__total_num_columns):
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
        return data