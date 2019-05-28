#! /usr/bin/env python


from ocsvm import One_class_svm
import rospy
import numpy as np
import pywt
from std_msgs.msg import String


class Live_classifier:

    def __init__(self, ):
        self.__frir_data = ""
        self.__imu_data = ""
        self.__mag_data = ""
        self.__odo_data = ""
	self.__only_once = True

    def init_window(self, ):
        self.__data_window = np.zeros([self.__live_ocsvm.return_param(param="window_width"),
                                       int(self.__live_ocsvm.return_param(param="total_num_columns"))])
        self.__old_data = self.__data_window[-1, ::]
        self.__new_data = self.__data_window[-1, ::]
        self.__window_shift_count = 0
        self.__window_shift = self.__live_ocsvm.return_param(param="window_shift")

    def subscribers(self, ):
        self.train_clf()
	if self.__only_once:
	    self.init_window()
	    self.__only_once = False
        rospy.init_node('live_classifier', anonymous=True)
        rospy.Subscriber('fr_ir_raw', String, self.classify_data)
        rospy.Subscriber('imu_raw', String, self.read_imu_data)
        rospy.Subscriber('mag_raw', String, self.read_mag_data)
        rospy.Subscriber('odo_raw', String, self.read_odo_data)
        rospy.spin()

    def classify_data(self, data):
        data = data.data
        self.__frir_data = data
        line = self.__frir_data + "," + self.__imu_data + "," + self.__mag_data + ',' + self.__odo_data + str("\n")

        line = line.split(",")
        for j in range(self.__live_ocsvm.return_param(param="total_num_columns")):
            try:
                # line[j] = line[j]
                try:
                    line[j] = float(line[j])
                except ValueError:
                    line[j] = 0
            except IndexError:
                line = []

        data = line

        if len(data) == int(self.__live_ocsvm.return_param(param="total_num_columns")):
            self.__new_data = data
            self.__old_data = data
        else:
            self.__new_data = self.__old_data
        self.__data_window = np.roll(self.__data_window, -1, 0)
        self.__data_window[-1, ::] = self.__new_data
        self.__window_shift_count += 1
        if self.__window_shift <= self.__window_shift_count:
            prediction = self.__live_ocsvm.live_classification(self.__data_window)
            print(prediction)
            self.__window_shift_count = 0

    def read_imu_data(self, data):
        data = data.data
        self.__imu_data = data

    def read_mag_data(self, data):
        data = data.data
        self.__mag_data = data

    def read_odo_data(self, data):
        data = data.data
        self.__odo_data = data

    def train_clf(self, ):
        self.__live_ocsvm = One_class_svm(window_width=100,
                       window_shift=10,
                       balanced_classes=False,
                       greedy_train=False,
                       num_coefficients=1,
                       num_frequencies=1,
                       wavelet='coif1',
                       ena_stat_feats=True,
                       corr_peaks=1,
                       ena_raw_feats=False,
                       stat_stages=1,
                       outlier_prob=0.00)

        self.__live_ocsvm.sel_var_subset(ir_vars=np.array([0, 1, 2, 3, 4, 5], dtype=int),
                     fr_vars=np.array([], dtype=int),
                     quat_axis=np.array([], dtype=int),
                     av_axis=np.array([], dtype=int),
                     la_axis=np.array([], dtype=int),
                     mag_axis=np.array([], dtype=int),
                     odo_vars=np.array([], dtype=int))

        # normal:
        self.__live_ocsvm.add_data_files("2019_1_15_16_52.txt", start_index=37000, stop_index=47000, label=0)
        self.__live_ocsvm.add_data_files("2019_1_15_18_4_Outdoor.txt", start_index=4500, stop_index=9500, label=0)
        self.__live_ocsvm.add_data_files("2019_1_15_18_4_Outdoor.txt", start_index=13000, stop_index=15000, label=0)
        self.__live_ocsvm.add_data_files("2019_1_15_18_4_Outdoor.txt", start_index=18500, stop_index=20250, label=0)
        self.__live_ocsvm.add_data_files("2019_1_15_18_4_Outdoor.txt", start_index=25000, stop_index=28000, label=0)
        self.__live_ocsvm.add_data_files("2019_1_15_18_4_Outdoor.txt", start_index=3000, stop_index=17000, label=0)
        # no gait:
        self.__live_ocsvm.add_data_files("2019_1_15_16_52.txt", start_index=9000, stop_index=13500, label=1)
        self.__live_ocsvm.add_data_files("2019_1_15_16_52.txt", start_index=30600, stop_index=31600, label=1)
        self.__live_ocsvm.add_data_files("2019_1_15_16_52.txt", start_index=47500, stop_index=50500, label=1)
        self.__live_ocsvm.add_data_files("2019_1_15_17_7_Gehsteig.txt", start_index=19200, stop_index=20200, label=1)
        self.__live_ocsvm.add_data_files("2019_1_15_17_7_Gehsteig.txt", start_index=22000, stop_index=25000, label=1)
        self.__live_ocsvm.add_data_files("2019_1_15_17_7_Gehsteig.txt", start_index=0, stop_index=1000, label=1)
        self.__live_ocsvm.add_data_files("2019_1_15_18_4_Outdoor.txt", start_index=0, stop_index=800, label=1)
        self.__live_ocsvm.add_data_files("2019_1_15_18_4_Outdoor.txt", start_index=29200, stop_index=30500, label=1)
        self.__live_ocsvm.set_window_function(functionName="tukey", alpha=0.1, plot_output=False)
        self.__live_ocsvm.read_data_set_from_files(plot_output=False)
        self.__live_ocsvm.split_training_test(plot_output=False)
        self.__live_ocsvm.window_data()
        self.__live_ocsvm.extract_and_normalize_features()
        self.__live_ocsvm.classify_data()


def main(f_n=None, f_p=None):
    clf = Live_classifier()
    clf.subscribers()


if __name__ == '__main__':
    main()


