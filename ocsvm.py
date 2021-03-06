#! /usr/bin/python3.6
"""
cb 21.01.19

general info:

function output:

"""

import numpy as np
import sklearn as sk
import scipy.signal
import pywt
import matplotlib.pyplot as plt
from statsmodels.robust import mad
from sklearn import svm
from io import open
from six.moves import cPickle


class One_class_svm:

    def __init__(self, window_width, window_shift, num_odo_sensors=2, balanced_classes=False, train_frac=0.6,
                 greedy_train=False, num_coefficients=5, num_frequencies=2, wavelet='haar', ena_stat_feats=False,
                 corr_peaks=2, ena_raw_feats=False, stat_stages=4, outlier_prob=0.00):

        self.__working_dir = "/home/bonenberger/dataRecordings/"

        self.__file_sink_name = ""
        self.__file_source_name = []
        self.__file_source_start_index = []
        self.__file_source_stop_index = []
        self.__file_labels = []
        self.__class_name = []
        self.__balanced_classes = balanced_classes                              # same number of samples for all classes
        self.__train_frac = train_frac                          # fraction of training data (n_train/(n_train + n_test))
        self.__greedy_train = greedy_train                  # if enabled window_shift is set to 1 for training data only

        self.__number_of_classes = 0
        self.__number_windows_per_class = 0

        self.__window_width = window_width
        self.__window_shift = window_shift
        self.__window_alpha = 0.1
        self.__window_function = scipy.signal.tukey(self.__window_width, self.__window_alpha)
        self.__sample_interval = 0.0165

        self.__num_ir_sensors = 6
        self.__num_fr_sensors = 2
        self.__num_odo_sensors = num_odo_sensors
        self.__num_imu_sensors = 13
        self.__ana_norm_fact = 1023
        self.__num_ana_sensors = self.__num_ir_sensors + self.__num_fr_sensors
        self.__total_num_sensor_vars = self.__num_ir_sensors + self.__num_fr_sensors + self.__num_odo_sensors + \
                                       self.__num_imu_sensors
        self.__total_num_columns = self.__total_num_sensor_vars + 1
        print(self.__total_num_columns)
        self.__selected_variables = np.linspace(1, self.__total_num_sensor_vars)
        self.__num_of_used_vars = len(self.__selected_variables)

        self.__source_data = []
        self.__raw_training_data = []
        self.__raw_test_data = []
        self.__training_data = []
        self.__test_data = []
        self.__training_labels = []
        self.__test_labels = []
        self.__training_data_raw_windows = []
        self.__test_data_raw_windows = []
        self.__feature_of_variable = []

        self.__num_coefficients = num_coefficients                          # considering n maximum wavelet coefficients
        self.__num_frequencies = num_frequencies                                     # considering n maximum frequencies
        self.__wavelet = wavelet                                                                        # mother wavelet
        self.__ena_stat_feats = ena_stat_feats                                      # enable use of statistical features
        self.__corr_peaks = corr_peaks                                              # considering n peaks in correlation
        self.__ena_raw_feats = ena_raw_feats                                            # add raw data to feature vector
        self.__stat_stages = stat_stages       # specifying number of stages on which mean and std is calculated (log_2)

        self.__std_mue = []                                    # data standardization by feature = (feature - mue)/sigma
        self.__std_sigma = []                                                                                      # ...
        self.__feature_validity = []                              # bool-Array, if feature has 0-variance entry is False
        self.__oidp = outlier_prob                                      # probability for a outlier in training data set

    def return_param(self, param):
        if param == "total_num_columns":
            return self.__total_num_columns
        elif param == "window_width":
            return self.__window_width
        elif param == "window_shift":
            return self.__window_shift

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

    def sel_var_subset(self, ir_vars=np.array([0, 1, 2, 3, 4, 5], dtype=int), fr_vars=np.array([0, 1], dtype=int),
                       quat_axis=np.array([], dtype=int), av_axis=np.array([], dtype=int),
                       la_axis=np.array([], dtype=int), mag_axis=np.array([], dtype=int),
                       odo_vars=np.array([], dtype=int)):
        """
        Select a subset of variables/sensors to be used...
        :param ir_vars: IR-data to be used (ir0, ..., ir5), e.g.: ir_vars=np.array([0,2,4]
        :param fr_vars: FSR-data to be used (fr0, fr1)
        :param quat_axis: Quaternion-data to be used (w,x,y,z), e.g. quat_axis=np.array([0,1,3] selects w,x,z
        :param av_axis: Angular velocity data (x,y,z-axis)
        :param la_axis: Linear acceleration data (x,y,z-axis)
        :param mag_axis: Magnetometer data (x,y,z-axis)
        :param odo_vars: Odometry data, left/right wheel (odo0, odo1)
        :return:
        Writes to:
         - self.__selected_variables: Int-Array specifying the column-index of the variables/sensors to be considered
         - self.__num_of_used_vars:  = len(self.__selected_variables)
        Depends on:
        """

        self.__selected_variables = []
        ir_offset = 1                                                                                                   # +1 offset (first column is time)
        fr_offset = ir_offset + self.__num_ir_sensors
        quat_offset = fr_offset + self.__num_fr_sensors
        av_offset = quat_offset + 4
        la_offset = av_offset + 3
        mag_offset = la_offset + 3
        odo_offset = mag_offset + 3

        if len(ir_vars) == 0:
            print("(sel_var_subset) IR-data NOT used!")
        else:
            for i in range(len(ir_vars)):
                self.__selected_variables.append(ir_vars[i] + ir_offset)

        if len(fr_vars) == 0:
            print("(sel_var_subset) FSR-data NOT used!")
        else:
            for i in range(len(fr_vars)):
                self.__selected_variables.append(fr_vars[i] + fr_offset)

        if (len(quat_axis)+len(av_axis)+len(la_axis)+len(mag_axis))==0:
            print("(sel_var_subset) BNO-data unused!")
        else:
            if len(quat_axis) == 0:
                print("(sel_var_subset) Quaternion-data unused!")
            else:
                for i in range(len(quat_axis)):
                    self.__selected_variables.append(quat_axis[i] + quat_offset)

            if len(av_axis) == 0:
                print("(sel_var_subset) Angular velocity-data unused!")
            else:
                for i in range(len(av_axis)):
                    self.__selected_variables.append(av_axis[i] + av_offset)

            if len(la_axis) == 0:
                print("(sel_var_subset) Linear acceleration-data unused!")
            else:
                for i in range(len(la_axis)):
                    self.__selected_variables.append(la_axis[i] + la_offset)

            if len(mag_axis) == 0:
                print("(sel_var_subset) Magnetometer-data unused!")
            else:
                for i in range(len(mag_axis)):
                    self.__selected_variables.append(mag_axis[i] + mag_offset)

        if len(odo_vars) == 0:
            print("(sel_var_subset) Odometry-data unused!")
        else:
            for i in range(len(odo_vars)):
                self.__selected_variables.append(odo_vars[i] + odo_offset)

        self.__selected_variables = np.array(self.__selected_variables, dtype=int)
        self.__num_of_used_vars = len(self.__selected_variables)

        return self.__selected_variables

    def add_data_files(self, file_source_name, start_index=0, stop_index=10**16, label=None, class_name=None):
        """
        Add a source file to the dataset. Given inputs are stored in arrays.
        :param file_source_name: String specifying the file name. Default "".
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

        if isinstance(file_source_name, str) and isinstance(start_index, (int, float)) and \
                isinstance(stop_index, (int, float)) and isinstance(label, int):
            self.__file_source_name.append(file_source_name)
            self.__file_source_start_index.append(start_index)
            self.__file_source_stop_index.append(stop_index)
            self.__file_labels.append(label)
        else:
            if not(isinstance(start_index, (int, float))) or not(isinstance(stop_index, (int, float))) or \
                    not(isinstance(file_source_name, str)):
                print("(add_data_files) File source name, start-time, stop-time or label are incorrect data-type or "
                      "format!")
            elif not(isinstance(label, int)):
                print("(add_data_files) Give proper (int-type) label!")

        new_class = True
        for i in range(len(self.__file_labels)-1):
            if label == self.__file_labels[i]:
                new_class = False

        self.__number_of_classes = len(set(self.__file_labels))
        self.__number_windows_per_class = np.zeros(self.__number_of_classes)

        if new_class is True:
            if isinstance(class_name, str):
                self.__class_name.append(class_name)
            else:
                self.__class_name.append(file_source_name[:-4])

    def read_data_set_from_files(self, plot_output=False):

        """
        Read the data-set specified by addDataFiles. Triggers self.__source_data!
        :param check_data: If True the data-set is plotted.
        :return: Array with shape data_set[i][j, k], where i refers to the i-th file loaded, k indicates the sensor and
        j is the "time"-index.
        Writes to:
         - self.__source_data
        Depends on:
         - self.__balanced_data: Is True the data read from different file is cut to equal length.
         - self.__selected_variables: Int-Array indicating the selected variables (in function self.sel_var_subset)
        """

        if len(self.__file_source_name) == 0:
            print("(read_data_set_from_file) No file sources given.")
            return False
        else:
            start_indices = np.array(self.__file_source_start_index)
            stop_indices = np.array(self.__file_source_stop_index)

            data_set = []
            for i, element in enumerate(self.__file_source_name):
                file_data = self.read_file(file_name=element)
                data_set_temp = []
                for j in range(len(self.__selected_variables)):
                    data_set_temp.append(file_data[start_indices[i]:stop_indices[i], self.__selected_variables[j]])
                selected_data = np.array(data_set_temp).T
                data_set.append(selected_data)

            if plot_output:
                for i, element in enumerate(self.__file_source_name):
                    plt.figure()
                    for j in range(len(data_set[i][0, ::])):
                        plt.plot(data_set[i][::, j], label=str(j))
                    title = "File: " + self.__file_source_name[i] + ", Label: " + \
                            str(self.__file_labels[i])
                    plt.title(title)
                    plt.legend()
                plt.show()

            self.__source_data = np.array(data_set)

            return np.array(data_set)

    def set_window_function(self, functionName, alpha, plot_output=False):

        """
        Set window function and parameter. Extra function, to emphasize if changed.
        :param functionName: Chooses a window function of the following (https://en.wikipedia.org/wiki/Window_function):
        tukey -> tukey window (flattened cosine)
        rect -> rectangular window
        bart -> bartlett window
        black -> blackman window
        ham -> hamming window
        hann -> hanning window (raised-cosine window)
        kaiser -> kaiser window
        gauss -> gaussian window
        Default "tukey".
        :param alpha: Shape parameter of window function (not relevant for all). Default 0.1.
        :return: writes to:
         - self.__window_function
         - self.__window_alpha
        """

        if isinstance(alpha, (int, float)):
            self.__window_alpha = alpha
        else:
            print("(set_window_function) Give int or float for window shape parameter alpha.")

        if isinstance(functionName, str):
            if functionName == 'tukey':
                self.__window_function = scipy.signal.tukey(self.__window_width, self.__window_alpha)
            elif functionName == 'rect':
                self.__window_function = np.ones(self.__window_width)
            elif functionName == 'bart':
                self.__window_function = np.bartlett(self.__window_width)
            elif functionName == 'black':
                self.__window_function = np.blackman(self.__window_width)
            elif functionName == 'ham':
                self.__window_function = np.hamming(self.__window_width)
            elif functionName == 'hann':
                self.__window_function = np.hanning(self.__window_width)
            elif functionName == 'kaiser':
                self.__window_function = np.kaiser(self.__window_width, self.__window_alpha)
            elif functionName == 'gauss':
                self.__window_function = scipy.signal.gaussian(self.__window_width, self.__window_alpha)
            else:
                print("(set_window_function) Give proper function name.")
        else:
            print("(set_window_function) Give str as window function name.")

        if plot_output:
            """
            Simply plot time function and spectrum of window function. Only for convenience.
            :return: -
            """

            timeAxis = np.linspace(0, self.__window_width * self.__sample_interval, self.__window_width)
            plt.plot(timeAxis, self.__window_function)
            plt.title('Time Function of Window')
            plt.xlabel('t in s')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.figure()
            freqAxis = np.linspace(0, 2 / self.__sample_interval, int(self.__window_width))
            windowFreqResponse = np.abs(np.fft.fftshift(np.fft.fft(np.concatenate((
                np.zeros(int(self.__window_width / 2)), self.__window_function,
                np.zeros(int(self.__window_width / 2)))))))
            windowFreqResponse = 20 * np.log10(windowFreqResponse / np.max(windowFreqResponse))
            plt.plot(freqAxis, windowFreqResponse[int(self.__window_width):])
            plt.xlim(0, )
            plt.ylim(-120, )
            plt.title("Frequency Response of chosen Window (Conv. Th.)")
            plt.xlabel('f in Hz')
            plt.ylabel('dB')
            plt.grid()
            plt.show()

    def split_training_test(self, plot_output=False):

        """
        Split the underlying data (self.__source_data) into training and test (according to the fraction specified as
        self.__train_frac).
        :param check_data: If True the data-output is plotted (from each file separately).
        :return:
        Writes to:
         - self.__raw_training_data
         - self.__raw_test_data
        Depends on:
         - self.__train_frac: see above
        """

        for i in range(len(self.__file_source_name)):
            len_train = int(len(self.__source_data[i][::, 0])*self.__train_frac)
            self.__raw_training_data.append(self.__source_data[i][0:len_train, ::])
            self.__raw_test_data.append(self.__source_data[i][len_train:-1, ::])

        if plot_output:
            for i in range(len(self.__file_source_name)):
                plt.figure()
                for j in range(len(self.__raw_training_data[i][0, ::])):
                    plt.plot(self.__raw_training_data[i][::, j])
            plt.show()

    def extract_features(self, data):

        """
        Extracts features of a single data window.
        :param data: Input data, given as data[j,k], with j-th data-point, k-th sensor.
        :return: A single feature vector.
        """

        feature_vector = []
        feature_of_variable = []
        np.seterr(all='raise')

        # wavelet features:

        coefficients_val = []
        coefficients_amp = []
        coefficients_amp_1 = []
        if self.__num_coefficients != 0:
            for i in range(self.__num_of_used_vars):
                coefficients = pywt.wavedec(data[::, i], wavelet=self.__wavelet, mode='symmetric', level=1)
                coefficients_0 = coefficients[0]
                coefficients_1 = coefficients[1]
                coefficients_amp.append(coefficients_0[coefficients_0.argsort()[-self.__num_coefficients:]])
                coefficients_amp_1.append(coefficients_1[coefficients_1.argsort()[-self.__num_coefficients:]])
                translation_axis = np.linspace(-1, 1, np.size(coefficients_0))
                #
                if np.max(coefficients_0) == 0:
                    coefficients_val.append(np.zeros(self.__num_coefficients))
                else:
                    coefficients_val.append(translation_axis[coefficients_0.argsort()[-self.__num_coefficients:]])
                for j in range(np.size(coefficients_val[i])):
                    temp = coefficients_val
                    feature_vector.append(temp[i][j])
                    feature_of_variable.append(i)
                for j in range(np.size(coefficients_amp[i])):
                    temp = coefficients_amp
                    feature_vector.append(temp[i][j])
                    feature_of_variable.append(i)

        # fourier features:

        dominant_freq_val = []
        dominant_freq_amp = []
        dominant_freq_pha = []
        if self.__num_frequencies != 0:
            freq_axis = np.linspace(-1, 1, int(self.__window_width))
            for i in range(self.__num_of_used_vars):
                spectrum = np.fft.fftshift(np.fft.fft(data[::, i]))[int(self.__window_width / 2):]
                abs_spectrum = np.abs(np.fft.fftshift(np.fft.fft(data[::, i])))[int(self.__window_width / 2):]
                real_s = np.real(spectrum[abs_spectrum.argsort()[-self.__num_frequencies:]])
                imag_s = np.imag(spectrum[abs_spectrum.argsort()[-self.__num_frequencies:]])
                dominant_freq_amp.append(np.sqrt(real_s ** 2 + imag_s ** 2))
                try:
                    dominant_freq_pha.append(np.arctan(imag_s / real_s))
                except FloatingPointError:
                    temp = real_s
                    temp[real_s == 0] = 1
                    temp_2 = np.arctan(imag_s / temp)
                    temp_2[real_s == 0] = 0
                    dominant_freq_pha.append(temp)
                dominant_freq_val.append(freq_axis[abs_spectrum.argsort()[-self.__num_frequencies:]])
                for j in range(np.size(dominant_freq_val[i])):
                    temp = dominant_freq_val
                    feature_vector.append(temp[i][j])
                    feature_of_variable.append(i)
                for j in range(np.size(dominant_freq_amp[i])):
                    temp = dominant_freq_amp
                    feature_vector.append(temp[i][j])
                    feature_of_variable.append(i)
                for j in range(np.size(dominant_freq_pha[i])):
                    temp = dominant_freq_pha
                    feature_vector.append(temp[i][j])
                    feature_of_variable.append(i)

        # statistical features:

        if self.__ena_stat_feats:
            for i in range(self.__num_of_used_vars):
                for j in range(self.__stat_stages):
                    interval_width = int(self.__window_width/(2**j))
                    if interval_width>1:
                        num_intervals = int(self.__window_width/interval_width)
                        for k in range(num_intervals):
                            feature_vector.append(np.var(data[k*interval_width:(k+1)*interval_width, i]))
                            feature_of_variable.append(i)
                            feature_vector.append(mad(data[k*interval_width:(k+1)*interval_width, i]))
                            feature_of_variable.append(i)

            if self.__corr_peaks != 0:
                for i in range(self.__num_of_used_vars):
                    for j in range(self.__num_of_used_vars - i - 1):
                        correlation = np.correlate(data[::, i], data[::, j + 1], mode='same') \
                                      / np.sum(data[::, i]) / np.size(data[::, i])
                        coefficients = pywt.wavedec(correlation, wavelet=self.__wavelet, mode='symmetric', level=0)
                        coefficients_0 = coefficients[0]
                        translation_axis = np.linspace(-1, 1, np.size(coefficients_0))
                        dom_corr_coeff_amp = coefficients_0[coefficients_0.argsort()[-self.__corr_peaks:]]
                        if np.max(coefficients_0) == 0:
                            dom_corr_coeff_val = np.zeros(self.__corr_peaks)
                        else:
                            dom_corr_coeff_val = translation_axis[coefficients_0.argsort()[-self.__corr_peaks:]]
                        for k in range(self.__corr_peaks):
                            feature_vector.append(dom_corr_coeff_val[k])
                            feature_of_variable.append(i)
                        for k in range(self.__corr_peaks):
                            feature_vector.append(dom_corr_coeff_amp[k])
                            feature_of_variable.append(i)

        # raw features

        if self.__ena_raw_feats:
            for i in range(self.__num_of_used_vars):
                for j in range(self.__window_width):
                    feature_vector.append(data[j, i])
                    feature_of_variable.append(i)

        feature_vector = np.reshape(feature_vector, np.size(feature_vector))
        feature_of_variable = np.reshape(feature_of_variable, np.size(feature_of_variable))
        self.__feature_of_variable = feature_of_variable

        return feature_vector

    def window_data(self, ):

        """
        Windowing of training and test data.
        :param data: data[i, j], where i is "time"-index and j is indexing the variable/sensor
        :param label: integer indicating the data
        :return:
        Writes to:
         - self.__training_data
         - self.__test_data
         - self.__training_labels
         - self.__test_labels
        """

        if self.__greedy_train:
            print("(window_data) Greedy exploitation of training data!")
            temp_window_shift = 1
        else:
            temp_window_shift = self.__window_shift

        for i in range(len(self.__file_source_name)):
            # training data
            temp_data = self.__raw_training_data[i]
            temp_length = len(temp_data[::, 0])
            temp_num_windows = int((temp_length - self.__window_width) / temp_window_shift + 1)
            for j in range(temp_num_windows):
                temp_window = []
                for k in range(len(temp_data[0, ::])):
                    temp_window.append(
                        temp_data[j*temp_window_shift:(j*temp_window_shift + self.__window_width), k] *
                        self.__window_function)
                self.__training_data_raw_windows.append(np.transpose(temp_window))
                self.__training_labels.append(self.__file_labels[i])

            # test data
            temp_data = self.__raw_test_data[i]
            temp_length = len(temp_data[::, 0])
            temp_num_windows = int((temp_length - self.__window_width) / self.__window_shift + 1)
            for j in range(temp_num_windows):
                temp_window = []
                for k in range(len(temp_data[0, ::])):
                    temp_window.append(
                        temp_data[j*self.__window_shift:(j*self.__window_shift + self.__window_width), k] *
                        self.__window_function)
                self.__test_data_raw_windows.append(np.transpose(temp_window))
                self.__test_labels.append(self.__file_labels[i])

        self.__training_labels = np.array(self.__training_labels)
        self.__test_labels = np.array(self.__test_labels)

    def extract_and_normalize_features(self, ):

        """
        Extract features and normalize training and test data.
        :param
        :return:
        Writes to:
        Depends on:
        """

        training_features_temp = []         # features extracted
        test_features_temp = []             # ...
        training_features_std_temp = []     # features normalized
        test_features_std_temp = []         # ...

        # extract training-features from windows:
        for i in range(len(self.__training_data_raw_windows)):
            training_features_temp.append(self.extract_features(self.__training_data_raw_windows[i]))
        training_features_temp = np.array(training_features_temp)

        # extract test-features from windows:
        for i in range(len(self.__test_data_raw_windows)):
            test_features_temp.append(self.extract_features(self.__test_data_raw_windows[i]))
        test_features_temp = np.array(test_features_temp)

        # normalize training features and sort out (close to) 0-variance features:
        self.__feature_validity = np.ones([len(training_features_temp[0, ::])], dtype=bool)
        for i in range(len(training_features_temp[0, ::])):
            mue_ = np.median(training_features_temp[::, i])
            sigma_ = mad(training_features_temp[::, i])
            if sigma_ < 10**(-16):
                self.__feature_validity[i] = False
                print("(extract_and_normalize_features) 0-mean feature detected...")
                self.__std_mue.append(0)
                self.__std_sigma.append(0)
            else:
                training_features_std_temp.append((training_features_temp[::, i] - mue_)/sigma_)
                self.__std_mue.append(mue_)
                self.__std_sigma.append(sigma_)

        # normalize test-features:
        for i in range(len(test_features_temp[0, ::])):
            if self.__feature_validity[i]:
                test_features_std_temp.append((test_features_temp[::, i] - self.__std_mue[i])/self.__std_sigma[i])

        training_features_std_temp = np.array(training_features_std_temp)
        test_features_std_temp = np.array(test_features_std_temp)

        if self.__balanced_classes:

            # find maximum allowed number of samples:
            windows_per_class_train = []
            windows_per_class_test = []
            for i in range(self.__number_of_classes):
                windows_per_class_train.append(np.sum(self.__training_labels[::] == i))
                windows_per_class_test.append(np.sum(self.__test_labels[::] == i))

            windows_per_class_train = np.min(windows_per_class_train)
            windows_per_class_test = np.min(windows_per_class_test)

            temp_labels_test = []
            temp_labels_training = []

            # select balanced number of samples:
            for i in range(self.__number_of_classes):
                # select training data:
                count = 0
                for j in range(len(self.__training_labels)):
                    if self.__training_labels[j] == i and count < windows_per_class_train:
                        self.__training_data.append(training_features_std_temp[::, j])
                        temp_labels_training.append(i)
                        count += 1
                # select test data:
                count = 0
                for j in range(len(self.__test_labels)):
                    if self.__test_labels[j] == i and count < windows_per_class_test:
                        self.__test_data.append(test_features_std_temp[::, j])
                        temp_labels_test.append(i)
                        count += 1

            self.__training_labels = temp_labels_training
            self.__test_labels = temp_labels_test
        else:
            self.__training_data = training_features_std_temp.T

            self.__test_data = test_features_std_temp.T

    def prepare_data(self, ):
        self.read_data_set_from_files()
        self.split_training_test()
        self.window_data()
        self.extract_and_normalize_features()

        return self.__training_data, self.__training_labels, self.__test_data, self.__test_labels

    def classify_data(self, ):

        print(np.shape(self.__training_data))
        print(np.shape(self.__training_labels))

        # index of abnormal training data:
        index = np.random.uniform(0, 1, len(self.__training_labels))
        index[self.__training_labels == 0] = False

        self.__training_data = np.concatenate([self.__training_data[self.__training_labels == 0],
                                               self.__training_data[index < self.__oidp]], 0)
        self.__training_labels = np.concatenate([self.__training_labels[self.__training_labels == 0],
                                                 self.__training_labels[index < self.__oidp]], 0)

        print("(testClassifier) " + str(np.sum(self.__training_labels[self.__training_labels == 1])) + " abnormal "
              "training samples")

        self.__clf = svm.OneClassSVM(nu=10**(-2), kernel="rbf", gamma=0.005)
        self.__clf.fit(self.__training_data)

        occurrence_count = np.zeros(self.__number_of_classes)

        conf_mat = np.zeros([self.__number_of_classes, self.__number_of_classes])

        for i in range(len(self.__test_data)):
            # print("(testClassifier) Progress: " + str(i*100/len(self.__test_data)) + "%")
            normedFeatVec = self.__test_data[i]
            prediction = self.__clf.predict(normedFeatVec.reshape(1, -1))

            print(np.shape(self.__test_data_raw_windows[i]))
            prediction = self.live_classification(self.__test_data_raw_windows[i])

            if prediction == -1:
                prediction = 1
            else:
                prediction = 0
            occurrence_count[int(self.__test_labels[i])] += 1
            conf_mat[int(prediction), int(self.__test_labels[i])] += 1

        for i in range(self.__number_of_classes):
            try:
                conf_mat[::, i] = conf_mat[::, i] / occurrence_count[i]
            except FloatingPointError:
                print("(testClassifier) Seems like class " + str(i) + " was not trained. Consider classSort as True or "
                                                                      "smaller fraction of training data (trainFraction"
                                                                      " = " + str(self.__train_frac) + ").")
            print("(testClassifier) For class " + str(i) + " the number of test samples/windows is "
                  + str(occurrence_count[i]))

        print("(testClassifier) The overall error is " + str(100 - np.sum(np.diag(conf_mat))/self.__number_of_classes
                                                             * 100)
              + "%.")

        self.plot_conf_mat(conf_mat)

    def live_classification(self, data):

        data_temp = []
        for j in range(len(self.__selected_variables)):
            data_temp.append(data[::, self.__selected_variables[j]] * self.__window_function)
        features = self.extract_features(data_temp)
        normalized_features = []
        for i in range(len(features)):
            if self.__feature_validity[i]:
                normalized_features.append((features[::, i] - self.__std_mue[i])/self.__std_sigma[i])
        prediction = self.__clf.predict(normalized_features)
        return prediction

    @staticmethod
    def plot_data(data, begin=0, end=10**32):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for i in range(np.size(data, 0)):
            ax1.plot(data[begin:end, i], data[begin:end, i], label=str(i))
        plt.show()

    @staticmethod
    def plot_conf_mat(matrix, title_=None, precision=3, show=True):

        """
        Plot matrix and its element values
        :param matrix: Matrix (m x n) to be plotted.
        :param title_: Title of the resulting plot, when None no title is given. Default None.
        :param precision: Precision of the element values.
        :param show: Show matrix within function. Default True.
        :return: -
        """

        matrix = np.array(matrix)
        x_range = np.size(matrix[::, 0])
        y_range = np.size(matrix[0, ::])
        fig, ax = plt.subplots()
        ax.matshow(matrix, cmap=plt.cm.Blues)
        for i in range(y_range):
            for j in range(x_range):
                c = np.round(matrix[j, i], precision)
                ax.text(i, j, str(c), va='center', ha='center')
        if title_ is not None:
            plt.title(title_)
        if show:
            plt.show()


if __name__ == '__main__':
    oc_clf = One_class_svm(window_width=100,
                           window_shift=100,
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

    oc_clf.sel_var_subset(ir_vars=np.array([0, 1, 2, 3, 4, 5], dtype=int),
                          fr_vars=np.array([], dtype=int),
                          quat_axis=np.array([], dtype=int),
                          av_axis=np.array([], dtype=int),
                          la_axis=np.array([], dtype=int),
                          mag_axis=np.array([], dtype=int),
                          odo_vars=np.array([], dtype=int))
    # normal:
    oc_clf.add_data_files("2019_1_15_16_52.txt", start_index=4000, stop_index=7500, label=0)
    oc_clf.add_data_files("2019_1_15_16_52.txt", start_index=37000, stop_index=47000, label=0)
    oc_clf.add_data_files("2019_1_15_18_4_Outdoor.txt", start_index=4500, stop_index=9500, label=0)
    oc_clf.add_data_files("2019_1_15_18_4_Outdoor.txt", start_index=13000, stop_index=15000, label=0)
    oc_clf.add_data_files("2019_1_15_18_4_Outdoor.txt", start_index=18500, stop_index=20250, label=0)
    oc_clf.add_data_files("2019_1_15_18_4_Outdoor.txt", start_index=25000, stop_index=28000, label=0)
    oc_clf.add_data_files("2019_1_15_18_4_Outdoor.txt", start_index=3000, stop_index=17000, label=0)
    # not normal:
    oc_clf.add_data_files("2019_1_15_16_52.txt", start_index=9000, stop_index=13500, label=1)
    oc_clf.add_data_files("2019_1_15_16_52.txt", start_index=30600, stop_index=31600, label=1)
    oc_clf.add_data_files("2019_1_15_16_52.txt", start_index=47500, stop_index=50500, label=1)
    oc_clf.add_data_files("2019_1_15_17_7_Gehsteig.txt", start_index=19200, stop_index=20200, label=1)
    oc_clf.add_data_files("2019_1_15_17_7_Gehsteig.txt", start_index=22000, stop_index=25000, label=1)
    oc_clf.add_data_files("2019_1_15_17_7_Gehsteig.txt", start_index=0, stop_index=1000, label=1)
    oc_clf.add_data_files("2019_1_15_18_4_Outdoor.txt", start_index=0, stop_index=800, label=1)
    oc_clf.add_data_files("2019_1_15_18_4_Outdoor.txt", start_index=29200, stop_index=30500, label=1)
    oc_clf.set_window_function(functionName="tukey", alpha=0.1, plot_output=False)
    oc_clf.read_data_set_from_files(plot_output=False)
    oc_clf.split_training_test(plot_output=False)
    oc_clf.window_data()
    oc_clf.extract_and_normalize_features()
    oc_clf.classify_data()
