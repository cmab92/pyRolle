
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import DictionaryLearning
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from io import open
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def read_file(file_name):
    file_id = "/home/bonenberger/Dokumente/data/UCR_TS_Archive_2015/ECG5000/" + file_name
    content = np.loadtxt(file_id, delimiter=',')
    labels = content[::, 0]
    labels[labels != 1] = 2
    labels[labels == 1] = 0
    labels[labels == 2] = 1
    data = content[::, 1:-1]
    return data, labels


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


training_data, training_labels = read_file("ECG5000_TRAIN")
test_data, test_labels = read_file("ECG5000_TEST")

n_atoms = 5
dcty = DictionaryLearning(n_components=n_atoms, alpha=10**(-6), max_iter=20, tol=1e-12, fit_algorithm='cd',
                          transform_algorithm='omp', transform_n_nonzero_coefs=n_atoms, transform_alpha=1,
                          n_jobs=-1, verbose=False, split_sign=False, random_state=None)
print('Training data size:')
print(np.shape(training_data))
print('Test data size:')
print(np.shape(test_data))
dcty = dcty.fit(training_data[training_labels == 0])
sparse_train = dcty.transform(training_data)
print('Fitted')
sparse_test = dcty.transform(test_data)
print('Transformed')

# normalize
mue = []
sigma = []
for i in range(len(sparse_train[0, ::])):
    mue.append(np.mean(sparse_train[::, i]))
    sigma.append(np.var(sparse_train[::, i]))
for i in range(len(sparse_train[0, ::])):
    sparse_train[::, i] = (sparse_train[::, i]-mue[i])/sigma[i]
for i in range(len(sparse_test[0, ::])):
    sparse_test[::, i] = (sparse_test[::, i]-mue[i])/sigma[i]

# plot some atoms amplitudes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.cm.get_cmap("coolwarm")
plot_atoms = [0, 1, 2]
ax.scatter(sparse_train[::, plot_atoms[0]], sparse_train[::, plot_atoms[1]], sparse_train[::, plot_atoms[2]],
           c=training_labels, alpha=0.5, cmap=cmhot)
ax.scatter(sparse_test[::, plot_atoms[0]], sparse_test[::, plot_atoms[1]], sparse_test[::, plot_atoms[2]],
           c=test_labels, cmap=cmhot)

# scatter all coefficients
plt.figure()
for i in range(len(sparse_train)):
    if training_labels[i] == 0:
        plt.scatter(np.linspace(0, n_atoms - 1, n_atoms), np.transpose(sparse_train)[::, i], c='r')
    else:
        plt.scatter(np.linspace(0, n_atoms - 1, n_atoms), np.transpose(sparse_train)[::, i], c='b')
for i in range(len(sparse_train)):
    if test_labels[i] == 0:
        plt.scatter(np.linspace(0, n_atoms - 1, n_atoms), np.transpose(sparse_test)[::, i], c='r')
    else:
        plt.scatter(np.linspace(0, n_atoms - 1, n_atoms), np.transpose(sparse_test)[::, i], c='b')

# plot atoms
plt.figure()
for i in range(n_atoms):
    plt.plot(dcty.components_[i], alpha=2.0 / (i + 1))
plt.title("...the atoms")

# classification

'''
parameter_candidates = [{'C': np.linspace(-1, 2, 8), 'gamma': np.linspace(-3, 0, 10)}]
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1, cv=5)
clf.fit(training_data, training_labels)
print('Best score for training data:', clf.best_score_)
print('Best C:', clf.best_estimator_.C)
print('Best Kernel:', clf.best_estimator_.kernel)
print('Best Gamma:', clf.best_estimator_.gamma)
'''
clf = svm.SVC(kernel='rbf', C=5, gamma=0.01).fit(sparse_train, training_labels)
clf.fit(sparse_train, training_labels)

plt.matshow(sparse_train)

number_of_classes = 2
occurrence_count = np.zeros(number_of_classes)
conf_mat = np.zeros([number_of_classes, number_of_classes])

for i in range(len(sparse_test)):
    normedFeatVec = sparse_test[i]
    prediction = clf.predict(normedFeatVec.reshape(1, -1))
    occurrence_count[int(test_labels[i])] += 1
    conf_mat[int(prediction), int(test_labels[i])] += 1

print("(testClassifier) The F1-score is " + str(2*conf_mat[0, 0]/(2*conf_mat[0, 0] + conf_mat[1, 0]+conf_mat[1, 1])))
print("(testClassifier) The 'F1-score error' " + str(1 - 2*conf_mat[0, 0]/(2*conf_mat[0, 0] + conf_mat[1, 0]+conf_mat[1, 1])))

for i in range(number_of_classes):
    conf_mat[::, i] = conf_mat[::, i] / occurrence_count[i]
    print("(testClassifier) For class " + str(i) + " the number of test samples/windows is "
          + str(occurrence_count[i]))

plot_conf_mat(conf_mat)
plt.show()
