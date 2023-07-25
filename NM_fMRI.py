# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os, requests, tarfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import requests
import seaborn as sns

HCP_DIR = "./hcp_task"
fname = "hcp_task.tgz"
url = "https://osf.io/2y3fw/download"

if not os.path.isfile(fname):
  try:
    r = requests.get(url)
  except requests.ConnectionError:
    print("!!! Failed to download data !!!")
  else:
    if r.status_code != requests.codes.ok:
      print("!!! Failed to download data !!!")
    else:
      with open(fname, "wb") as fid:
        fid.write(r.content)
        with tarfile.open(fname) as tfile:
          # extracting file
          tfile.extractall('.')

# The data shared for NMA projects is a subset of the full HCP dataset
N_SUBJECTS = 100

# The data have already been aggregated into ROIs from the Glasser parcellation
N_PARCELS = 360

# The acquisition parameters for all tasks were identical
TR = 0.72  # Time resolution, in seconds

# The parcels are matched across hemispheres with the same order
HEMIS = ["Right", "Left"]

# Each experiment was repeated twice in each subject
RUNS   = ['LR','RL']
N_RUNS = 2

# There are 7 tasks. Each has a number of 'conditions'
# TIP: look inside the data folders for more fine-graned conditions

EXPERIMENTS = {
    'MOTOR'      : {'cond':['lf','rf','lh','rh','t','cue']},
    'WM'         : {'cond':['0bk_body','0bk_faces','0bk_places','0bk_tools','2bk_body','2bk_faces','2bk_places','2bk_tools']},
    'EMOTION'    : {'cond':['fear','neut']},
    'GAMBLING'   : {'cond':['loss','win']},
    'LANGUAGE'   : {'cond':['math','story']},
    'RELATIONAL' : {'cond':['match','relation']},
    'SOCIAL'     : {'cond':['ment','rnd']}
}


subjects = np.loadtxt(os.path.join(HCP_DIR, "subjects_list.txt"))
subjects = subjects.astype(int)
regions = np.load(os.path.join(HCP_DIR, 'regions.npy')).T
region_info = dict(
    name=regions[0].tolist(),
    network=regions[1],
    hemi=['Right']*int(N_PARCELS/2) + ['Left']*int(N_PARCELS/2),
)

# import sqlite3
# conn = sqlite3.connect(os.path.join(HCP_DIR, "state.db"))
# cursor = conn.cursor()

def load_single_timeseries(subject, experiment, run, remove_mean=True):
  """Load timeseries data for a single subject and single run.

  Args:
    subject (str):      subject ID to load
    experiment (str):   Name of experiment
    run (int):          (0 or 1)
    remove_mean (bool): If True, subtract the parcel-wise mean (typically the mean BOLD signal is not of interest)

  Returns
    ts (n_parcel x n_timepoint array): Array of BOLD data values

  """
  bold_run  = RUNS[run]
  bold_path = f"{HCP_DIR}/subjects/{subject}/{experiment}/tfMRI_{experiment}_{bold_run}"
  bold_file = "data.npy"
  ts = np.load(f"{bold_path}/{bold_file}")
  if remove_mean:
    ts -= ts.mean(axis=1, keepdims=True)
  return ts


def load_evs(subject, experiment, run):
  """Load EVs (explanatory variables) data for one task experiment.

  Args:
    subject (str): subject ID to load
    experiment (str) : Name of experiment
    run (int): 0 or 1

  Returns
    evs (list of lists): A list of frames associated with each condition

  """
  frames_list = []
  task_key = f'tfMRI_{experiment}_{RUNS[run]}'
  for cond in EXPERIMENTS[experiment]['cond']:
    ev_file  = f"{HCP_DIR}/subjects/{subject}/{experiment}/{task_key}/EVs/{cond}.txt"
    ev_array = np.loadtxt(ev_file, ndmin=2, unpack=True)
    ev       = dict(zip(["onset", "duration", "amplitude"], ev_array))
    # Determine when trial starts, rounded down
    start = np.floor(ev["onset"] / TR).astype(int) - 1
    # Use trial duration to determine how many frames to include for trial
    duration = np.ceil(ev["duration"] / TR).astype(int)

    frames = []
    for index, _ in enumerate(start):
      _start = start[index]

      # HACK: Forces all block to be 15 frames in length. In reality all but the last are 25 frames.
      _duration =  15 # duration[index]

      # if last block
      #if (_start + _duration >= 175):
      #  print("last block")
      #  _duration = 176 - _start
      #  print(_duration)

      _frames = list(range(_start, _start + _duration))
      frames.append(_frames)

    # Take the range of frames that correspond to this specific trial
    # frames = [s + np.arange(0, d) for s, d in zip(start, duration)]
    frames_list.append(frames)

  return frames_list

my_exp = 'EMOTION'
my_subj = subjects[1]
my_run = 1

data = load_single_timeseries(subject=my_subj,
                              experiment=my_exp,
                              run=my_run,
                              remove_mean=True)
print(data.shape)

evs = load_evs(subject=my_subj, experiment=my_exp, run=my_run)

# for i in range(len(subjects)):
#     df = load_single_timeseries(subject=subjects[i],
#                                 experiment=my_exp,
#                                 run=my_run,
#                                 remove_mean=True)
    
#     # to load explanatory variable
evs = load_evs(subject=subjects[0], experiment=my_exp, run=my_run)

# we need a little function that averages all frames from any given condition
def average_frames(data, evs, experiment, cond):
  idx = EXPERIMENTS[experiment]['cond'].index(cond)
  return np.mean(np.concatenate([np.mean(data[:, evs[idx][i]], axis=1, keepdims=True) for i in range(len(evs[idx]))], axis=-1), axis=1)


fear_activity = average_frames(data, evs, my_exp, 'fear')
neut_activity = average_frames(data, evs, my_exp, 'neut')
contrast = fear_activity - neut_activity  # difference between left and right hand movement

# Plot activity level in each ROI for both conditions
plt.plot(fear_activity,label='fear')
plt.plot(neut_activity,label='neut')
plt.xlabel('ROI')
plt.ylabel('activity')
plt.legend()
plt.show()

df = pd.DataFrame({'fear_activity' : fear_activity,
                   'neut_activity' : neut_activity,
                   'regions' : region_info['name'],
                   'hemi' : region_info['hemi']})

fig, (ax1, ax2) = plt.subplots(1, 2)
sns.barplot(y='regions', x='fear_activity', data=df, hue='hemi',ax=ax1)
sns.barplot(y='regions', x='neut_activity', data=df, hue='hemi',ax=ax2)
plt.show()


timeseries_emotion = []
for subject in subjects:
  ts_concat = load_single_timeseries(subject=my_subj,
                                     experiment=my_exp,
                                     run=my_run,
                                     remove_mean=True)
  timeseries_emotion.append(ts_concat)


  fc = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
for sub, ts in enumerate(timeseries_emotion):
  fc[sub] = np.corrcoef(ts)

group_fc = fc.mean(axis=0)

plt.figure()
plt.imshow(group_fc, interpolation="none", cmap="bwr", vmin=-1, vmax=1)
plt.colorbar()
plt.show()

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import time

x_tot = []
y_tot = []

for i in range(len(subjects)):
    df = load_single_timeseries(subject=subjects[i],
                                experiment=my_exp,
                                run=my_run,
                                remove_mean=True)
    
    # to load explanatory variable
    evs = load_evs(subject=subjects[i], experiment=my_exp, run=my_run)
    # to averages all frames from any given condition
    fa = average_frames(df, evs, my_exp, 'fear')
    na = average_frames(df, evs, my_exp, 'neut')
    
    x_tot.append(fa)
    y_tot.append(1)
    x_tot.append(na)
    y_tot.append(0)


X = np.array(x_tot)
y = np.array(y_tot)

# # take just 20 data
# X_s = X[:20, ]
# y_s = y[:20, ]


print(X.shape, y.shape)

# Assuming 'X' is your feature array with shape (200, 360)
# Assuming 'y' is your label array with shape (200,)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


# Create and train the SVC classifier
start_time = time.time()
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time
print("Training time:", training_time, "seconds")


# Make predictions on test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)


import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create and train the XGBoost classifier
xgb_classifier = xgb.XGBClassifier(
    objective='binary:logistic',  # For binary classification tasks
    learning_rate=0.1,            # Learning rate (step size) for boosting process
    max_depth=3,                  # Maximum depth of each tree
    n_estimators=100,             # Number of boosting rounds (trees)
    random_state=42               # Set a random seed for reproducibility
)
xgb_classifier.fit(X, y)

# Make predictions on the test set
y_pred = xgb_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)

import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
