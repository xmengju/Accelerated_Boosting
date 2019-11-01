    
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_hastie_10_2
import matplotlib.pyplot as plt
from scipy.special import expit

from sklearn.datasets.samples_generator import make_regression
from sklearn.tree import DecisionTreeRegressor
from scipy import optimize


_MACHINE_EPSILON = np.finfo(np.float64).eps
MAXZ = 2