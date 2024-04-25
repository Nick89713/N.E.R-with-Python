import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List, Union
from keras.models import Model
from keras.layers import Reshape,Input, LSTM, GlobalMaxPooling2D, Embedding, Dense, TimeDistributed, Bidirectional, concatenate, SpatialDropout1D, Conv2D, PReLU, BatchNormalization
from keras.regularizers import l2
from keras.initializers import he_uniform, glorot_uniform, orthogonal
from keras.src.engine.functional import Functional

from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator