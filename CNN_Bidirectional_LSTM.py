import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Reshape,Input, Flatten, LSTM, GlobalMaxPooling2D, Embedding, Dense, TimeDistributed, Bidirectional, concatenate, SpatialDropout1D, Conv2D, PReLU, BatchNormalization
from keras.regularizers import l2
from keras.initializers import he_uniform, glorot_uniform, orthogonal
from keras.callbacks import EarlyStopping
from keras.metrics import F1Score
from keras.src.engine.functional import Functional

from time import time

# TODO: handle as regression problem with multiple target variables, where each variable's values are discrete (classes)
class SentenceGetter(object):
    
    def __init__(self, data: pd.DataFrame):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self) -> List[tuple]:
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

        
if __name__ == "__main__":
# region - Load Data
    filepath = "datasets/ner_dataset/"
    dataset = "ner_datasetreference.csv"
    data = pd.read_csv(filepath + dataset, encoding="latin1").fillna(method="ffill")
# endregion - Load Data

# region - Preprocessing
    # extract unique words
    words = list(set(data["Word"].values))
    n_words = len(words)

    # extract unique tags
    tags = list(set(data["Tag"].values))
    n_tags = len(tags)
  
    # extract sentences
    getter = SentenceGetter(data)
    s = getter.get_next()
    sentences = getter.sentences

    # create dictionary of words
    word2idx = {word: indx + 2 for indx, word in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0
    idx2word = {indx: word for word, indx in word2idx.items()}

    # create dictionary of tags
    tag2idx = {tag: indx + 1 for indx, tag in enumerate(tags)}
    tag2idx["PAD"] = 0
    idx2tag = {indx: tag for tag, indx in tag2idx.items()}

    # pad sequences, using 'max_len' as the dimension
    # of each word
    max_len = 75 
    X_word = [[word2idx[word_tuple[0]] for word_tuple in sentence] for sentence in sentences]                                            
    X_word = pad_sequences(maxlen=max_len, 
                           sequences=X_word, 
                           value=word2idx["PAD"], 
                           padding='post', 
                           truncating='post')

    # create dictionary of characters
    chars = set([char for word in words for char in word])
    n_chars = len(chars)
    char2idx = {char: indx + 2 for indx, char in enumerate(chars)}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0

    max_len_char = 10
    X_char = np.zeros((len(sentences), max_len, max_len_char), dtype=np.int32)
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            if j == max_len: 
                break
            for k, char in enumerate(word[0]):
                if k == max_len_char:  
                    break
                char_idx = char2idx.get(char, char2idx["PAD"])
                X_char[i, j, k] = char_idx

    # mapping and padding for our tag sequence.
    y = [[tag2idx[word_tuple[2]] for word_tuple in sentence] for sentence in sentences]

    y = pad_sequences(maxlen = max_len, 
                      sequences = y, 
                      value = tag2idx["PAD"], 
                      padding = 'post', 
                      truncating = 'post')
    
    # pairs_array = [[(index, value) for index, value in enumerate(row)] for row in y]
    # flattened_pairs = [pair for row in pairs_array for pair in row]
    # unique_pairs = set(flattened_pairs)
    # pair_to_integer = {pair: i for i, pair in enumerate(unique_pairs)}
    # integer_pairs_array = [[pair_to_integer[pair] for pair in row] for row in pairs_array]

    # train-test split at word level & char. level. Targets remain the same!
    X_word = X_word.astype(np.float64)
    X_char = X_char.astype(np.float64)
    y = y.astype(np.float64)

    X_word_tr, X_word_te, y_tr, y_te = train_test_split(X_word, y, 
                                                        test_size = 0.1, 
                                                        random_state = 2018)
    X_char_tr, X_char_te, _, _ = train_test_split(X_char, y, 
                                                  test_size = 0.1, 
                                                  random_state = 2018)

    # fix shapes
    X_char_tr = np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char))
    y_tr =  np.array(y_tr).reshape(len(y_tr), max_len, 1)

# endregion - Preprocessing
    

    def ConvBidirectionalLSTM(
        X_train : np.ndarray,
        num_of_classes: int, 
        max_word_len : int,
        number_of_words : int,
        embedding_layer_output_dim : int,
        num_of_conv_filters : int,
        num_of_conv_groups: int,
        conv_layer_regularization_strength : float,
        lstm_units : int,
        lstm_dropout_rate : float,
        lstm_recurrent_dropout_rate : float,
        lstm_layer_regularization_strength : float,
        dense_layer_regularization_strength : float
    ) -> Functional:
        word_input = Input(shape = tuple([max_word_len]))          
        word_emebeddings = Embedding(input_dim = number_of_words + 2,  
                            output_dim = embedding_layer_output_dim,                                                 
                            input_length = max_word_len,             
                            mask_zero = True)(word_input)    
       
        char_input_shape = (*X_train.shape[1:],1)
        char_input = Input(shape = char_input_shape)
        char_input_reshaped = Reshape((*char_input_shape, 1))(char_input)
       
        convolved_images = TimeDistributed(
                                Conv2D( filters = num_of_conv_filters,                                           
                                        kernel_size = (3,3),
                                        strides = (1,1),
                                        padding = "same",
                                        data_format = "channels_last", 
                                        dilation_rate = 1,   
                                        groups = num_of_conv_groups,                                            
                                        activation = PReLU(),
                                        use_bias = True,
                                        kernel_initializer = he_uniform(),                     
                                        bias_initializer = "zeros",
                                        kernel_regularizer = l2(conv_layer_regularization_strength) ,                        
                                        bias_regularizer = l2(conv_layer_regularization_strength),
                                        activity_regularizer = None,                           
                                        kernel_constraint = None,
                                        bias_constraint = None))(char_input_reshaped)
        batch_normalized_images = BatchNormalization()(convolved_images)
        max_pooled_embeddings = TimeDistributed(
                                        GlobalMaxPooling2D(data_format = "channels_last",
                                                        keepdims = False))(batch_normalized_images)
      
        concatenated_embeddings = concatenate([word_emebeddings, max_pooled_embeddings])  
        concatenated_embeddings = SpatialDropout1D(0.3)(concatenated_embeddings)          
                                            
        BidirectionalLSTM_embeddings = Bidirectional(                                 
                                            LSTM(units = lstm_units,                                  
                                                activation = "tanh",                          
                                                recurrent_activation = "hard_sigmoid",
                                                use_bias = True,
                                                kernel_initializer = glorot_uniform(),        
                                                recurrent_initializer = orthogonal(),            
                                                bias_initializer = "zeros",
                                                unit_forget_bias = True,      
                                                kernel_regularizer = l2(lstm_layer_regularization_strength),       
                                                recurrent_regularizer = l2(lstm_layer_regularization_strength),   
                                                bias_regularizer = l2(lstm_layer_regularization_strength),         
                                                activity_regularizer = None,   
                                                kernel_constraint = None,
                                                recurrent_constraint = None,
                                                bias_constraint = None,
                                                dropout = lstm_dropout_rate,            
                                                recurrent_dropout = lstm_recurrent_dropout_rate,   
                                                seed = None,
                                                return_sequences = True,
                                                return_state = False,           
                                                go_backwards = False,           
                                                stateful = False,                
                                                unroll = True))(concatenated_embeddings)
        # flattened_embeddings = Flatten()(BidirectionalLSTM_embeddings)

        dense_layer_output = TimeDistributed(
                                    Dense(units = num_of_classes,           
                                    activation = "sigmoid",
                                    use_bias = True,
                                    kernel_initializer = "glorot_uniform",
                                    bias_initializer = "zeros",
                                    kernel_regularizer = l2(dense_layer_regularization_strength),
                                    bias_regularizer = l2(dense_layer_regularization_strength),
                                    activity_regularizer = None,
                                    kernel_constraint = None,
                                    bias_constraint = None))(BidirectionalLSTM_embeddings)
        
        model = Model(inputs = [word_input,char_input],     
                    outputs = dense_layer_output)
        
        return model
    
    model = ConvBidirectionalLSTM(  X_train = X_char_tr,
                                    num_of_classes = n_tags + 1,
                                    max_word_len = max_len,
                                    number_of_words = n_words,
                                    embedding_layer_output_dim = 20,
                                    num_of_conv_filters = 20,
                                    num_of_conv_groups = 1,
                                    conv_layer_regularization_strength = 0.01,
                                    lstm_units = 50,
                                    lstm_dropout_rate = 0.3,
                                    lstm_recurrent_dropout_rate = 0.3,
                                    lstm_layer_regularization_strength = 0.3,
                                    dense_layer_regularization_strength = 0.3)
    model.compile(optimizer = "adam", 
                  loss = "sparse_categorical_crossentropy", # TODO fix loss function
                  metrics = ["acc"] )                                  
    # model.summary()
   


    
# region - Train                                                             # TODO: hyperparameters for fit
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(
                    x = [ X_word_tr, X_char_tr],
                    y = y_tr,
                    shuffle = True,
                    initial_epoch = 0,
                    epochs = 10,                          # In each epoch, iterate over all training samples, performing ceil(N/B) iterations.             
                    batch_size = 32,                      # In each iteration, process B samples, where B = batch_size
                    validation_split = 0.1,               # Percent of training data used for validation during each epoch
                    verbose = 1,
                    steps_per_epoch = None,               # process all training samples per epoch
                    validation_steps = None,              # process all available validation data in each epoch
                    validation_batch_size = None,         # use the same batch size for both  training and validation phases
                    validation_freq = 1,                  # run validation every 1 epoch
                    callbacks = [early_stopping])
    hist = pd.DataFrame(history.history)

    plt.style.use("ggplot")
    plt.figure(figsize=(12,12))
    plt.plot(hist["acc"])
    plt.plot(hist["val_acc"])
    plt.show()
# endregion - Train
    
# region - Predict
    y_pred = model.predict([X_word_te, np.array(X_char_te).reshape((len(X_char_te), max_len, max_len_char))])
    i = 1925
    p = np.argmax(y_pred[i], axis=-1)
    print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
    print(30 * "=")
    for w, t, pred in zip(X_word_te[i], y_te[i], p):
        if w != 0:
            print("{:15}: {:5} {}".format(idx2word[w], idx2tag[t], idx2tag[pred]))
# endregion - Predict





