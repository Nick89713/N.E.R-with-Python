# import numpy as np
# from keras.models import Model
# from keras.layers import Reshape,Input, LSTM, GlobalMaxPooling2D, Embedding, Dense, TimeDistributed, Bidirectional, concatenate, SpatialDropout1D, Conv2D, PReLU, BatchNormalization
# from keras.regularizers import l2
# from keras.initializers import he_uniform, glorot_uniform, orthogonal

# from keras.src.engine.functional import Functional

from .__init__ import *
class ConvBidirectionalLSTM():
    def __init__(self,
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
    ):
        self.X_train = X_train
        self.num_of_classes = num_of_classes
        self.max_word_len = max_word_len
        self.number_of_words = number_of_words
        self.embedding_layer_output_dim = embedding_layer_output_dim
        self.num_of_conv_filters = num_of_conv_filters
        self.num_of_conv_groups = num_of_conv_groups
        self.conv_layer_regularization_strength = conv_layer_regularization_strength
        self.lstm_units = lstm_units
        self.lstm_dropout_rate = lstm_dropout_rate
        self.lstm_recurrent_dropout_rate = lstm_recurrent_dropout_rate
        self.lstm_layer_regularization_strength = lstm_layer_regularization_strength
        self.dense_layer_regularization_strength = dense_layer_regularization_strength

        self.model = self.build_model()


    def build_model(self) -> Functional:
        word_input = Input(shape = tuple([self.max_word_len]))          
        word_emebeddings = Embedding(input_dim = self.number_of_words + 2,  
                            output_dim = self.embedding_layer_output_dim,                                                 
                            input_length = self.max_word_len,             
                            mask_zero = True)(word_input)    
       
        char_input_shape = (*self.X_train.shape[1:],1)
        char_input = Input(shape = char_input_shape)
        char_input_reshaped = Reshape((*char_input_shape, 1))(char_input)
       
        convolved_images = TimeDistributed(
                                Conv2D( filters = self.num_of_conv_filters,                                           
                                        kernel_size = (3,3),
                                        strides = (1,1),
                                        padding = "same",
                                        data_format = "channels_last", 
                                        dilation_rate = 1,   
                                        groups = self.num_of_conv_groups,                                            
                                        activation = PReLU(),
                                        use_bias = True,
                                        kernel_initializer = he_uniform(),                     
                                        bias_initializer = "zeros",
                                        kernel_regularizer = l2(self.conv_layer_regularization_strength) ,                        
                                        bias_regularizer = l2(self.conv_layer_regularization_strength),
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
                                            LSTM(units = self.lstm_units,                                  
                                                activation = "tanh",                          
                                                recurrent_activation = "hard_sigmoid",
                                                use_bias = True,
                                                kernel_initializer = glorot_uniform(),        
                                                recurrent_initializer = orthogonal(),            
                                                bias_initializer = "zeros",
                                                unit_forget_bias = True,      
                                                kernel_regularizer = l2(self.lstm_layer_regularization_strength),       
                                                recurrent_regularizer = l2(self.lstm_layer_regularization_strength),   
                                                bias_regularizer = l2(self.lstm_layer_regularization_strength),         
                                                activity_regularizer = None,   
                                                kernel_constraint = None,
                                                recurrent_constraint = None,
                                                bias_constraint = None,
                                                dropout = self.lstm_dropout_rate,            
                                                recurrent_dropout = self.lstm_recurrent_dropout_rate,   
                                                seed = None,
                                                return_sequences = True,
                                                return_state = False,           
                                                go_backwards = False,           
                                                stateful = False,                
                                                unroll = True))(concatenated_embeddings)

        dense_layer_output = TimeDistributed(
                                    Dense(units = self.num_of_classes,           
                                    activation = "sigmoid",
                                    use_bias = True,
                                    kernel_initializer = "glorot_uniform",
                                    bias_initializer = "zeros",
                                    kernel_regularizer = l2(self.dense_layer_regularization_strength),
                                    bias_regularizer = l2(self.dense_layer_regularization_strength),
                                    activity_regularizer = None,
                                    kernel_constraint = None,
                                    bias_constraint = None))(BidirectionalLSTM_embeddings)
        
        model = Model(inputs = [word_input,char_input],     
                    outputs = dense_layer_output)
        return model
        


def visualize_model_history(hist: pd.DataFrame, columns: Union[List[str], str]):
    if( columns == 'all'):
        cols = hist.columns
    else:
        cols = columns 
    fig, axs = plt.subplots(1, len(cols), figsize = (10,6))
    fig.tight_layout(pad = 4, w_pad = 2)
    for i, ax in enumerate(axs):
        ax.plot(hist.index,hist[cols[i]])
        ax.axis('tight')
        ax.set_ylabel(hist.columns[i])
        ax.set_xlabel('Epochs')

        ax.yaxis.set_major_locator(MultipleLocator(0.01)) 
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

        ax.tick_params(axis = 'y', 
                       which = 'major',
                       direction = 'inout',
                       length = 8.0,
                       width = 0.5, 
                       color = 'black')
        ax.grid(linestyle = '-', color = 'b')

        # TODO add point