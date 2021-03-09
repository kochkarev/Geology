import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Input, concatenate
from tensorflow.keras.initializers import GlorotNormal

def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

def conv2d_block(
    inputs, 
    use_batch_norm=True, 
    filters=16, 
    kernel_size=(3,3), 
    activation='relu', 
    kernel_initializer=GlorotNormal(), 
    padding='same'):
    
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding) (inputs)
    # if use_batch_norm:
        # c = tfa.layers.GroupNormalization(groups=16)(c)
        # c = BatchNormalization()(c)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding) (c)
    # if use_batch_norm:
        # c = tfa.layers.GroupNormalization(groups=16)(c)
        # c = BatchNormalization()(c)
    return c

def custom_unet(
    input_shape,
    n_classes,
    use_batch_norm, 
    filters=16,
    n_layers=4,
    output_activation='sigmoid'): # 'sigmoid' or 'softmax'
    
    inputs = Input(input_shape)
    x = inputs   

    down_layers = []
    for _ in range(n_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm)
        down_layers.append(x)
        x = MaxPooling2D((2, 2)) (x)
        filters = filters * 2 # double the number of filters with each layer

    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm)

    for conv in reversed(down_layers):        
        filters //= 2 # decreasing number of filters with each layer 
        x = upsample_conv(filters, (2, 2), strides=(2, 2), padding='same') (x)
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm)
    
    outputs = Conv2D(n_classes, (1, 1), activation=output_activation) (x)    
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model