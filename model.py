from tensorflow import keras
import keras
from keras.models import Model
from keras.layers import Input, Lambda, BatchNormalization, Conv2D, Activation
from keras.layers import add, dot, concatenate
from morphological_layers import *


def _morph_unit(inputs, filters=4, kernel=(3, 3)):
    erosion = Erosion2D(
        filters, kernel, padding="same", strides=(1, 1), kernel_initializer="he_normal"
    )(inputs)
    dilation = Dilation2D(
        filters, kernel, padding="same", strides=(1, 1), kernel_initializer="he_normal"
    )(inputs)
    xc = Concatenate()([erosion, dilation])
    xc = Conv2D(
        2 * filters,
        kernel,
        padding="same",
        activation="linear",
        kernel_initializer="he_normal",
    )(xc)
    return xc


def _bn_relu_conv_block(inputs, filters, kernel=(3, 3), stride=(1, 1)):
    x = Conv2D(
        filters,
        kernel,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        strides=stride,
    )(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    return x


def _grouped_morph_block(input, grouped_channels, cardinality):
    group_morph_list = []
    for c in range(cardinality):
        x = Lambda(
            lambda z: z[:, :, :, c * grouped_channels : (c + 1) * grouped_channels]
        )(input)
        x = _morph_unit(x, grouped_channels)
        group_morph_list.append(x)
    merged_output = concatenate(group_morph_list, axis=-1)
    merged_output = BatchNormalization(axis=-1)(merged_output)
    return merged_output


def _dual_path_block(
    input,
    pointwise_filters_a,
    grouped_conv_filters_b,
    pointwise_filters_c,
    filter_increment,
    cardinality,
    block_type="normal",
):
    grouped_channels = int(grouped_conv_filters_b / cardinality)
    inputs = concatenate(input, axis=-1) if isinstance(input, list) else input
    stride = (1, 1)

    if block_type == "projection":
        projection = True
    elif block_type == "normal":
        projection = False
    else:
        raise ValueError('"block_type" must be either "projection" or "normal"')

    if projection:
        projection_path = _bn_relu_conv_block(
            inputs,
            filters=pointwise_filters_c + 2 * filter_increment,
            kernel=(1, 1),
            stride=stride,
        )
        input_residual_path = Lambda(lambda z: z[:, :, :, :pointwise_filters_c])(
            projection_path
        )
        input_dense_path = Lambda(lambda z: z[:, :, :, pointwise_filters_c:])(
            projection_path
        )
    else:
        input_residual_path = input[0]
        input_dense_path = input[1]

    x = _bn_relu_conv_block(inputs, filters=pointwise_filters_a, kernel=(1, 1))
    x = _grouped_morph_block(
        x, grouped_channels=grouped_channels, cardinality=cardinality
    )
    x = _bn_relu_conv_block(
        x, filters=pointwise_filters_c + filter_increment, kernel=(1, 1)
    )

    output_residual_path = Lambda(lambda z: z[:, :, :, :pointwise_filters_c])(x)
    output_dense_path = Lambda(lambda z: z[:, :, :, pointwise_filters_c:])(x)

    residual_path = add([input_residual_path, output_residual_path])
    dense_path = concatenate([input_dense_path, output_dense_path], axis=-1)
    return [residual_path, dense_path]


def _decoder_block(inputs, skip_connection, filters):
    upsampled_input = Conv2DTranspose(
        filters=int(ip.shape[-1]), kernel_size=(3, 3), strides=(2, 2), padding="same"
    )(inputs)
    concatenated_output = Concatenate()([upsampled_input, skip_connection])
    concatenated_output = _morph_block(concatenated_output, filters)
    return concatenated_output


def _morph_block(inputs, num_filters, filter_increment=12, cardinality=6):
    x = inputs
    x = _dual_path_block(
        x,
        num_filters,
        num_filters,
        num_filters,
        filter_increment,
        cardinality,
        "projection",
    )
    x = _dual_path_block(
        x,
        num_filters,
        num_filters,
        num_filters,
        filter_increment,
        cardinality,
        "normal",
    )
    x = concatenate(x, axis=-1)
    return x


def DPM_UNet(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    morph_ip = _morph_unit(ip, 6)

    E1 = _morph_block(morph_ip, 12)
    pool1 = AveragePooling2D(pool_size=(2, 2))(E1)

    E2 = _morph_block(pool1, 12 * 2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(E2)

    E3 = _morph_block(pool2, 12 * 3)
    pool3 = AveragePooling2D(pool_size=(2, 2))(E3)

    C = _morph_block(pool3, 12 * 4)

    D3 = _decoder_block(C, E3, 12 * 3)
    D2 = _decoder_block(D3, E2, 12 * 2)
    D1 = _decoder_block(D2, E1, 12)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(D1)
    return Model(inputs, outputs)
