from __future__ import print_function
import os
import re
import traceback
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from tensorflow.python.training import moving_averages
from tensorflow.python.client import device_lib
from pybh import log_utils, utils


logger = log_utils.get_logger("pybh.tf_utils")


def get_available_device_names():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def get_available_cpu_names():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']


def get_available_gpu_names():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def _get_device_ids(device_names):
    device_id_regex = re.compile(".+:(\d)")
    device_ids = []
    for device_name in device_names:
        match = device_id_regex.match(device_name)
        device_id = int(match.group(1))
        device_ids.append(device_id)
    return device_ids


def get_available_cpu_ids():
    cpu_names = get_available_cpu_names()
    cpu_ids = _get_device_ids(cpu_names)
    return cpu_ids


def get_available_gpu_ids():
    gpu_names = get_available_gpu_names()
    gpu_ids = _get_device_ids(gpu_names)
    return gpu_ids
    #gpu_id_regex = re.compile("(/gpu:(\d))|(/device:GPU:(\d))", re.IGNORECASE)
    #gpu_ids = []
    #for gpu_name in gpu_names:
    #    match = gpu_id_regex.match(gpu_name)
    #    gpu_id = int(match.group(1))
    #    gpu_ids.append(gpu_id)
    #return gpu_ids


def cpu_device_name(index=0):
    cpu_names = get_available_cpu_names()
    cpu_ids = _get_device_ids(cpu_names)
    cpu_id = list(filter(lambda x: x[1] == index, zip(cpu_names, cpu_ids)))
    #assert len(cpu_id) > 0, "Could not find a CPU device with index 0"
    if len(cpu_id) == 0:
        return None
    cpu_name = cpu_id[0][0]
    return cpu_name
    #return "/cpu:0"


def gpu_device_name(index=0):
    gpu_names = get_available_gpu_names()
    gpu_ids = _get_device_ids(gpu_names)
    gpu_id = list(filter(lambda x: x[1] == index, zip(gpu_names, gpu_ids)))
    #assert len(gpu_id) > 0, "Could not find a GPU device with index 0"
    if len(gpu_id) == 0:
        return None
    gpu_name = gpu_id[0][0]
    return gpu_name
    #return "/gpu:{:d}".format(index)


def tf_device_name(device_id=0):
    if device_id >= 0:
        return gpu_device_name(device_id)
    elif device_id == -1:
        return cpu_device_name()
    else:
        raise NotImplementedError("Unknown device id: {}".format(device_id))


def get_activation_function_by_name(name, default):
    if name is None:
        return default
    name = name.lower()
    if name == "relu":
        return tf.nn.relu
    elif name == "elu":
        return tf.nn.elu
    elif name == "softplus":
        return tf.nn.softplus
    elif name == "tanh":
        return tf.nn.tanh
    elif name == "sigmoid":
        return tf.nn.sigmoid
    elif name == "identity":
        return tf.identity
    elif name == "none":
        return None
    else:
        raise NotImplementedError("Unknown activation function: {}".format(name))


def get_optimizer_by_name(name, default):
    if name is None:
        return default
    name = name.lower()
    if name == "adam":
        return tf.train.AdamOptimizer
    elif name == "sgd":
        return tf.train.GradientDescentOptimizer
    elif name == "rmsprop":
        return tf.train.RMSPropOptimizer
    elif name == "adadelta":
        return tf.train.AdadeltaOptimizer
    elif name == "momentum":
        return tf.train.MomentumOptimizer
    elif name == "adagrad":
        return tf.train.AdagradOptimizer
    else:
        raise NotImplementedError("Unknown optimizer: {}".format(name))


def get_weights_initializer_by_name(name, default):
    if name is None:
        return default
    name = name.lower()
    if name == "relu_uniform":
        return relu_uniform_weights_initializer
    elif name == "relu_normal":
        return relu_normal_weights_initializer
    elif name == "xavier_uniform":
        return xavier_uniform_weights_initializer
    elif name == "xavier_normal":
        return xavier_normal_weights_initializer
    else:
        raise NotImplementedError("Unknown weights initializer: {}".format(name))


def tensor_size(var):
    """Return number of elements of tensor"""
    return np.prod(var.get_shape().as_list())


def tensor_bytes(var):
    """Return number of bytes of tensor"""
    return tensor_size(var) * var.dtype.size


def flatten_tensor(x):
    """Fuse all dimension"""
    return tf.reshape(x, [-1])


def flatten_batch(x):
    """Fuse all but the first dimension, assuming that we're dealing with a batch"""
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def reshape_batch_flat_to_1d(x):
    """Insert a dimension of size 1 before the filter dimension for a flat batched input.
    I.e. if input has shape [N, C] the output has shape [N, 1, C]."""
    # return tf.reshape(x, [-1, 1, int(x.get_shape()[-1])])
    return tf.expand_dims(x, -2)


def reshape_batch_1d_to_2d(x):
    """Insert a dimension of size 1 before the filter dimension for a 1d batched input.
    I.e. if input has shape [N, X, C] the output has shape [N, X, 1, C]."""
    # return tf.reshape(x, [-1, int(x.get_shape()[1]), 1, int(x.get_shape()[-1])])
    return tf.expand_dims(x, -2)


def reshape_batch_2d_to_3d(x):
    """Insert a dimension of size 1 before the filter dimension for a 2d batched input.
    I.e. if input has shape [N, X, Y, C] the output has shape [N, X, Y, 1, C]."""
    # return tf.reshape(x, [-1, int(x.get_shape()[1]), int(x.get_shape()[2]), 1, int(x.get_shape()[-1])])
    return tf.expand_dims(x, -2)


def reshape_batch_3d_to_2d(x):
    """Fuse the last two 3d-dimensions to a single dimension.
    I.e. if input has shape [N, X, Y, Z, C] the output has shape [N, X, Y * Z, C]."""
    return tf.reshape(x, [-1, int(x.get_shape()[1]), np.prod(x.get_shape().as_list()[2:-1]), int(x.get_shape()[-1])])


def reshape_batch_2d_to_1d(x):
    """Fuse the last two 2d-dimensions to a single dimension.
    I.e. if input has shape [N, X, Y, C] the output has shape [N, X * Y, C]."""
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:-1]), int(x.get_shape()[-1])])


def reshape_batch_1d_to_flat(x):
    """Fuse the last two 1d-dimensions to a single dimension.
    I.e. if input has shape [N, X, C] the output has shape [N, X * C]."""
    # return tf.reshape(x, [-1, 1, int(x.get_shape()[-1])])
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def reshape_fuse_dimension(x, first_axis):
    """Fuse the axis `first_axis` and the next one to a single dimension."""
    x_shape = x.get_shape().as_list()
    new_shape = x_shape[:first_axis] + [x_shape[first_axis] * x_shape[first_axis + 1]] + x_shape[first_axis+2:]
    return tf.reshape(x, new_shape)


def batch_norm_custom(
        inputs,
        decay=0.999,
        center=True,
        scale=False,
        epsilon=0.001,
        updates_collections=tf.GraphKeys.UPDATE_OPS,
        is_training=True,
        reuse=None,
        variables_collections=None,
        trainable=True,
        zero_debias_moving_mean=False,
        variables_on_cpu=False,
        scope="bn"):
    if scope is None:
        scope = tf.get_variable_scope()

    params_shape = [inputs.shape[-1]]
    dtype = inputs.dtype

    with tf.variable_scope(scope, reuse=reuse):

        # Allocate parameters for the beta and gamma of the normalization.
        trainable_beta = trainable and center
        beta = get_tf_variable("beta", shape=params_shape, dtype=dtype,
                               initializer=tf.zeros_initializer(), trainable=trainable_beta,
                               collections=variables_collections, variable_on_cpu=variables_on_cpu)

        trainable_gamma = trainable and scale
        gamma = get_tf_variable("gamma", shape=params_shape, dtype=dtype,
                                initializer=tf.ones_initializer(), trainable=trainable_gamma,
                                collections=variables_collections, variable_on_cpu=variables_on_cpu)

        # Moving mean and variance
        moving_mean = get_tf_variable(
            "moving_mean",
            shape=params_shape,
            dtype=dtype,
            initializer=tf.zeros_initializer(),
            trainable=False,
            collections=variables_collections,
            variable_on_cpu=variables_on_cpu)
        moving_variance = get_tf_variable(
            "moving_variance",
            shape=params_shape,
            dtype=dtype,
            initializer=tf.ones_initializer(),
            trainable=False,
            collections=variables_collections,
            variable_on_cpu=variables_on_cpu)

    def _fused_batch_norm_training():
        return tf.nn.fused_batch_norm(
            inputs, gamma, beta, epsilon=epsilon)

    def _fused_batch_norm_inference():
        return tf.nn.fused_batch_norm(
            inputs,
            gamma,
            beta,
            mean=moving_mean,
            variance=moving_variance,
            epsilon=epsilon,
            is_training=False)

    if is_training:
        outputs, mean, variance = _fused_batch_norm_training()
    else:
        outputs, mean, variance = _fused_batch_norm_inference()

    need_updates = is_training
    if need_updates:
        if updates_collections is None:
            def _force_updates():
                """Internal function forces updates moving_vars if is_training."""
                update_moving_mean = moving_averages.assign_moving_average(
                    moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
                update_moving_variance = moving_averages.assign_moving_average(
                    moving_variance, variance, decay, zero_debias=False)
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(outputs)
            outputs = _force_updates()
        else:
            def _delay_updates():
                """Internal function that delay updates moving_vars if is_training."""
                update_moving_mean = moving_averages.assign_moving_average(
                    moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
                update_moving_variance = moving_averages.assign_moving_average(
                    moving_variance, variance, decay, zero_debias=False)
                return update_moving_mean, update_moving_variance
            update_mean, update_variance = _delay_updates()
            tf.add_to_collection(updates_collections, update_mean)
            tf.add_to_collection(updates_collections, update_variance)

    return outputs


def batch_norm_on_flat(inputs, *args, **kwargs):
    original_shape = tf.shape(inputs)
    inputs = reshape_batch_flat_to_1d(inputs)
    outputs = batch_norm_on_1d(inputs, *args, **kwargs)
    outputs = tf.reshape(outputs, original_shape)
    return outputs


def batch_norm_on_1d(inputs, *args, **kwargs):
    original_shape = tf.shape(inputs)
    inputs = reshape_batch_1d_to_2d(inputs)
    outputs = batch_norm_on_2d(inputs, *args, **kwargs)
    outputs = tf.reshape(outputs, original_shape)
    return outputs


def batch_norm_on_2d(inputs, *args, **kwargs):
    variables_on_cpu = kwargs.get("variables_on_cpu", False)
    if "variables_on_cpu" in kwargs:
        del kwargs["variables_on_cpu"]
    if variables_on_cpu:
        kwargs["variables_on_cpu"] = variables_on_cpu
        return batch_norm_custom(inputs, *args, **kwargs)
    return batch_norm_custom(inputs, *args, **kwargs)


def batch_norm_on_3d(inputs, *args, **kwargs):
    original_shape = tf.shape(inputs)
    inputs = reshape_batch_3d_to_2d(inputs)
    outputs = batch_norm_on_2d(inputs, *args, **kwargs)
    outputs = tf.reshape(outputs, original_shape)
    return outputs
    # Use tensorflow batch norm implementation
    # original_shape = inputs.shape
    # outputs = tf_layers.batch_norm(reshape_batch_3d_to_2d(inputs), *args, **kwargs)
    # outputs = tf.reshape(outputs, original_shape)
    # return outputs


def xavier_uniform_weights_initializer():
    # According to
    # Xavier Glorot and Yoshua Bengio (2010):
    # Understanding the difficulty of training deep feedforward neural networks.
    # International conference on artificial intelligence and statistics.
    # w_bound = np.sqrt(6. / (fan_in + fan_out))
    # return tf.random_uniform_initializer(-w_bound, w_bound)
    return tf_layers.xavier_initializer(uniform=True)


def xavier_normal_weights_initializer():
    # According to
    # Xavier Glorot and Yoshua Bengio (2010):
    # Understanding the difficulty of training deep feedforward neural networks.
    # International conference on artificial intelligence and statistics.
    # w_bound = np.sqrt(2. / (fan_in + fan_out))
    # return tf.random_normal_initializer(-w_bound, w_bound)
    return tf_layers.xavier_initializer(uniform=False)


def relu_uniform_weights_initializer():
    # According to https://arxiv.org/pdf/1502.01852.pdf (arXiv:1502.01852)
    # with modification
    # w_bound = np.sqrt(6. / fan_in)
    # return tf.random_uniform_initializer(-w_bound, w_bound)
    return tf_layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG', uniform=True)


def relu_normal_weights_initializer():
    # According to https://arxiv.org/pdf/1502.01852.pdf (arXiv:1502.01852)
    # with modification
    # w_bound = np.sqrt(2. / fan_in)
    # return tf.random_normal_initializer(-w_bound, w_bound)
    return tf_layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG', uniform=False)


def get_tf_variable(name, shape=None, dtype=None, initializer=None, collections=None, variable_on_cpu=False, **kwargs):
    if variable_on_cpu:
        with tf.device(cpu_device_name()):
            return tf.get_variable(name, shape, dtype, initializer, collections, **kwargs)
    else:
        return tf.get_variable(name, shape, dtype, initializer, collections, **kwargs)


def conv1d(x, num_filters, filter_size=(3,), stride=(1,),
           activation_fn=tf.nn.relu, add_bias=False, use_batch_norm=True,
           batch_norm_after_activation=False, is_training=True,
           weights_initializer=None, padding="SAME",
           dtype=tf.float32, name=None, variables_on_cpu=False, collections=None,
           regularized_variables_collection=None,
           return_pre_activation=False):
    if use_batch_norm:
        assert(not add_bias)

    if name is None:
        name = tf.get_variable_scope()
    with tf.variable_scope(name):
        filter_shape = [filter_size[0], int(x.get_shape()[3]), num_filters]

        if weights_initializer is None:
            if activation_fn == tf.nn.relu or activation_fn == tf.nn.elu:
                weights_initializer = relu_normal_weights_initializer()
            else:
                weights_initializer = xavier_uniform_weights_initializer()

        w = get_tf_variable("weights", filter_shape, dtype, initializer=weights_initializer,
                            collections=collections, variable_on_cpu=variables_on_cpu)
        if regularized_variables_collection is not None:
            tf.add_to_collection(regularized_variables_collection, w)
        print("{}: {}".format(w.name, w.shape))
        conv_out = tf.nn.conv1d(x, w, stride[0], padding, name="conv1d_out")

        if add_bias:
            b = get_tf_variable("biases", [1, 1, num_filters], dtype, initializer=tf.zeros_initializer(),
                                collections=collections, variable_on_cpu=variables_on_cpu)
            if regularized_variables_collection is not None:
                tf.add_to_collection(regularized_variables_collection, b)
            print("{}: {}".format(b.name, b.shape))
            conv_out = tf.add(conv_out, b, name="linear_out")
        # else:
        #     # Generate tensor with zero elements (so we can retrieve the biases anyway)
        #     b = get_tf_variable("biases", [1, 1, 0], dtype, initializer=tf.zeros_initializer(),
        #                         collections=collections, variable_on_cpu=variables_on_cpu)
        #     print("{}: {}".format(b.name, b.shape))

        if use_batch_norm and not batch_norm_after_activation:
            conv_out = batch_norm_on_1d(conv_out,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope="bn",
                                        variables_collections=collections,
                                        variables_on_cpu=variables_on_cpu)

        pre_activation = conv_out
        if activation_fn is not None:
            conv_out = activation_fn(conv_out, name="activation")

        if use_batch_norm and batch_norm_after_activation:
            conv_out = batch_norm_on_1d(conv_out,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope="bn",
                                        variables_collections=collections)

        if return_pre_activation:
            return conv_out, pre_activation
        return conv_out


def conv1d_transpose(x, num_filters, output_shape=None, filter_size=(3), stride=(1), **kwargs):
    x = tf.expand_dims(x, axis=1)
    if output_shape is None:
        input_shape = x.get_shape().as_list()
        output_shape = tf.TensorShape([int(input_shape[0]),
                                       1, 2 * input_shape[2], num_filters])
    else:
        output_shape = [output_shape[0], 1] + output_shape[1:]
    filter_size = [1, filter_size[0]]
    stride = [1, stride[0]]
    conv_trans_out = conv2d_transpose(x, num_filters, output_shape, filter_size, stride, **kwargs)
    conv_trans_out = tf.squeeze(conv_trans_out, axis=1)
    return conv_trans_out


def conv2d(x, num_filters, filter_size=(3, 3,), stride=(1, 1,),
           activation_fn=tf.nn.relu, add_bias=False, use_batch_norm=True,
           batch_norm_after_activation=False, is_training=True,
           weights_initializer=None, padding="SAME",
           dtype=tf.float32, name=None, variables_on_cpu=False, collections=None,
           regularized_variables_collection=None,
           return_pre_activation=False):
    if use_batch_norm:
        assert(not add_bias)

    if name is None:
        name = tf.get_variable_scope()
    with tf.variable_scope(name):
        strides = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        if weights_initializer is None:
            if activation_fn == tf.nn.relu or activation_fn == tf.nn.elu:
                weights_initializer = relu_normal_weights_initializer()
            else:
                weights_initializer = xavier_uniform_weights_initializer()

        w = get_tf_variable("weights", filter_shape, dtype, initializer=weights_initializer,
                            collections=collections, variable_on_cpu=variables_on_cpu)
        if regularized_variables_collection is not None:
            tf.add_to_collection(regularized_variables_collection, w)
        print("{}: {}".format(w.name, w.shape))
        conv_out = tf.nn.conv2d(x, w, strides, padding, name="conv2d_out")

        if add_bias:
            b = get_tf_variable("biases", [1, 1, 1, num_filters], dtype, initializer=tf.zeros_initializer(),
                                collections=collections, variable_on_cpu=variables_on_cpu)
            if regularized_variables_collection is not None:
                tf.add_to_collection(regularized_variables_collection, b)
            print("{}: {}".format(b.name, b.shape))
            conv_out = tf.add(conv_out, b, name="linear_out")
        # else:
        #     # Generate tensor with zero elements (so we can retrieve the biases anyway)
        #     b = get_tf_variable("biases", [1, 1, 1, 0], dtype, initializer=tf.zeros_initializer(),
        #                         collections=collections, variable_on_cpu=variables_on_cpu)
        #     print("{}: {}".format(b.name, b.shape))

        if use_batch_norm and not batch_norm_after_activation:
            conv_out = batch_norm_on_2d(conv_out,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope="bn",
                                        variables_collections=collections,
                                        variables_on_cpu=variables_on_cpu)

        pre_activation = conv_out
        if activation_fn is not None:
            conv_out = activation_fn(conv_out, name="activation")

        if use_batch_norm and batch_norm_after_activation:
            conv_out = batch_norm_on_2d(conv_out,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope="bn",
                                        variables_collections=collections)

        if return_pre_activation:
            return conv_out, pre_activation
        return conv_out


def conv2d_transpose(x, num_filters, output_shape=None, filter_size=(3, 3), stride=(1, 1),
                     activation_fn=tf.nn.relu, add_bias=False, use_batch_norm=True,
                     batch_norm_after_activation=False, is_training=True,
                     weights_initializer=None, padding="SAME",
                     dtype=tf.float32, name=None, variables_on_cpu=False, collections=None,
                     regularized_variables_collection=None,
                     return_pre_activation=False):
    if use_batch_norm:
        assert(not add_bias)

    # num_filters = int(x.get_shape()[4])

    if output_shape is None:
        input_shape = x.get_shape().as_list()
        output_shape = tf.TensorShape([int(input_shape[0]),
                                       2 * input_shape[1], 2 * input_shape[2], num_filters])
    else:
        assert(output_shape[-1] == num_filters)

    if name is None:
        name = tf.get_variable_scope()
    with tf.variable_scope(name):
        strides = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], num_filters, int(x.get_shape()[3])]
        # filter_shape = [filter_size[0], filter_size[1], filter_size[2], int(x.get_shape()[4]), num_filters]

        if weights_initializer is None:
            if activation_fn == tf.nn.relu or activation_fn == tf.nn.elu:
                weights_initializer = relu_normal_weights_initializer()
            else:
                weights_initializer = xavier_uniform_weights_initializer()

        print("  INFO: weights.shape:", filter_shape)
        w = get_tf_variable("weights", filter_shape, dtype, initializer=weights_initializer,
                            collections=collections, variable_on_cpu=variables_on_cpu)
        if regularized_variables_collection is not None:
            tf.add_to_collection(regularized_variables_collection, w)
        print("{}: {}".format(w.name, w.shape))
        print("  output_shape:", output_shape)
        print("  num_filters:", num_filters)
        print("  filter_shape:", filter_shape)
        print("  x.shape:", x.shape)
        conv_trans_out = tf.nn.conv2d_transpose(x, w, output_shape, strides, padding, name="conv2d_trans_out")

        if add_bias:
            b = get_tf_variable("biases", [1, 1, 1, num_filters], dtype, initializer=tf.zeros_initializer(),
                                collections=collections, variable_on_cpu=variables_on_cpu)
            if regularized_variables_collection is not None:
                tf.add_to_collection(regularized_variables_collection, b)
            print("{}: {}".format(b.name, b.shape))
            conv_trans_out = tf.add(conv_trans_out, b, name="linear_out")
        # else:
        #     # Generate tensor with zero elements (so we can retrieve the biases anyway)
        #     b = get_tf_variable("biases", [1, 1, 1, 0], dtype, initializer=tf.zeros_initializer(),
        #                         collections=collections, variable_on_cpu=variables_on_cpu)
        #     print("{}: {}".format(b.name, b.shape))

        if use_batch_norm and not batch_norm_after_activation:
            conv_trans_out = batch_norm_on_2d(conv_trans_out,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope="bn",
                                        variables_collections=collections,
                                        variables_on_cpu=variables_on_cpu)

        pre_activation = conv_trans_out
        if activation_fn is not None:
            conv_trans_out = activation_fn(conv_trans_out, name="activation")

        if use_batch_norm and batch_norm_after_activation:
            conv_trans_out = batch_norm_on_2d(conv_trans_out,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope="bn",
                                        variables_collections=collections)

        if return_pre_activation:
            return conv_trans_out, pre_activation
        return conv_trans_out


def conv3d(x, num_filters, filter_size=(3, 3, 3), stride=(1, 1, 1),
           activation_fn=tf.nn.relu, add_bias=False, use_batch_norm=True,
           batch_norm_after_activation=False, is_training=True,
           weights_initializer=None, padding="SAME",
           dtype=tf.float32, name=None, variables_on_cpu=False, collections=None,
           regularized_variables_collection=None,
           return_pre_activation=False):
    if use_batch_norm:
        assert(not add_bias)

    if name is None:
        name = tf.get_variable_scope()
    with tf.variable_scope(name):
        strides = [1, stride[0], stride[1], stride[2], 1]
        filter_shape = [filter_size[0], filter_size[1], filter_size[2], int(x.get_shape()[4]), num_filters]

        if weights_initializer is None:
            if activation_fn == tf.nn.relu or activation_fn == tf.nn.elu:
                weights_initializer = relu_normal_weights_initializer()
            else:
                weights_initializer = xavier_uniform_weights_initializer()

        w = get_tf_variable("weights", filter_shape, dtype, initializer=weights_initializer,
                            collections=collections, variable_on_cpu=variables_on_cpu)
        if regularized_variables_collection is not None:
            tf.add_to_collection(regularized_variables_collection, w)
        print("{}: {}".format(w.name, w.shape))
        conv_out = tf.nn.conv3d(x, w, strides, padding, name="conv3d_out")

        if add_bias:
            b = get_tf_variable("biases", [1, 1, 1, 1, num_filters], dtype, initializer=tf.zeros_initializer(),
                                collections=collections, variable_on_cpu=variables_on_cpu)
            if regularized_variables_collection is not None:
                tf.add_to_collection(regularized_variables_collection, b)
            print("{}: {}".format(b.name, b.shape))
            conv_out = tf.add(conv_out, b, name="linear_out")
        # else:
        #     # Generate tensor with zero elements (so we can retrieve the biases anyway)
        #     b = get_tf_variable("biases", [1, 1, 1, 1, 0], dtype, initializer=tf.zeros_initializer(),
        #                         collections=collections, variable_on_cpu=variables_on_cpu)
        #     print("{}: {}".format(b.name, b.shape))

        if use_batch_norm and not batch_norm_after_activation:
            conv_out = batch_norm_on_3d(conv_out,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope="bn",
                                        variables_collections=collections,
                                        variables_on_cpu=variables_on_cpu)

        pre_activation = conv_out
        if activation_fn is not None:
            conv_out = activation_fn(conv_out, name="activation")

        if use_batch_norm and batch_norm_after_activation:
            conv_out = batch_norm_on_3d(conv_out,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope="bn",
                                        variables_collections=collections)

        if return_pre_activation:
            return conv_out, pre_activation
        return conv_out


def conv3d_transpose(x, num_filters, output_shape=None, filter_size=(3, 3, 3), stride=(1, 1, 1),
                     activation_fn=tf.nn.relu, add_bias=False, use_batch_norm=True,
                     batch_norm_after_activation=False, is_training=True,
                     weights_initializer=None, padding="SAME",
                     dtype=tf.float32, name=None, variables_on_cpu=False, collections=None,
                     regularized_variables_collection=None,
                     return_pre_activation=False):
    if use_batch_norm:
        assert(not add_bias)

    # num_filters = int(x.get_shape()[4])

    if output_shape is None:
        input_shape = x.get_shape().as_list()
        output_shape = tf.TensorShape([int(input_shape[0]),
                                       2 * input_shape[1], 2 * input_shape[2], 2 * input_shape[3], num_filters])
    else:
        assert(output_shape[-1] == num_filters)

    if name is None:
        name = tf.get_variable_scope()
    with tf.variable_scope(name):
        strides = [1, stride[0], stride[1], stride[2], 1]
        filter_shape = [filter_size[0], filter_size[1], filter_size[2], num_filters, int(x.get_shape()[4])]
        # filter_shape = [filter_size[0], filter_size[1], filter_size[2], int(x.get_shape()[4]), num_filters]

        if weights_initializer is None:
            if activation_fn == tf.nn.relu or activation_fn == tf.nn.elu:
                weights_initializer = relu_normal_weights_initializer()
            else:
                weights_initializer = xavier_uniform_weights_initializer()

        w = get_tf_variable("weights", filter_shape, dtype, initializer=weights_initializer,
                            collections=collections, variable_on_cpu=variables_on_cpu)
        if regularized_variables_collection is not None:
            tf.add_to_collection(regularized_variables_collection, w)
        print("{}: {}".format(w.name, w.shape))
        print("  output_shape:", output_shape)
        print("  num_filters:", num_filters)
        print("  filter_shape:", filter_shape)
        print("  x.shape:", x.shape)
        conv_trans_out = tf.nn.conv3d_transpose(x, w, output_shape, strides, padding, name="conv3d_trans_out")

        if add_bias:
            b = get_tf_variable("biases", [1, 1, 1, 1, num_filters], dtype, initializer=tf.zeros_initializer(),
                                collections=collections, variable_on_cpu=variables_on_cpu)
            if regularized_variables_collection is not None:
                tf.add_to_collection(regularized_variables_collection, b)
            print("{}: {}".format(b.name, b.shape))
            conv_trans_out = tf.add(conv_trans_out, b, name="linear_out")
        # else:
        #     # Generate tensor with zero elements (so we can retrieve the biases anyway)
        #     b = get_tf_variable("biases", [1, 1, 1, 1, 0], dtype, initializer=tf.zeros_initializer(),
        #                         collections=collections, variable_on_cpu=variables_on_cpu)
        #     print("{}: {}".format(b.name, b.shape))

        if use_batch_norm and not batch_norm_after_activation:
            conv_trans_out = batch_norm_on_3d(conv_trans_out,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope="bn",
                                        variables_collections=collections,
                                        variables_on_cpu=variables_on_cpu)

        pre_activation = conv_trans_out
        if activation_fn is not None:
            conv_trans_out = activation_fn(conv_trans_out, name="activation")

        if use_batch_norm and batch_norm_after_activation:
            conv_trans_out = batch_norm_on_3d(conv_trans_out,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope="bn",
                                        variables_collections=collections)

        if return_pre_activation:
            return conv_trans_out, pre_activation
        return conv_trans_out


def fully_connected(x, num_units, activation_fn=tf.nn.relu,
                    add_bias=False, use_batch_norm=True, batch_norm_after_activation=False, is_training=True,
                    weights_initializer=None, dtype=tf.float32, name=None, variables_on_cpu=False, collections=None,
                    regularized_variables_collection=None,
                    return_pre_activation=False):
    if use_batch_norm:
        assert(not add_bias)

    if name is None:
        name = tf.get_variable_scope()
    with tf.variable_scope(name):
        x = flatten_batch(x)

        if weights_initializer is None:
            if activation_fn == tf.nn.relu or activation_fn == tf.nn.elu:
                weights_initializer = relu_normal_weights_initializer()
            else:
                weights_initializer = xavier_uniform_weights_initializer()

        w = get_tf_variable("weights", [x.shape[-1], num_units], dtype, initializer=weights_initializer,
                            collections=collections, variable_on_cpu=variables_on_cpu)
        if regularized_variables_collection is not None:
            tf.add_to_collection(regularized_variables_collection, w)
        print("{}: {}".format(w.name, w.shape))
        out = tf.matmul(x, w, name="linear_out")

        if add_bias:
            b = get_tf_variable("biases", [num_units], dtype, initializer=tf.zeros_initializer(),
                                collections=collections, variable_on_cpu=variables_on_cpu)
            if regularized_variables_collection is not None:
                tf.add_to_collection(regularized_variables_collection, b)
            print("{}: {}".format(b.name, b.shape))
            out = tf.add(out, b, name="output")
        # else:
        #     # Generate tensor with zero elements (so we can retrieve the biases anyway)
        #     b = get_tf_variable("biases", [0], dtype, initializer=tf.zeros_initializer(),
        #                         collections=collections, variable_on_cpu=variables_on_cpu)
        #     print("{}: {}".format(b.name, b.shape))

        if use_batch_norm and not batch_norm_after_activation:
            out = batch_norm_on_flat(out,
                                     center=True, scale=True,
                                     is_training=is_training,
                                     scope="bn",
                                     variables_collections=collections)

        pre_activation = out
        if activation_fn is not None:
            out = activation_fn(out, name="activation")

        if use_batch_norm and batch_norm_after_activation:
            out = batch_norm_on_flat(out,
                                     center=True, scale=True,
                                     is_training=is_training,
                                     scope="bn",
                                     variables_collections=collections)

        if return_pre_activation:
            return out, pre_activation
        return out


def maxpool3d(x, num_filters, filter_size=(3, 3, 3), stride=(1, 1, 1),
              activation_fn=tf.nn.relu, add_bias=False, use_batch_norm=True,
              batch_norm_after_activation=False, is_training=True,
              padding="SAME", dtype=tf.float32, name=None, variables_on_cpu=False, collections=None,
              regularized_variables_collection=None,
              return_pre_activation=False):
    if use_batch_norm:
        assert(not add_bias)

    if name is None:
        name = tf.get_variable_scope()
    with tf.variable_scope(name):
        strides = [1, stride[0], stride[1], stride[2], 1]
        filter_shape = [1, filter_size[0], filter_size[1], filter_size[2], 1]

        pool_out = tf.nn.max_pool3d(x, filter_shape, strides, padding, name="maxpool3d_out")

        if add_bias:
            b = get_tf_variable("biases", [1, 1, 1, 1, num_filters], dtype, initializer=tf.zeros_initializer(),
                                collections=collections, variable_on_cpu=variables_on_cpu)
            if regularized_variables_collection is not None:
                tf.add_to_collection(regularized_variables_collection, b)
            print("{}: {}".format(b.name, b.shape))
            pool_out = tf.add(pool_out, b, name="linear_out")
        # else:
        #     # Generate tensor with zero elements (so we can retrieve the biases anyway)
        #     b = get_tf_variable("biases", [1, 1, 1, 1, 0], dtype, initializer=tf.zeros_initializer(),
        #                         collections=collections, variable_on_cpu=variables_on_cpu)
        #     print("{}: {}".format(b.name, b.shape))

        if use_batch_norm and not batch_norm_after_activation:
            pool_out = batch_norm_on_3d(pool_out,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope="bn",
                                        variables_collections=collections,
                                        variables_on_cpu=variables_on_cpu)

        pre_activation = pool_out
        if activation_fn is not None:
            pool_out = activation_fn(pool_out, name="activation")

        if use_batch_norm and batch_norm_after_activation:
            pool_out = batch_norm_on_3d(pool_out,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope="bn",
                                        variables_collections=collections)

        if return_pre_activation:
            return pool_out, pre_activation
        return pool_out


def loss_l2_norm(x, y, reduce_sum_axis=False):
    with tf.name_scope('loss_l2_norm'):
        loss = tf.square(x - y)

        if reduce_sum_axis is not False:
            loss = tf.reduce_sum(loss, reduce_sum_axis)
        return loss


def loss_l1_norm(x, y, reduce_sum_axis=False):
    with tf.name_scope('loss_l2_norm'):
        loss = tf.square(x - y)

        if reduce_sum_axis is not False:
            loss = tf.reduce_sum(loss, reduce_sum_axis)
        return loss


def log_likelihood_normal(x, mu, sigma, reduce_sum_axis=False, epsilon=1e-7):
    """
    Isotropic Gaussian log-likelihood.

    Args:
        x:
        mu:
        sigma: standard deviation.
        reduce_sum:

    Returns:

    """
    with tf.name_scope('log_likelihood_normal'):
        sigma_sq = tf.square(sigma)
        denominator_log = tf.log(epsilon + tf.sqrt(2 * np.pi * sigma_sq))
        norm = tf.square(x - mu)
        nominator_log = tf.div(-norm, (epsilon + 2 * sigma_sq))
        log_likelihood = nominator_log - denominator_log

        if reduce_sum_axis is not False:
            log_likelihood = tf.reduce_sum(log_likelihood, reduce_sum_axis)
        return log_likelihood


class ScalarSummaryWrapper(object):

    def __init__(self):
        self._placeholders = []
        self._placeholders_dict = {}
        self._summaries = []
        self._summary_op = None

    def append_with_placeholder(self, name, dtype):
        placeholder = tf.placeholder(dtype, (), name="{}_summary_ph".format(name))
        self._summaries.append(tf.summary.scalar(name, placeholder))
        self._placeholders.append(placeholder)
        self._placeholders_dict[name] = placeholder

    def append_tensor(self, name, tensor):
        self._summaries.append(tf.summary.scalar(name, tensor))

    def append_histogram_with_placeholder(self, name, dtype, shape=None):
        if shape is None:
            shape = (None)
        placeholder = tf.placeholder(dtype, shape, name="{}_histogram_ph".format(name))
        self._summaries.append(tf.summary.histogram(name, placeholder))
        self._placeholders.append(placeholder)
        self._placeholders_dict[name] = placeholder

    def append_histogram_tensor(self, name, tensor):
        self._summaries.append(tf.summary.histogram(name, tensor))

    def finalize(self):
        self._summary_op = tf.summary.merge(self._summaries)

    def get_summary_op(self):
        if self._summary_op is None:
            self.finalize()
        return self._summary_op

    def create_summary(self, sess, values):
        summary_op = self.get_summary_op()
        if type(values) == dict:
            for key in self._placeholders_dict:
                assert key in values, "No value for placeholder {}".format(key)
            feed_dict = {}
            for key, value in values.items():
                if key in self._placeholders_dict:
                    # A placeholder name from this summary wrapper
                    feed_dict[self._placeholders_dict[key]] = value
                else:
                    # Probably a tensor
                    feed_dict[key] = value
        else:
            assert len(values) == len(self._placeholders),\
                "Length of value error must be equal to number of placeholders"
            feed_dict = {self._placeholders[i]: value for i, value in enumerate(values)}
        summary = sess.run(summary_op, feed_dict)
        return summary


class HistogramSummary(object):

    def __init__(self, name_tensor_dict):
        self._fetches = []
        self._placeholders = []
        summaries = []
        for name, tensor in name_tensor_dict.items():
            var_batch_shape = [None] + tensor.shape[1:].as_list()
            placeholder = tf.placeholder(tensor.dtype, var_batch_shape)
            summaries.append(tf.summary.histogram(name, placeholder))
            self._fetches.append(tensor)
            self._placeholders.append(placeholder)
        self._summary_op = tf.summary.merge(summaries)

    @property
    def placeholders(self):
        return self._placeholders

    @property
    def fetches(self):
        return self._fetches

    @property
    def summary_op(self):
        return self._summary_op


def save_model(sess, saver, model_dir, model_name, global_step_tf, max_trials, retry_save_wait_time=5, verbose=False):
    saved = False
    trial = 0
    model_filename = os.path.join(model_dir, model_name)
    while not saved and trial < max_trials:
        try:
            trial += 1
            timer = utils.Timer()
            if verbose:
                if trial > 1:
                    logger.info("Saving model to {}. Trial {}.".format(model_filename, trial))
                else:
                    logger.info("Saving model to {}".format(model_filename))
            filename = saver.save(sess, model_filename, global_step=global_step_tf)
            save_time = timer.restart()
            saved = True
            if verbose:
                logger.info("Saving took {} s".format(save_time))
                logger.info("Saved model to file: {}".format(filename))
            latest_checkpoint = tf.train.latest_checkpoint(model_dir)
            assert(latest_checkpoint is not None)
            latest_checkpoint_basename = os.path.basename(latest_checkpoint)
            model_ckpt_name = os.path.basename(filename)
            assert(latest_checkpoint_basename == model_ckpt_name)
            assert(os.path.isfile(filename + ".index"))
            assert(os.path.isfile(filename + ".meta"))
        except Exception as err:
            logger.error("ERROR: Exception when trying to save model: {}".format(err))
            traceback.print_exc()
            if trial < max_trials:
                if verbose:
                    logger.error("Retrying to save model in {} s...".format(retry_save_wait_time))
                time.sleep(retry_save_wait_time)
            else:
                raise RuntimeError("Unable to save model after {} trials".format(max_trials))
