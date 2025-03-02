from tensorflow._api.v2.data import Dataset
import tensorflow as tf
import numpy as np
import torch.distributions.constraints

def build_pipeline(x, y, w=None, val_size=200, batch_size=128,
                   bootstraps=0, bootstraps_noise=None, buffer=None):
    """Split a model-based optimization dataset consisting of a set of design
    values x and prediction values y into a training and validation set,
    supporting bootstrapping and importance weighting

    Args:

    x: tf.Tensor
        a tensor containing design values from a model-based optimization
        dataset, typically taken from task.x
    y: tf.Tensor
        a tensor containing prediction values from a model-based optimization
        dataset, typically taken from task.y
    w: None or tf.Tensor
        an optional tensor of the same shape as y that specifies the
        importance weight of samples in a model-based optimization dataset
    val_size: int
        the number of samples randomly chosen to be in the validation set
        returned by the function
    batch_size: int
        the number of samples to load in every batch when drawing samples
        from the training and validation sets
    bootstraps: int
        the number of copies of the dataset to draw with replacement
        for training an ensemble of forward models
    bootstraps_noise: float
        the standard deviation of zero mean gaussian noise independently
        sampled and added to each bootstrap of the dataset

    Returns:

    training_dataset: tf.data.Dataset
        a tensorflow dataset that has been batched and prefetched
        with an optional importance weight and optional bootstrap included
    validation_dataset: tf.data.Dataset
        a tensorflow dataset that has been batched and prefetched
        with an optional importance weight and optional bootstrap included

    """

    # shuffle the dataset using a common set of indices
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    # create a training and validation split
    x = x[indices]
    y = y[indices]
    train_inputs = [x[val_size:], y[val_size:]]
    validate_inputs = [x[:val_size], y[:val_size]]
    size = x.shape[0] - val_size

    if bootstraps > 0:

        # sample the data set with replacement
        train_inputs.append(tf.stack([
            tf.math.bincount(tf.random.uniform([size], minval=0,
                                               maxval=size, dtype=tf.int32),
                             minlength=size, dtype=tf.float32)
            for b in range(bootstraps)], axis=1))

        # add noise to the labels to increase diversity
        if bootstraps_noise is not None:
            train_inputs.append(bootstraps_noise *
                                tf.random.normal([size, bootstraps]))

    if w is not None:
        # add importance weights to the data set
        train_inputs.append(w[indices[val_size:]])

    # build the parallel tensorflow data loading pipeline
    training_dataset = Dataset.from_tensor_slices(tuple(train_inputs))
    validation_dataset = Dataset.from_tensor_slices(tuple(validate_inputs))
    training_dataset = training_dataset.shuffle(size if buffer is None else buffer)
    validation_dataset = validation_dataset

    # batch and prefetch each data set
    training_dataset = training_dataset.batch(batch_size)
    validation_dataset = validation_dataset.batch(batch_size)
    return (training_dataset.prefetch(tf.data.experimental.AUTOTUNE),
            validation_dataset.prefetch(tf.data.experimental.AUTOTUNE))