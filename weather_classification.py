import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import utils.multicrop_dataset as multicrop_dataset
import utils.architecture as architecture

import argparse

tf.random.set_seed(0)
np.random.seed(0)

options = tf.data.Options()
options.experimental_optimization.noop_elimination = True             # eliminate no-op transformations
tf.compat.v1.data.experimental.OptimizationOptions.map_vectorization = True    # vectorize map transformations
options.experimental_optimization.apply_default_optimizations = True  # apply default graph optimizations
options.experimental_deterministic = True                            # False disable deterministic order
options.threading.max_intra_op_parallelism = 1           # overrides the maximum degree of intra-op parallelism 


def main():

    parser = argparse.ArgumentParser(description='Process some integers and other parameters.')
    
    # Mandatory positional arguments
    parser.add_argument('BS', type=int, help='Batch size (int)')
    parser.add_argument('EPOCHS', type=int, help='Number of epochs (int)')
    parser.add_argument('MODEL_TAG', type=str, help='Model tag (str)')
    
    # Optional arguments
    parser.add_argument('--PROTOTYPES', type=int, help='Number of prototypes (int)')
    parser.add_argument('--PROPORTIONS', type=float, nargs='*', default=[], help='Proportions (list of floats)')
    parser.add_argument('--LOAD_WEIGHTS', action='store_true', help='Load weights (bool)')

    args = parser.parse_args()

    if args.PROPORTIONS is None and args.PROTOTYPES is None:
        parser.error('The --PROTOTYPES argument is required if --PROPORTIONS is not provided.')

    elif len(args.PROPORTIONS):
        args.PROTOTYPES = len(args.PROPORTIONS)

    train_swav(args.BS, args.EPOCHS, args.MODEL_TAG, args.PROTOTYPES, args.PROPORTIONS, args.LOAD_WEIGHTS)


def train_swav(batch_size, epochs, model_tag, prototypes, proportions=[], load_weights=False):

    #SCALE FOR CROPPING
    MIN_SCALE = [0.7, 0.5]
    MAX_SCALE = [1., 0.7]
    NUM_CROPS= [2,3]
    SIZE_CROPS = [50, 32]  

    data = np.load('data/test_data.npy')
    data = tf.data.Dataset.from_tensor_slices(data) 

    # Get multiple data loaders
    trainloaders = multicrop_dataset.get_multires_dataset(data,
        size_crops=SIZE_CROPS,
        num_crops=NUM_CROPS,
        min_scale=MIN_SCALE,
        max_scale=MAX_SCALE,
        options=options)

    trainloaders_zipped = tf.data.Dataset.zip(trainloaders)

    trainloaders_zipped = (
            trainloaders_zipped
            .batch(batch_size)
            .prefetch( tf.data.experimental.AUTOTUNE)
        )

    feature_backbone = architecture.get_resnet_backbone()
    projection_prototype = architecture.get_projection_prototype(prototype_dimension = prototypes)

    if load_weights:
        feature_backbone.load_weights(f'model_weights/feature2d_{model_tag}.h5')
        projection_prototype.load_weights(f'model_weights/proj2d_{model_tag}.h5')

    lr_decayed_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate=0.1, decay_steps=1000)

    opt = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn)

    epoch_wise_loss, models_tr = architecture.train_swav(feature_backbone,projection_prototype,
                                            trainloaders_zipped,opt,
                                            crops_for_assign=[0, 1],temperature=0.1,epochs=epochs,
                                            n_crops = NUM_CROPS, proto_proportions = proportions,
                                            model_name=model_tag)

    plt.plot(epoch_wise_loss, label='Epoch-wise Loss')
    plt.axhline(y=np.log(prototypes), color='r', linestyle='--', label='ln(#prototypes)')

 
if __name__ == "__main__":
    main()