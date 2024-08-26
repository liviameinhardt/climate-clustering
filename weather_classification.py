import tensorflow as tf
import numpy as np
import xarray as xr

from scipy.special import softmax

import utils.multicrop_dataset as multicrop_dataset
import utils.architecture as architecture
import utils.clusters_plots as plots

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
    parser.add_argument('VARIABLE', type=str, help='Variable (str)')
    parser.add_argument('DATA_WIND', type=str, help='DATA_WIND (int)')
    
    # Optional arguments
    parser.add_argument('--PROTOTYPES', type=int, help='Number of prototypes (int)')
    parser.add_argument('--PROPORTIONS', type=float, nargs='*', default=[], help='Proportions (list of floats)')
    parser.add_argument('--LOAD_WEIGHTS', action='store_false', help='Load weights (bool)')
    parser.add_argument('--SAVE_ASSIGMENTS', action='store_true', help='Save assigments (bool)')

    args = parser.parse_args()

    if args.PROPORTIONS is None and args.PROTOTYPES is None:
        parser.error('The --PROTOTYPES argument is required if --PROPORTIONS is not provided.')

    elif len(args.PROPORTIONS):
        args.PROTOTYPES = len(args.PROPORTIONS)

    print(args.BS, args.EPOCHS, args.VARIABLE, args.DATA_WIND, args.PROTOTYPES,
                args.PROPORTIONS, args.LOAD_WEIGHTS, args.SAVE_ASSIGMENTS)


    # train_swav(args.BS, args.EPOCHS, args.VARIABLE, args.DATA_WIND, args.PROTOTYPES,
    #             args.PROPORTIONS, args.LOAD_WEIGHTS, args.SAVE_ASSIGMENTS)


def open_nc(file_path):

    try:
        ds = xr.open_dataset(file_path)

    except Exception as e:
        print(f"Error opening file: {e}")
        return 

    else:
        return ds
    

def get_values(variable, file= 'data/msl_t2m_1983_2023.nc',wind = 4, quantile=False):
    #just to save process

    ds = open_nc(file)
    df = ds[variable].values

    if quantile:
        max_o = np.quantile(df,0.99)
        min_o = np.quantile(df,0.1)

        df = np.clip(df, min_o, max_o)

    else:
        min_o = np.min(df)
        max_o = np.max(df)

    norm_imgs = (df - min_o)/(max_o -min_o)

    time, lat, long = norm_imgs.shape

    norm_imgs = norm_imgs.reshape(int(time/wind), wind, lat, long)
    norm_imgs = np.transpose(norm_imgs, (0, 2, 3, 1))

    # np.save(f'data/{variable}',norm_imgs,allow_pickle=False)

    return norm_imgs


def train_swav(batch_size, epochs, variable, compare_windown, 
               prototypes, proportions=[], load_weights=False,save_assigments=True):

    print('Loading data')
    #SCALE FOR CROPPING
    MIN_SCALE = [0.7, 0.5]
    MAX_SCALE = [1., 0.7]
    NUM_CROPS= [2,3]
    SIZE_CROPS = [50, 32]  
    model_tag  = f'{variable}_{prototypes}_bs{batch_size}_wind{compare_windown}'

    data = get_values(variable,wind=compare_windown)
    data_tensors = tf.data.Dataset.from_tensor_slices(data) 


    # Get multiple data loaders
    trainloaders = multicrop_dataset.get_multires_dataset(data_tensors,
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

    print('Start training')
    epoch_wise_loss, models_tr = architecture.train_swav(feature_backbone,projection_prototype,
                                            trainloaders_zipped,opt,
                                            crops_for_assign=[0, 1],temperature=0.1,epochs=epochs,
                                            n_crops = NUM_CROPS, proto_proportions = proportions,
                                            model_name=model_tag)


    np.save(f'model_weights/loss_{model_tag}.npy',epoch_wise_loss)

    print('Done')

    if save_assigments:
        get_assigments(data,model_tag,prototypes)


def get_assigments(data,model_name,prototypes):
    
    print('Getting assigments')

    #treat data
    final_data = []
    for sample in data:

        for v in range(sample.shape[-1]):
            final_data.append(sample[:,:,v])


    #get model 
    feature_backbone = architecture.get_resnet_backbone()
    projection_prototype = architecture.get_projection_prototype(prototype_dimension = prototypes)

    feature_backbone.load_weights(f'model_weights/feature2d_{model_name}.h5')
    projection_prototype.load_weights(f'model_weights/proj2d_{model_name}.h5')

    #get assigments
    blocks=100
    size = int(len(final_data)/blocks)

    for i in range(blocks):

        embeddings_ = feature_backbone(np.asarray(final_data)[i*size:(i+1)*size])
        projection_, prototype_ = projection_prototype(embeddings_)

        if i == 0:
            prototype=np.asarray(prototype_)
        else:
            prototype=np.concatenate([prototype,np.asarray(prototype_)])

    prototype = np.asarray(prototype)
    del projection_,embeddings_

    assignments = np.argmax(softmax(prototype),axis=1)
    np.save(f'model_weights/assignments_{model_name}',assignments,allow_pickle=False)

    print('Done')

    #save plots
    print('Saving plots')
    plots.clusters_plots(final_data, assignments, prototypes, model_name)



if __name__ == "__main__":
    main()

