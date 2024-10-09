from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf

import numpy as np
from tqdm import trange
from itertools import groupby
from typing import List

def residual_block(x: tf.Tensor, filters: int, kernel_size: int, stride: int, bn: bool, name: str = None) -> tf.Tensor:

    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same', name=name + '_0_conv')(x)
        if bn:
            shortcut = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)

    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same', name=name + '_1_conv')(x)
    if bn:
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = tf.keras.layers.ReLU(name=name + '_1_relu')(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same', name=name + '_2_conv')(x)
    if bn:
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_2_bn')(x)

    x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
    x = tf.keras.layers.ReLU(name=name + '_out')(x)

    return x


def resnet_encoder(input_shape: List[int], num_layers: List[int], filters: List[int], strides: List[int],
                   latent_dim: int, bn: bool) -> tf.keras.Model:

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same", name='conv1_conv')(inputs)
    if bn:
        x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='conv1_bn')(x)
    x = tf.keras.layers.ReLU(name='conv1_relu')(x)

    for block_idx, (n, f, stride) in enumerate(zip(num_layers, filters, strides)):

        x = residual_block(x, filters=f, kernel_size=3, stride=stride, bn=bn, name=f'b{block_idx+1}_1')
        for i in range(1, n):
            x = residual_block(x, filters=f, kernel_size=3, stride=1, bn=bn, name=f'b{block_idx+1}_{i+1}')

    encoder = tf.keras.Model(inputs, x, name="encoder")

    return encoder

def get_resnet_backbone():

  base_model = resnet_encoder([None, None, 1], [2, 2, 2, 2],[64, 128, 256, 512], [1, 2, 2, 2] ,300, True)

  base_model.trainable = True

  inputs = layers.Input((None, None, 1))

  h = base_model(inputs, training=True)
  h = layers.GlobalAveragePooling2D()(h)
  backbone = models.Model(inputs, h)
    
  return backbone


# def get_resnet_backbone():

# 	base_model = tf.keras.applications.ResNet50(include_top=False,
# 		weights=None, input_shape=(None, None, 3))
	
# 	base_model.trainable = True

# 	inputs = layers.Input((None, None, 3))
# 	h = base_model(inputs, training=True)
# 	h = layers.GlobalAveragePooling2D()(h)
# 	backbone = models.Model(inputs, h)

# 	return backbone


def get_projection_prototype(dense_1=1024, dense_2=96, prototype_dimension=10):

	inputs = layers.Input((512, ))
	projection_1 = layers.Dense(dense_1)(inputs)
	projection_1 = layers.BatchNormalization()(projection_1)
	projection_1 = layers.Activation("relu")(projection_1)

	projection_2 = layers.Dense(dense_2)(projection_1)
	projection_2_normalize = tf.math.l2_normalize(projection_2, axis=1, name='projection')

	prototype = layers.Dense(prototype_dimension, use_bias=False, name='prototype')(projection_2_normalize)

	return models.Model(inputs=inputs,
		outputs=[projection_2_normalize, prototype])



def sinkhorn(sample_prototype_batch,y_prop=[]):
    
    Q = tf.transpose(tf.exp(sample_prototype_batch/0.05))
    Q /= tf.keras.backend.sum(Q)
    K, B = Q.shape

    u = tf.zeros_like(K, dtype=tf.float32)
    c = tf.ones_like(B, dtype=tf.float32) / B
    
    if len(y_prop): #add condition to use known proportions instead of equiprobable hypothesis
        r = y_prop
        
    else:
        r = tf.ones_like(K, dtype=tf.float32) / K #equiprobable hypothesis
    
    for _ in range(3):
        u = tf.keras.backend.sum(Q, axis=1)
        Q *= tf.expand_dims((r / u), axis=1)
        Q *= tf.expand_dims(c / tf.keras.backend.sum(Q, axis=0), 0)

    final_quantity = Q / tf.keras.backend.sum(Q, axis=0, keepdims=True)
    final_quantity = tf.transpose(final_quantity)

    return final_quantity


def train_step(input_views, feature_backbone, projection_prototype,
				optimizer, crops_for_assign, temperature, n_crops, y_prop):

	# ============ retrieve input data ... ============
	im1, im2, im3, im4, im5  = input_views
	inputs = [im1, im2, im3, im4, im5]

	batch_size = inputs[0].shape[0]

	# ============ create crop entries with same shape ... ============
	#A vector of indices to reorder as views with similar resolutions
	crop_sizes = [inp.shape[1] for inp in inputs] # list of crop size of views
	unique_consecutive_count = [len([elem for elem in g]) for _, g in groupby(crop_sizes)] # equivalent to torch.unique_consecutive

	#(unique_consecutive_count)
	idx_crops = tf.cumsum(unique_consecutive_count)

	# ============ multi-res forward passes ... ============
	# tf.stop_gradient have been placed carefully in order to exclude the computations from dependency tracing.
	# This is useful any time you want to compute a value with TensorFlow but need to pretend that the value
	#was a constant.
	start_idx = 0

	with tf.GradientTape() as tape:

		for end_idx in idx_crops:

			concat_input = tf.stop_gradient(tf.concat(inputs[start_idx:end_idx], axis=0))
			_embedding = feature_backbone(concat_input) # get embedding of same dim views together

			if start_idx == 0:
				embeddings = _embedding # for first iter

			else:
				embeddings = tf.concat((embeddings, _embedding), axis=0) # concat all the embeddings from all the views
				
			start_idx = end_idx

		projection, prototype = projection_prototype(embeddings) # get normalized projection and prototype
		projection = tf.stop_gradient(projection)

		# ============ swav loss ... ============
		# https://github.com/facebookresearch/swav/issues/19

		loss = 0
		for i, crop_id in enumerate(crops_for_assign): # crops_for_assign = [0,1] hold that we use to create these codes the views in these positions.
			with tape.stop_recording():   # there to ensure the computations for cluster assignments do not get traced for gradient updates
				out = prototype[batch_size * crop_id: batch_size * (crop_id + 1)]

				# get assignments
				q = sinkhorn(out,y_prop) # sinkhorn is used for cluster assignment

			# cluster assignment prediction
			subloss = 0
			for v in np.delete(np.arange(np.sum(n_crops)), crop_id): # (for rest of the portions compute p and take cross entropy with q)
				p = tf.nn.softmax(prototype[batch_size * v: batch_size * (v + 1)] / temperature)
				subloss -= tf.math.reduce_mean(tf.math.reduce_sum(q * tf.math.log(p), axis=1))
			loss += subloss / tf.cast((tf.reduce_sum(n_crops) - 1), tf.float32)

		loss /= len(crops_for_assign)

	# ============ backprop ... ============
	variables = feature_backbone.trainable_variables + projection_prototype.trainable_variables
	gradients = tape.gradient(loss, variables)
	optimizer.apply_gradients(zip(gradients, variables))

	return loss


def train_swav(feature_backbone,
				projection_prototype,
				dataloader,
				optimizer,
				crops_for_assign,
				temperature,
				n_crops,
				proto_proportions=[],
				epochs=50,
				model_name=''):

	step_wise_loss = []
	epoch_wise_loss = []

	for epoch in range(epochs):

		# normalize the prototypes
		w = projection_prototype.get_layer('prototype').get_weights()
		w = tf.transpose(w)
		w = tf.math.l2_normalize(w, axis=1)
		projection_prototype.get_layer('prototype').set_weights(tf.transpose(w))

		iter_data = iter(dataloader)
		t = trange(len(dataloader), position=0, leave=True)

		for i in  t:

			inputs = next(iter_data)

			loss = train_step(inputs, feature_backbone, projection_prototype,
								optimizer, crops_for_assign, temperature,n_crops ,proto_proportions)
			
			step_wise_loss.append(loss)
			t.set_postfix(loss='{:05.3f}'.format(loss))

		epoch_wise_loss.append(np.mean(step_wise_loss))

		print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

		print('Saving weights')

		feature_backbone_weights = feature_backbone.get_weights()
            
		feature_backbone.compile()
		projection_prototype.compile()
            
		feature_backbone.save(f'model_weights/feature2d_{model_name}.h5')
		projection_prototype.save(f'model_weights/proj2d_{model_name}.h5')

	return epoch_wise_loss, [feature_backbone, projection_prototype]