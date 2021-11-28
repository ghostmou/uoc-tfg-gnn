import spektral
import tensorflow as tf
import numpy as np
from spektral.layers import MessagePassing
from spektral.data.loaders import SingleLoader
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.activations import elu


class MPNNLayer(MessagePassing):
    def __init__(self, n_out, activation, **kwargs):
        super().__init__(activation=activation, **kwargs)
        self.n_out = n_out

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            shape=(batch_input_shape[0][-1], self.n_out)
        )

    def call(self, inputs):
        x, a = inputs

        x = tf.matmul(x, self.kernel)

        return self.propagate(x=x, a=a)

    def message(self, x):
        return self.get_j(x)
    
    def aggregate(self, messages): 
        return spektral.layers.ops.scatter_mean(messages, self.index_i, self.n_nodes)

    def update(self, embeddings):
        return self.activation(embeddings)

# Download CORA dataset and its different members
dataset = spektral.datasets.citation.Citation(
    'cora', 
    random_split=False, # split randomly: 20 nodes per class for training, 30 nodes 
        # per class for validation; or "Planetoid" (Yang et al. 2016)
    normalize_x=False,  # normalize the features
    dtype=np.float32 # numpy data type for the graph data
    )

mask_tr, mask_va, mask_te = dataset.mask_tr, dataset.mask_va, dataset.mask_te

# Parameters
l2_reg = 5e-6  # L2 regularization rate
learning_rate = 0.2  # Learning rate
epochs = 20  # Number of training epochs
patience = 200  # Patience for early stopping
a_dtype = dataset[0].a.dtype  # Only needed for TF 2.1

N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels  # Number of classes

# Model definition
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True, dtype=a_dtype)

output = MPNNLayer(
    n_out, 
    activation=keras.activations.softmax,  # Kipf 2016
    kernel_regularizer=keras.regularizers.l2(l2_reg), 
    use_bias=False
)([x_in, a_in])

# Build model
model = Model(inputs=[x_in, a_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", weighted_metrics=["acc"]
)
model.summary()

# Train model
loader_tr = SingleLoader(dataset, sample_weights=mask_tr)
loader_va = SingleLoader(dataset, sample_weights=mask_va)
model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_va.load(),
    validation_steps=loader_va.steps_per_epoch,
    epochs=epochs,
    callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
)

# Evaluate model
print("Evaluating model.")
loader_te = SingleLoader(dataset, sample_weights=mask_te)
eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))