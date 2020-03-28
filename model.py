from keras.models import Sequential
from keras.layers import Dense
from preprocessing import preprocessing
import tensorflow as tf

features,labels = preprocessing.data()

model = Sequential()
model.add(Dense(500, input_shape=(features[0].shape), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(18, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(features, labels, epochs=20,batch_size = 10)

model.save("pokemon.model")

#----------------------------------tensorflow model---------------------
def tensroflow_model(n_node_hl1,n_nodes_hl2,data):

    dense = tf.keras.layers.Dense(500, activation = 'relu')(data)

    l2 = tf.add(tf.matmul(dense,tf.random_normal([n_node_hl1, n_nodes_hl2])), tf.random_normal([n_nodes_hl2]))
    l2 = tf.nn.relu(l2)

    l3 = tf.keras.layers.Dense(500)(l2)
    l3 = tf.nn.relu(l3)

    output = tf.keras.layers.Dense(18,activation='sigmoid')(l3)
