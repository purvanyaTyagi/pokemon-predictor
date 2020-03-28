from keras.models import Sequential
from keras.layers import Dense
from preprocessing import preprocessing

features,labels = preprocessing.data()

model = Sequential()
model.add(Dense(500, input_shape=(features[0].shape), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(18, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(features, labels, epochs=20,batch_size = 10)

model.save("pokemon.model")