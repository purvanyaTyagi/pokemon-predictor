import pandas as pd 
import numpy as np
class preprocessing:
	def data():
		data = pd.read_csv("Pokemon.csv")

		features = data.drop('Type 1', axis = 1)

		labels = data['Type 1'].copy()

		n_classes = len(np.unique(labels))

			##type 2 is not required
		features = features.drop('Type 2',axis = 1)

		from sklearn.preprocessing import OneHotEncoder

		features = OneHotEncoder().fit_transform(features).toarray() 

		labels = labels.to_numpy()

		from sklearn.preprocessing import LabelEncoder 

		encoder = LabelEncoder()

		labels = encoder.fit_transform(labels)

		from keras.utils import to_categorical
		labels = to_categorical(labels, 18)

		from sklearn.preprocessing import StandardScaler 
		sc_X = StandardScaler()
		features = sc_X.fit_transform(features)

		return features,labels

