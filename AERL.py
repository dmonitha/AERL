from tensorflow import keras
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Dealing with dataset
label_encoder = LabelEncoder()
dataset=pd.read_csv("path_to_your_datasets",sep=',',header=None,low_memory=False)

dataset = dataset.dropna(axis=0, how='any', subset=None, inplace=False)
df = pd.DataFrame(dataset)

df.iloc[1:,-1] = label_encoder.fit_transform(df.iloc[1:,-1])

df = df.iloc[1:].apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col)

df = df.dropna()
X = df.iloc[1:, :-1]
y =  df.iloc[1:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
# Fit the scaler on the data and transform it
X = scaler.fit_transform(X)

# Autoencoder class
class AERL(keras.Model):
  def __init__(self,input_dim):
    super().__init__()
    self.inp_shape = input_dim
    self.reduced_shape = round(0.25*input_dim)
    self.enc1 = Dense(round(0.75*self.inp_shape), activation='relu', input_shape=(self.inp_shape,), dtype='float64')
    self.enc2 = Dense(round(0.50*self.inp_shape), activation='relu', dtype='float64')
    self.dec1 = Dense(round(0.50*self.inp_shape), activation='relu', dtype='float64')
    self.dec2 = Dense(self.inp_shape, activation='sigmoid', dtype='float64')


  def encoder(self,inputs):
    x = self.enc1(inputs)
    x = self.enc2(x)
    return x

  def decoder(self,inputs):
    x = self.dec1(inputs)
    x = self.dec2(inputs)
    return x

  def call(self,x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

epsilon = tf.keras.backend.epsilon()
def kl(y_true, y_pred):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    y_true = tf.clip_by_value(y_true, epsilon, 1.0 - epsilon)

    # Compute KL Divergence
    kl_div = tf.reduce_sum(y_true * tf.math.log(y_true / y_pred + epsilon), axis=-1)

    return tf.reduce_mean(kl_div)


def custom_loss(y_true,y_pred):
    # Approximate mutual information
    mi_score = kl(y_true, y_pred)
    reconstruction_loss =tf.reduce_mean(tf.square(y_true - y_pred), axis =-1)
    return  reconstruction_loss + (epsilon*mi_score)


input_dim = X.shape[1]
myModel = AERL(input_dim)
myModel.compile(loss=custom_loss,optimizer='adam')
myModel.fit(X_train,X_train,epochs=30,shuffle=True)


encoded_features = myModel.encoder(X_train)
print("Shape of encoded representation",encoded_features.shape)