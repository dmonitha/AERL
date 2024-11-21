!pip install shap
!pip install xgboost

from tensorflow import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier as xgb
import shap
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

def build_AERL(input_dim,inputs_filtered):
  AERL = AERL(input_dim)
  AERL.compile(loss=custom_loss,optimizer='adam')
  AERL.fit(inputs_filtered,inputs_filtered,epochs=30,shuffle=True)
  return AERL

predictions = AERL.predict(X_test)

# Calculate custom loss for each feature
def custom_loss_per_feature(y_true, y_pred):
    epsilon = tf.keras.backend.epsilon()

    # Clip values
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    y_true = tf.clip_by_value(y_true, epsilon, 1.0 - epsilon)

    # Compute KL Divergence
    kl_div = y_true * tf.math.log(y_true / y_pred + epsilon)

    # Compute reconstruction loss
    reconstruction_loss = tf.square(y_true - y_pred)

    # Combine losses
    return reconstruction_loss + (epsilon * kl_div)


custom_errors = custom_loss_per_feature(X_test, predictions)
# Convert to numpy and calculate mean across all records for each feature
custom_loss_per_feature = pd.Series(np.mean(custom_errors.numpy(), axis=0))


sorted_least_loss_values = custom_loss_per_feature.sort_values(ascending=True)

print(custom_loss_per_feature)
print(sorted_least_loss_values)

values = sorted_least_loss_values.values
x_axis = range(len(values))

# Keeping top 25% of the features
num_features_to_keep = int(len(sorted_least_loss_values) * 0.25)
top_features = sorted_least_loss_values.index[:num_features_to_keep]
print("top",top_features)
above_threshold = top_features

X_test_filtered = X_test[above_threshold]
myNewModel = build_AERL(X_test_filtered.shape[1],X_test_filtered)

encoder_input = keras.Input(shape=(X_test_filtered.shape[1],))
encoder_output = myNewModel.encoder(encoder_input)
encoder_model = keras.Model(encoder_input, encoder_output)

print(X_test_filtered.shape)
print(encoder_model.input_shape)

background_data = X_test_filtered

# Calculating SHAP values
background_array = background_data.values
background_array = background_array.astype(np.float32)
explainer = shap.DeepExplainer(encoder_model, background_array)

X_test_array = X_test_filtered.values.astype(np.float32)
shap_values = explainer.shap_values(X_test_array)

# Calculate mean absolute SHAP values for feature importance
feature_importance = np.abs(shap_values).mean(axis=2).mean(axis=0)
feature_names = background_data.columns.tolist()
# Create a DataFrame with feature names and their importance
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
})

feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

X_check = X_test_filtered[feature_importance_df['feature']]

# Classification Accuracy
gb = xgb()
best_score_gb  = None
total = feature_importance_df.shape[0]
for i in range(10,total,20):
  top_features = feature_importance_df['feature'][:i].tolist()
  X_reduced=X_test[top_features]
  scaler = MinMaxScaler()
  X_scaled_new = scaler.fit_transform(X_reduced)
  cv_score = cross_val_score(gb,X_scaled_new,y_test,cv=5)
  if best_score_gb is None or cv_score.mean() > best_score_gb:
    best_score_gb = cv_score.mean()
    best_features_gb = top_features

print("Best score:", best_score_gb)
print("Best features:", len(best_features_gb))

best_score_rf = None
rf = RandomForestClassifier(n_estimators = 9)
total = feature_importance_df.shape[0]
for i in range(10,200,50):
  top_features = feature_importance_df['feature'][:i].tolist()
  X_reduced=X_test[top_features]
  scaler = MinMaxScaler()
  X_scaled_new = scaler.fit_transform(X_reduced)

  cv_score = cross_val_score(rf,X_scaled_new,y_test,cv=3)
  if best_score_rf is None or cv_score.mean() > best_score_rf:
    best_score_rf = cv_score.mean()
    best_features_rf = top_features

print("Best score:", best_score_rf)
print("Best features:", len(best_features_rf))

best_score_nn = None
knn = KNeighborsClassifier(n_neighbors=3)
total = feature_importance_df.shape[0]
for i in range(10,total,20):
  top_features = feature_importance_df['feature'][:i].tolist()
  X_reduced=X_test[top_features]
  scaler = MinMaxScaler()
  X_scaled_new = scaler.fit_transform(X_reduced)

  cv_score = cross_val_score(knn,X_scaled_new,y_test,cv=3)
  if best_score_nn is None or cv_score.mean() > best_score_nn:
    best_score_nn = cv_score.mean()
    best_features_nn = top_features

print("Best score:", best_score_nn)
print("Best features:", len(best_features_nn))

best_score_svc = None
svc = LinearSVC()
total = feature_importance_df.shape[0]
for i in range(10,100,5):
  top_features = feature_importance_df['feature'][:i].tolist()
  X_reduced=X_test[top_features]
  scaler = MinMaxScaler()
  X_scaled_new = scaler.fit_transform(X_reduced)

  cv_score = cross_val_score(svc,X_scaled_new,y_test,cv=3)
  if best_score_svc is None or cv_score.mean() > best_score_svc:
    best_score_svc = cv_score.mean()
    best_features_svc = top_features

print("Best score:", best_score_svc)
print("Best features:", len(best_features_svc))

print(len(above_threshold))
print(round(best_score_gb,3),"(",len(best_features_gb),")");
print(round(best_score_rf,3),"(",len(best_features_rf),")");
print(round(best_score_nn,3),"(",len(best_features_nn),")");
print(round(best_score_svc,3),"(",len(best_features_svc),")");