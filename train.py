import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df=pd.read_csv(r'Dataset\critical_heat_flux_dataset.csv')

df.head()

df.shape

df.isnull().sum()

df.info()

df['author'].value_counts()

"""**Distribution plots with respect to each geometry**"""

Tube = df[df['geometry'] == 'tube']

# Check the distribution of the dataset with respect to each geometry
plt.figure(figsize=(15, 10))
for i, feature in enumerate(Tube.columns[2:]):
    plt.subplot(3, 3, i + 1)
    sns.histplot(Tube[feature], kde=True, color='orange')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(f'{feature}')
    plt.ylabel('Density')

plt.tight_layout()
plt.show()

Annulus = df[df['geometry'] == 'annulus']

# Check the distribution of the dataset with respect to each geometry
plt.figure(figsize=(15, 10))
for i, feature in enumerate(Annulus.columns[2:]):
    plt.subplot(3, 3, i + 1)
    sns.histplot(Annulus[feature], kde=True, color='green')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(f'{feature}')
    plt.ylabel('Density')

plt.tight_layout()
plt.show()

Plate = df[df['geometry'] == 'plate']

# Check the distribution of the dataset with respect to each geometry
plt.figure(figsize=(15, 10))
for i, feature in enumerate(Plate.columns[2:]):
    plt.subplot(3, 3, i + 1)
    sns.histplot(Plate[feature], kde=True, color='blue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(f'{feature}')
    plt.ylabel('Density')

plt.tight_layout()
plt.show()

df.drop(columns=['id'], inplace=True, axis=1)

df.head()

X = df.drop(columns=['chf_exp [MW/m2]'], axis=1)
y = df['chf_exp [MW/m2]']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

"""**Encoding the Data**"""

#Encoding Train Data

author_encoder = LabelEncoder()
geometry_encoder = LabelEncoder()

X_train['author'] = author_encoder.fit_transform(X_train['author'])
X_train['geometry'] = geometry_encoder.fit_transform(X_train['geometry'])
X_train.head()

X_train['author'].unique()



#Encoding Test Data
X_test['author'] = author_encoder.transform(X_test['author'])
X_test['geometry'] = geometry_encoder.transform(X_test['geometry'])
X_test.head()

from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # For regression, output layer has one neuron

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=10)

# Evaluate the model on test data
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Mean Absolute Error: {mae}')

predictions = model.predict(X_test)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation ='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))

model.summary()

model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

y_pred = model.predict(X_test)

# Calculate the mean squared error and r2 score
mse = mean_squared_error(y_test, y_pred)

# Calculate the metrics
r2 = r2_score(y_test, y_pred)
print('R2 Score:', r2)
print("Mean Square Error:",mse)
print("Root Mean Square Error:",np.sqrt(mse))

"""##⬇ Using ML"""

new_cols = {'author':'author', 	'geometry':'geometry',	'pressure [MPa]':'pressure',	'mass_flux [kg/m2-s]':'mass_flux',	'x_e_out [-]':'x_e_out',	'D_e [mm]':'D-e',	'D_h [mm]':'D_h',	'length [mm]':'length'}

X_train.rename(columns=new_cols, inplace=True)
X_test.rename(columns=new_cols, inplace=True)

'''
#Hyperparameter Tuning

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from scipy.stats import uniform, randint

# Define the parameter distribution
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(3, 6),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    'gamma': uniform(0, 0.3)
}

# Initialize the XGBRegressor
xgb = XGBRegressor()

# Initialize the RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist,
                                   n_iter=100, scoring='neg_mean_squared_error',
                                   cv=3, verbose=1, n_jobs=-1, random_state=42)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Best parameters and best score
best_params = random_search.best_params_
best_score = random_search.best_score_

# Output the best parameters and score
print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")

# Train the model with the best parameters
xgb_best = XGBRegressor(**best_params)
xgb_best.fit(X_train, y_train)

# Predict with the test set
xgb_prediction = xgb_best.predict(X_test)

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

#xgb = XGBRegressor(colsample_bytree=0.79, gamma=0.013, learning_rate=0.19, max_depth=3, n_estimators=263, subsample=0.86)
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_prediction = xgb.predict(X_test)

mae_xgb = mean_absolute_error(y_test, xgb_prediction)
mse_xgb = mean_squared_error(y_test, xgb_prediction)
rsq_xgb = r2_score(y_test, xgb_prediction)

print('MAE: %.3f' % mae_xgb)
print('MSE: %.3f' % mse_xgb)
print('R-Square: %.3f' % rsq_xgb)

"""**Saving the model**"""

import joblib

joblib.dump(xgb, 'model.joblib')
joblib.dump(author_encoder, 'author_encoder.joblib')
joblib.dump(geometry_encoder, 'geometry_encoder.joblib')