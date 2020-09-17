# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 18:43:03 2020

@author: jenny
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


train = pd.read_csv(os.path.join("train.csv"), header=0)
test = pd.read_csv(os.path.join("test.csv"), header=0)
print(train.shape, test.shape)


def data_preprocessing(data):
    # one hot encode Sex
    data['Sex'] = [np.where(sex=="female", 0, 1) for sex in data['Sex']]
    data['Sex'] = data['Sex'].astype('int')
    
    
    # # one hot encode Embarked
    # print(data.groupby(['Embarked']).size())
    # embark = dict(C=0, Q=1, S=2)
    # tmp = [embark[i] for i in (data['Embarked'].fillna('S'))]
    # tmp = tf.one_hot(tmp, 3).numpy().astype('int')
    # data = data.drop(labels=['Embarked'], axis=1)
    # data['Embarked_C'], data['Embarked_Q'], data['Embarked_S'] = \
    #     tmp[:,0], tmp[:,1], tmp[:,2]
    
    # after investigating encodded Embarked location and relationship with 
    # others through correlation table, the feature seems to be replaceable
    # by other features such as fair and pclass.
    # logically it tells us which is wealthier area, and age distribution
    # differences in these area. But doesn't contribute much on survival rate,
    # it's more related to where you at (cabin/pclass) on board
    data = data.drop(['Embarked'], axis=1)
    
    
    # remove prefix in Ticket
    tmp = []
    for row in data['Ticket']:
        if " " in row:
            for i in range(len(row)-1, 0, -1):
                if row[i-1]==" ":
                    tmp.append(int(row[i:]))
                    break
        
        elif "LINE" in row:
            tmp.append(int(min(data['Ticket']))-1)
        else:
            tmp.append(int(row))
    data['Ticket'] = tmp
            
    # Or drop Ticket????
    # data = data.drop(['Ticket'])
    
    
    # Drop Cabin, too many nan (687) and doesn't provide too much info
    print("instances with no Cabin value:", sum(data['Cabin'].isnull()))
    data = data.drop(['Cabin'], axis=1)
    
    
    # find which feature has high correlation relationship with Pclass
    print("instances with no Age value:", sum(data['Age'].isnull()))
    print(data.groupby('Pclass').describe()['Age'])
    print(data.corr()['Age'])
    # pd.plotting.scatter_matrix(data)
    # plt.show()
    # use mean Age of Pclass to fill up missing value
    tmp = data.groupby('Pclass').describe()['Age']['mean']
    for row in data.index[data['Age'].isnull() == True]:
        data['Age'][row] = tmp[data['Pclass'][row]]
    
    
    # drop name
    data = data.drop(['PassengerId', 'Name'], axis=1)
    
    
    # Pclass
    print(data.groupby(['Pclass']).size())
    
    
    # SibSp and ParCh can be generalized as just family member
    data['famMem'] = data['SibSp'] + data['Parch']
    data = data.drop(['SibSp', 'Parch'], axis=1)
    
    
    print(data.corr())
    
    return data

train = data_preprocessing(train)
# Survived as label
train_label = train['Survived'].copy().to_numpy()
train_data = train.drop(['Survived'], axis=1).to_numpy()


# DNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * 10**(epoch/20))
model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])
history = model.fit(train_data, train_label, epochs=150)#, callbacks=[lr_scheduler])
"""
Epoch 145/150
891/891 [==============================] - 0s 130us/sample - loss: 0.4385 - accuracy: 0.8081
Epoch 146/150
891/891 [==============================] - 0s 137us/sample - loss: 0.4420 - accuracy: 0.8070
Epoch 147/150
891/891 [==============================] - 0s 149us/sample - loss: 0.4447 - accuracy: 0.8081
Epoch 148/150
891/891 [==============================] - 0s 133us/sample - loss: 0.4364 - accuracy: 0.8159
Epoch 149/150
891/891 [==============================] - 0s 131us/sample - loss: 0.4387 - accuracy: 0.8025
Epoch 150/150
891/891 [==============================] - 0s 129us/sample - loss: 0.4389 - accuracy: 0.8182
"""

# plt.figure(figsize=(10,6))
# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-4, 0, 20])
# plt.xlabel("learning rate")
# plt.ylabel("loss")


plt.figure()
plt.plot(history.history['accuracy'])
plt.title("Accuracy")
plt.figure()
plt.plot(history.history['loss'])
plt.title("Loss")
plt.show()

test_data = data_preprocessing(test).to_numpy()
prediction = model.predict(test_data)
with open("prediction_dnn.csv", 'w') as file:
    file.write("PassengerId,Survived\n")
    for i in range(len(prediction)):
        survived = np.where(prediction[i][0]>0.5, 1, 0)
        input_str = str(test['PassengerId'][i]) + "," + str(survived) + "\n"
        file.write(input_str)
    
