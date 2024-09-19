#!/usr/bin/env python
# coding: utf-8

# In[24]:


pip install pydot


# In[4]:


import sys

assert sys.version_info >= (3, 7)


# In[5]:


from packaging import version
import sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")


# In[6]:


import tensorflow as tf

assert version.parse(tf.__version__) >= version.parse("2.8.0")


# In[10]:


import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]


# In[11]:


X_train.shape


# In[12]:


X_train.dtype


# In[13]:


X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.


# In[14]:


import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()


# In[15]:


y_train


# In[16]:


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# In[17]:


class_names[y_train[0]]


# In[18]:


tf.random.set_seed(42)
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=[28, 28]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))


# In[19]:


tf.keras.backend.clear_session()
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])


# In[20]:


model.summary()


# In[22]:


model.layers


# In[23]:


hidden1 = model.layers[1]
hidden1.name


# In[24]:


model.get_layer('dense') is hidden1


# In[25]:


weights, biases = hidden1.get_weights()
weights


# In[26]:


weights.shape


# In[27]:


biases


# In[28]:


biases.shape


# In[29]:


model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])


# In[30]:


tf.keras.utils.to_categorical([0, 5, 1, 0], num_classes=10)


# In[33]:


import numpy as np
np.argmax(
    [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
     [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
     [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
    axis=1
)


# In[34]:


history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))


# In[35]:


history.params


# In[36]:


print(history.epoch)


# In[38]:


import matplotlib.pyplot as plt
import pandas as pd

pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.show()


# In[39]:


plt.figure(figsize=(8, 5))
for key, style in zip(history.history, ["r--", "r--.", "b-", "b-*"]):
    epochs = np.array(history.epoch) + (0 if key.startswith("val_") else -0.5)
    plt.plot(epochs, history.history[key], style, label=key)
plt.xlabel("Epoch")
plt.axis([-0.5, 29, 0., 1])
plt.legend(loc="lower left")
plt.grid()
plt.show()


# In[40]:


model.evaluate(X_test, y_test)


# In[41]:


X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)


# In[42]:


y_pred = y_proba.argmax(axis=-1)
y_pred


# In[43]:


np.array(class_names)[y_pred]


# In[44]:


y_new = y_test[:3]
y_new


# In[46]:


plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(class_names[y_test[index]])
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()


# In[ ]:




