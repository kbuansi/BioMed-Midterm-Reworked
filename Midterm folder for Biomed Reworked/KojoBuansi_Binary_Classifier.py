#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[4]:


from tensorflow.keras.applications import VGG16


# In[5]:


from tensorflow.keras.models import Sequential


# In[6]:


from tensorflow.keras.layers import Dense, Flatten, Dropout


# In[7]:


from tensorflow.keras.optimizers import Adam


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


train_dir = r'C:\Users\Kojo PC\Dropbox\PC (3)\Downloads\Data1\train'
val_dir = r'C:\Users\Kojo PC\Dropbox\PC (3)\Downloads\Data1\test'


# In[10]:


batch_size = 16
num_epochs = 10
learning_rate = 0.0001


# In[11]:


input_shape = (128, 128, 3)


# In[12]:


vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)


# In[13]:


vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)


# In[15]:


model = Sequential()
model.add(vgg)


# In[16]:


model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[17]:


opt = Adam(lr=learning_rate)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# In[18]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)


# In[19]:


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen


# In[21]:


val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)


# In[22]:


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=num_epochs,
    validation_data=val_generator,
    validation_steps=val_generator.n // batch_size
)

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

