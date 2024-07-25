#test
import tensorflow as tf
import keras
from tensorflow import keras
from keras import layers, models
import pathlib
from typing import Self
import tensorflow as tf
import keras 
from keras import datasets,layers,models
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models,Model
import pathlib
import os
from PIL import Image



# Bozuk resimleri kontrol eden ve silen bir işlev
def remove_corrupt_images(directory):
    num_skipped = 0
    for folder_name in ("Dog", "Cat"):
        folder_path = os.path.join(directory, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                # Resmi aç ve kanalları kontrol et
                with Image.open(fpath) as img:
                    img.verify()  # Resim bozuksa burada hata verir
                    if img.mode not in ('RGB', 'L'):  # Kanal kontrolü
                        raise ValueError(f"Invalid image mode: {img.mode}")
            except (IOError, SyntaxError, ValueError) as e:
                print(f"Siliniyor: {fpath} ({e})")
                num_skipped += 1
                os.remove(fpath)
    print(f"Toplamda {num_skipped} resim silindi.")

# Veri kümesi ayarları
batch_size = 32
img_height = 128
img_width = 128
DATASET_PATH = pathlib.Path("C:\\Users\\Hazar\\resim")

# Bozuk resimleri kaldırma
remove_corrupt_images(DATASET_PATH)

# Eğitim ve doğrulama veri kümelerini yükleme
train_dataset = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

validation_dataset = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Sınıf isimlerini yazdırma
class_names = train_dataset.class_names
print("Sınıf İsimleri:", class_names)

# Normalizasyon katmanı
normalization_layer = layers.Rescaling(1./255)

# Veri kümesine normalizasyon uygulama
normalized_train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
normalized_validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

# Modelin tanımlanması
model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')  # Sınıf sayısına göre çıktı katmanı
])

# Modeli derleme
model.compile(optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])

# Modeli eğitme
model.fit(
    normalized_train_dataset,
    validation_data=normalized_validation_dataset,
    epochs=3
)

# Modeli kaydetme
# tf.keras.models.save_model(
#     model, filePathString, overwrite=True,
#     include_optimizer=True, save_format=None,
#     signatures=None, options=None)


import numpy as np
import keras
from keras import ops

import tensorflow as tf
import os
import shutil
from PIL import Image
import numpy as np
# Örnek görselleştirme fonksiyonu
def show_image(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label.numpy())
    plt.axis('off')
    plt.show()

# Veri setindeki ilk birkaç görüntüyü ve etiketlerini gösterme
for images, labels in train_dataset.take(1):
    for i in range(5):  # İlk 5 görüntü
        show_image(images[i], labels[i])
        
model.export("cat_dog_classifier_model")
print("a")

# Eğitilmiş modeli yükleme
keras.layers.TFSMLayer('cat_dog_classifier_model')
print("b")
# model = keras.models.load_model('cat_dog_classifier_model')


# Yeni resmi sınıflandırma ve taşıma işlevi
def classify_and_move_image(image_path, model, class_names, base_path):
    img = Image.open(image_path)
    img = img.resize((img_height, img_width))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Batch boyutunu ekle

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    # Yeni dosya yolunu oluştur
    new_file_path = os.path.join(base_path, predicted_class, os.path.basename(image_path))
    
    # Dosyayı taşı
    shutil.move(image_path, new_file_path)
    print(f"{os.path.basename(image_path)} dosyası {predicted_class} klasörüne taşındı.")

# Kullanım örneği
# Yeni resimleri sınıflandırmak için aşağıdaki kodu kullanabilirsiniz.
new_image_path = "C:\\Users\\Hazar\\NewImages\\3169.jpg"
classify_and_move_image(new_image_path, model, class_names, "C:\\Users\\Hazar\\resim")

