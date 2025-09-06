import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Загружаем датасет IMDB
max_words = 10000 # ограничим словарь 10 тысячами наиболее частых слов
max_len = 200 #длина отзыва

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words = max_words)

#Приводим все отзывы к одинаковой длине

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen = max_len)

model = keras.Sequential([
    layers.Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    layers.Conv1D(64, 7, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    x_train, y_train,
    epochs = 5,
    batch_size = 128,
    validation_split = 0.2
)

loss, acc = model.evaluate(x_test, y_test)
print(f"Точность на тесте: {acc:.2f}")



