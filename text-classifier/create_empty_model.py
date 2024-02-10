import tensorflow as tf
import re
import string

keras = tf.keras
layers = keras.layers
losses = keras.losses

embedding_dim = 16
max_features = 10000
sequence_length = 250

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')

vectorize_layer = layers.TextVectorization(
      standardize=custom_standardization,
      max_tokens=max_features,
      output_mode='int',
      output_sequence_length=sequence_length)

def create_model():
    model = tf.keras.Sequential([
        layers.Embedding(max_features, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1)])

    model.summary()

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                    optimizer='adam',
                    metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    export_model = tf.keras.Sequential([
        vectorize_layer,
        model,
        layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )

    return export_model