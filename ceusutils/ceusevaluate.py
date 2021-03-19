# model.evaluate(val_ds1)

#test_img = "E:/MY/My Databases/MEDICAL/CEUS/UMF/DBV50LEZIUNI - FNH1/test/FNH/FNH1a244.jpg"

# for test_img in val_ds.file_paths:

#     img = tf.keras.preprocessing.image.load_img(
#         test_img, target_size=(img_height, img_width)
#     )
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0) # Create a batch

#     predictions = model.predict(img_array)
#     score = tf.nn.softmax(predictions[0])

#     print(
#         "This image most likely belongs to {} with a {:.2f} percent confidence."
#         .format(class_names[np.argmax(score)], 100 * np.max(score))
#     )