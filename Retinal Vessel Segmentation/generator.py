import tensorflow as tf

IMAGE_SIZE = 512

def generator(image):
    
    def read_files(image_path, mask=False):
        image = tf.io.read_file(image_path)
        if mask:
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.divide(image, 255)
            image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])
            image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
            image = tf.cast(image, tf.int32)
        else:
            image = tf.image.decode_png(image, channels=4)
            image = image[:,:,:3]
            image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
            image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
            image = image / 255.
        return image

    def load_data(image_list, mask_list):
        image = read_files(image_list)
        mask  = read_files(mask_list, mask=True)
        return image, mask

    def data_generator(image_list, mask_list):
        dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
        dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(1, drop_remainder=False)
        return dataset
    
    return data_generator([image], [image])