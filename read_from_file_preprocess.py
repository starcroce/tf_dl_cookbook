import tensorflow as tf

DATA_FILE = 'boston_housing.csv'
BATCH_SIZE = 10
NUM_FEATURES = 14


def data_generater(filename):
    f_queue = tf.train.string_input_producer(filename)
    reader = tf.TextLineReader(skip_header_lines=1)
    _, value = reader.read(f_queue)

    # We specify the data to use in case data is missing, and choose RM, PTRATIO and LSTAT
    record_defaults = [[0.0] for _ in range(NUM_FEATURES)]
    data = tf.decode_csv(value, record_defaults=record_defaults)
    features = tf.stack(tf.gather_nd(data, [[5], [10], [12]]))
    label = data[-1]

    min_after_deque = 10 * BATCH_SIZE
    capacity = 20 * BATCH_SIZE
    feature_batch, label_batch = tf.train.shuffle_batch(
        [features, label],
        batch_size=BATCH_SIZE,
        capacity=capacity,
        min_after_dequeue=min_after_deque
    )

    return feature_batch, label_batch


def generate_data(feature_batch, label_batch):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for _ in range(5):
            features, _ = sess.run([feature_batch, label_batch])
            print(features, 'HI')

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    feature_batch, label_batch = data_generater([DATA_FILE])
    generate_data(feature_batch, label_batch)
