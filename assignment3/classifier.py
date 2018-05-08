import collections

import util

import tensorflow as tf


def main():
    # TRAINING HYPERPARAMETERS
    # Modify the following lines to change the training hyperparameters.

    # Regularisation strength
    reg_lambda = 0.001

    # Learning rate
    learning_rate = 0.001

    # Number of training iterations
    niterations = 15

    # Loss function to use (select one and comment out the other)
    loss_function = logistic_loss
    #loss_function = tf.losses....

    # Type of regularisation to use (select one and comment out the other)
    regulariser = tf.contrib.layers.l1_regularizer(reg_lambda)

    # This should only be enabled once you've decided on a final set of hyperparameters
    enable_test_set_scoring = False

    # Type of features to use. This can be set to 'bigram' or 'unigram+bigram' to use
    # bigram features instead of or in addition to unigram features.
    # Not required for assignment.
    feature_type = 'unigram'

    # END OF HYPERPARAMETERS

    # Load the data.

    print()
    print('===================')
    print('CLASSIFIER TRAINING')
    print('===================')
    print()
    print('Loading data sets...')

    #data_dir = '/local/kurs/ml/2017/assignment2/poldata/poldata.zip'
    data_dir = 'poldata.zip'
    data = util.load_movie_data(data_dir)

    data.select_feature_type(feature_type)

    # Split the data set randomly into training, validation and test sets.
    training_data, val_data, test_data = data.train_val_test_split()

    # Convert the sparse indices into dense vectors

    nfeatures = len(training_data.vocabulary)

    ds_training = util.sparse_to_dense(training_data, nfeatures)
    ds_val = util.sparse_to_dense(val_data, nfeatures)
    ds_test = util.sparse_to_dense(test_data, nfeatures)

    # Build the computational graph

    graph = tf.Graph()

    with graph.as_default():
        with tf.variable_scope('classifier'):

            # Define the placeholder where we feed in the data
            features = tf.placeholder(tf.int32, [None, nfeatures], name='input_placeholder')
            #labels = tf.placeholder

            # Define the weights of the classifier
            weights = tf.get_variable('weights', [nfeatures], initializer=tf.zeros_initializer())
            #bias = tf.get_variable

            # Two tensors must have same dtype and compatible shape for dot product
            features = tf.cast(features, tf.float32)
            exp_weights = tf.reshape(weights, [nfeatures, 1])
  
            # Compute dot product
            #logits = tf.matmul
            logits = tf.reshape(logits, [-1])
            # Define loss

            loss_ureg = loss_function(labels, logits)

            # Regularisation
            # L1_regularisation
            loss_reg = regulariser(weights)
            loss = loss_ureg + loss_reg

	    # Configuerate gradient descent
            config =

            # Initialiser 
            init = tf.global_variables_initializer()

    graph.finalize()

    # Train the classifier.
    print('Starting training.')

    #Define a training session and train the classifier

    with tf.Session(graph=graph) as sess:

        def predict(input_features):
            """Applies the classifier to the data and returns a list of predicted labels."""
            predictions = []
            pred = sess.run(logits, feed_dict={features: input_features})
            for x in pred:
                if x > 0:
                    predictions.append(1.0)
                else:
                    predictions.append(-1.0)
            return predictions

        # Before starting, initialize the variables.  We will 'run' this first.
        sess.run(init)
        training_log = []
        # Training iterations
        for i in range(niterations):
            _, t_loss_unreg, t_loss_reg = sess.run([config, loss_ureg, loss],
                                                   feed_dict={features: ds_training, labels: training_data.labels})

            v_loss = sess.run(loss, feed_dict={features: ds_val, labels:val_data.labels})

            training_predictions = predict(ds_training)
            training_accuracy = accuracy(training_data.labels, training_predictions)

            val_predictions = predict(ds_val)
            val_accuracy = accuracy(val_data.labels, val_predictions)

            log_record = collections.OrderedDict()
            log_record['training_loss_reg'] = t_loss_unreg
            log_record['training_loss_unreg'] = t_loss_reg
            log_record['training_acc'] = training_accuracy
            log_record['val_loss'] = v_loss
            log_record['val_acc'] = val_accuracy

            training_log.append(log_record)

            # Display info on training progress
            util.display_log_record(i, log_record)

        print('Training completed.')

        print()
        print('=====================')
        print('MODEL CHARACTERISTICS')
        print('=====================')
        print()

        # Display some useful statistics about the model and the training process.
        title = 'Data set: %s - Regulariser: %g - Learning rate: %g' % (data.name, reg_lambda, learning_rate)

        print()

        final_weights = sess.run(weights)
        final_bias = sess.run(bias)
        util.show_stats(title, training_log, final_weights, final_bias, data.vocabulary, top_n=20)
        #util.create_plots(title, training_log, weights, log_keys=['training_loss_reg', 'val_loss'])

        if enable_test_set_scoring:
            # Check the performance on the test set.
            test_loss = sess.run(loss, feed_dict={features: ds_test, labels:test_data.labels})
            test_predictions = predict(ds_test)
            test_accuracy = accuracy(test_data.labels, test_predictions)

            print()
            print('====================')
            print('TEST SET PERFORMANCE')
            print('====================')
            print()
            print('Test loss: %g' % test_loss)
            print('Test accuracy: %g' % test_accuracy)


def accuracy(gold, hypothesis):
    """Computes an accuracy score given two vectors of labels."""
    assert len(gold) == len(hypothesis)
    return sum(g == h for g, h in zip(gold, hypothesis)) / len(gold)


def logistic_loss(y, pred):
    y = tf.cast(y, tf.float32)
    pred = tf.cast(pred, tf.float32)
    return tf.reduce_mean(tf.log(1.0 + tf.exp(-y*pred)))


if __name__ == '__main__':
    main()
