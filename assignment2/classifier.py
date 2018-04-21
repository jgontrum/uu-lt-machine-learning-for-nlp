import collections
import math
import pprint

import util

training_data = val_data = test_data = data = None


def main(reg_lambda, learning_rate, loss_function, regulariser,
         niterations=10, enable_test_set_scoring=False, **kwargs):
    global data

    # Type of features to use. This can be set to 'bigram' or 'unigram+bigram' to use
    # bigram features instead of or in addition to unigram features.
    # Not required for assignment.
    feature_type = 'unigram'

    # First test the parts to be implemented and warn if something's wrong.
    print('=============')
    print('SANITY CHECKS')
    print('=============')
    print()

    util.run_tests()

    # Load the data.
    training_data, val_data, test_data, data = load_data(feature_type)

    # Train the classifier.
    print('Starting training.')
    weights, bias, training_log = train(training_data, val_data,
                                        loss_function, regulariser,
                                        reg_lambda, learning_rate, niterations)
    print('Training completed.')

    print()
    print('=====================')
    print('MODEL CHARACTERISTICS')
    print('=====================')
    print()

    # Display some useful statistics about the model and the training process.
    title = 'Data set: %s - Regulariser(%s): %g - Learning rate: %g ' \
            '- Loss Function: %s' % (
                data.name, regulariser, reg_lambda, learning_rate,
                loss_function)

    print()

    # Get final accuracy
    val_predictions = predict(weights, bias, val_data)
    val_accuracy = accuracy(val_data.labels, val_predictions)

    print('Accuracy: %g' % val_accuracy)

    util.show_stats(title, training_log, weights, bias, data.vocabulary,
                    top_n=1,
                    write_to_file="results.csv",
                    configuration={
                        'reg_lambda': reg_lambda,
                        'learning_rate': learning_rate,
                        'loss_function': loss_function,
                        'regulariser': regulariser,
                        'niterations': niterations,
                        'val_accuracy': val_accuracy
                    })

    util.create_plots(title, training_log, weights, log_keys=['training_loss_reg', 'val_loss'])

    if enable_test_set_scoring:
        # Check the performance on the test set.
        test_loss = loss_function.unregularised_loss(weights, bias, test_data)
        test_predictions = predict(weights, bias, test_data)
        test_accuracy = accuracy(test_data.labels, test_predictions)

        print()
        print('====================')
        print('TEST SET PERFORMANCE')
        print('====================')
        print()
        print('Test loss: %g' % test_loss)
        print('Test accuracy: %g' % test_accuracy)


def predict(weights, bias, data):
    """Applies the classifier defined by the weights and the bias to the data and returns a list of predicted labels."""
    predictions = []
    for X in data.features():
        margin = sum(weights[feature] for feature in X) + bias
        if margin > 0:
            predictions.append(1.0)
        else:
            predictions.append(-1.0)

    return predictions


def train(training_data, val_data, loss_fn, regulariser, reg_lambda, learning_rate, niterations):
    """Trains a linear classifier.

    Arguments:
        training_data -- the training data set
        val_data -- the validation data set
        loss_fn -- an object implementing the loss function to use
        regulariser -- an object implementing the regulariser to use
        reg_lambda -- the regularisation strength (lambda)
        learning_rate -- the learning rate for gradient descent
        niterations -- number of iterations to run."""

    log = []
    nfeatures = len(training_data.vocabulary)

    # Initialise weights and bias to 0
    weights = [0.0 for _ in range(nfeatures)]
    bias = 0.0

    # Training loop
    for i in range(niterations):
        # Compute the gradients of the loss function and the regulariser separately
        loss_grads, bias_grad = loss_fn.gradients(weights, bias, training_data)
        reg_grads = regulariser.gradients(weights)

        # Add the gradients, weighting the regulariser gradient by lambda.
        weight_grads = [x + reg_lambda * y for x, y in zip(loss_grads, reg_grads)]

        # Perform the gradient descent update on the weights and the bias.
        weights, bias = gradient_descent_step(learning_rate,
                                              weights, bias,
                                              weight_grads, bias_grad)

        # Now we're done.
        # The rest of the loop just computes some useful info to monitor the training progress,
        # but it's not required for the algorithm to work.

        # Training and validation loss
        training_loss_unreg = loss_fn.unregularised_loss(weights, bias, training_data)
        training_loss_reg = training_loss_unreg + reg_lambda * regulariser.loss(weights)
        val_loss = loss_fn.unregularised_loss(weights, bias, val_data)

        # Size of the weights and the gradients
        param_norm = l2_norm(weights + [bias])
        grad_norm = l2_norm(weight_grads + [bias_grad])

        # Training and validation accuracy
        training_predictions = predict(weights, bias, training_data)
        training_accuracy = accuracy(training_data.labels, training_predictions)

        val_predictions = predict(weights, bias, val_data)
        val_accuracy = accuracy(val_data.labels, val_predictions)

        # All of these values are stored in a dictionary.
        log_record = collections.OrderedDict()
        log_record['training_loss_reg'] = training_loss_reg
        log_record['training_loss_unreg'] = training_loss_unreg
        log_record['training_acc'] = training_accuracy
        log_record['val_loss'] = val_loss
        log_record['val_acc'] = val_accuracy
        log_record['param_norm'] = param_norm
        log_record['grad_norm'] = grad_norm

        # Display info on training progress
        util.display_log_record(i, log_record)
        log.append(log_record)

    return weights, bias, log


def gradient_descent_step(learning_rate, old_weights, old_bias, weight_grads, bias_grad):
    """Performs a gradient descent update on the weights and the bias and returns the new values."""

    # Calculate the weight vector and the bias after a single gradient descent update
    # given the learning rate and gradients and return the new weights and bias.
    # Until you've implemented this correctly, we just return the old weights and bias
    # without updating.

    new_weights = list(
        map(
            lambda weight_gradient_pair: weight_gradient_pair[0] -
                                         weight_gradient_pair[1],
            zip(
                old_weights,
                map(
                    lambda gradient: gradient * learning_rate,
                    weight_grads
                )
            )
        )
    )

    new_bias = old_bias - learning_rate * bias_grad

    return new_weights, new_bias


def l2_norm(vector):
    """Computes the l2 norm of a vector."""
    return math.sqrt(sum(x * x for x in vector))


def accuracy(gold, hypothesis):
    """Computes an accuracy score given two vectors of labels."""
    return sum(g == h for g, h in zip(gold, hypothesis)) / len(gold)


class LogisticLoss:
    @staticmethod
    def unregularised_loss(weights, bias, data):
        """Computes the unregularised loss of the linear classifier on the provided data set."""
        loss = 0.0
        for X, y in zip(data.features(), data.labels):
            margin = sum(weights[feature] for feature in X) + bias
            loss += math.log(1.0 + math.exp(-y * margin))

        loss /= math.log(2)

        # Divide the loss by the number of examples to make it independent of the data size.
        loss /= len(data)
        return loss

    @staticmethod
    def gradients(weights, bias, data):
        """Computes the gradients of the loss function with respect to the weights and bias."""
        weight_gradients = [0.0 for _ in weights]
        bias_gradient = 0.0
        for X, y in zip(data.features(), data.labels):
            margin = sum(weights[feature] for feature in X) + bias
            bias_gradient += -1.0 / math.log(2) * y / (1.0 + math.exp(y * margin))
            for feature in X:
                weight_gradients[feature] += -1.0 / math.log(2) * y / (1.0 + math.exp(y * margin))

        # Divide the loss by the number of examples to make it independent of the data size.
        weight_gradients_per_example = [g / len(data) for g in weight_gradients]
        bias_gradient_per_example = bias_gradient / len(data)

        return weight_gradients_per_example, bias_gradient_per_example

    def __str__(self):
        return "LogisticLoss"


class HingeLoss:
    @staticmethod
    def unregularised_loss(weights, bias, data):
        """Computes the unregularised loss of the linear classifier on the provided data set."""
        loss = 0.0

        # Compute the hinge loss of the data given weights and bias
        for X, y in zip(data.features(), data.labels):
            margin = sum(weights[feature] for feature in X) + bias
            loss += 1 - y * margin if (y * margin) < 1 else 0

        # Divide the loss by the number of examples to make it independent of the data size.
        loss /= len(data)
        return loss

    @staticmethod
    def gradients(weights, bias, data):
        """Compute the gradients of the hinge loss with respect to the weights and bias"""
        weight_gradients = [0.0 for _ in weights]
        bias_gradient = 0.0

        for X, y in zip(data.features(), data.labels):
            margin = sum(weights[feature] for feature in X) + bias

            bias_gradient += -y if (y * margin) < 1 else 0
            for feature in X:
                weight_gradients[feature] += -y if (y * margin) < 1 else 0

        # Divide the loss by the number of examples to make it independent of the data size.
        weight_gradients_per_example = [g / len(data) for g in weight_gradients]
        bias_gradient_per_example = bias_gradient / len(data)

        return weight_gradients_per_example, bias_gradient_per_example

    def __str__(self):
        return "HingeLoss"

class L1Regulariser:
    @staticmethod
    def loss(weights):
        """Computes the loss term added by the regulariser (a scalar)."""
        return sum(abs(w) for w in weights)

    @staticmethod
    def gradients(weights):
        """Computes subgradients of the regularisation term with respect to the weights (a vector)."""
        return [1.0 if w > 0 else -1.0 for w in weights]

    def __str__(self):
        return "L1Regulariser"


class L2Regulariser:
    @staticmethod
    def loss(weights):
        """Computes the loss term added by the regulariser (a scalar)."""
        # Compute the value of the l2 regularisation term.
        reg_loss = sum(pow(w, 2) for w in weights)

        return reg_loss

    @staticmethod
    def gradients(weights):
        """Computes the gradients of the regularisation term with respect to the weights (a vector)."""
        # Compute the correct gradient vector corresponding to the l2 loss term.
        # The existing code just returns a vector of zeros. This should be replaced.
        reg_gradient = [2 * w for w in weights]

        return reg_gradient

    def __str__(self):
        return "L2Regulariser"


def load_data(feature_type):
    global training_data, val_data, test_data, data
    if training_data is not None:
        return training_data, val_data, test_data, data

    print()
    print('===================')
    print('CLASSIFIER TRAINING')
    print('===================')
    print()
    print('Loading data sets...')

    data_dir = '/local/kurs/ml/2017/assignment2/poldata/poldata.zip'

    data = util.load_movie_data(data_dir)

    data.select_feature_type(feature_type)

    # Split the data set randomly into training, validation and test sets.
    training_data, val_data, test_data = data.train_val_test_split()
    return training_data, val_data, test_data, data

if __name__ == '__main__':
    configurations = []

    configurations.append({
        "reg_lambda": 0.003,
        "learning_rate": 0.003,
        "loss_function": LogisticLoss(),
        "regulariser": L1Regulariser(),
        "niterations": 10,
        'enable_test_set_scoring': True
    })

    for configuration in configurations:
        main(**configuration)
        pprint.pprint(configuration, indent=2)
