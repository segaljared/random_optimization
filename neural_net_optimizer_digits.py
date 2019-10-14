import csv
import mlrose
import numpy
import time
from sklearn.impute import SimpleImputer
import sklearn.model_selection as sklearn_m
import sklearn.neural_network
import sklearn.datasets
import sklearn.preprocessing
import sklearn.metrics
import matplotlib.pyplot as plot
import math

class NeuralNetOptimizerDigits:
    def __init__(self, folds=5):
        features, classes = load_digits_data()
        test_size = int(len(features) * 0.3)
        self.training_features, self.test_features, self.training_classes, self.test_classes = sklearn_m.train_test_split(features, classes, test_size=test_size, train_size=(len(features) - test_size), random_state=50207)
        self.folds = folds
        self.fold_size = math.ceil(len(self.training_classes) / float(folds))

    def error_rate(self, predicted_classes, true_classes):
        one_class_indices = numpy.nonzero(true_classes == 1.0)
        zero_class_indices = numpy.nonzero(true_classes == 0.0)
        one_correct = numpy.sum(predicted_classes[one_class_indices] == 1.0) / len(one_class_indices)
        zero_correct = numpy.sum(predicted_classes[zero_class_indices] == 0.0) / len(zero_class_indices)
        return zero_correct, one_correct

    def gradient_descent(self):
        nn = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(10,), solver='sgd', max_iter=1000)

        start = time.time()
        nn.fit(self.training_features, self.training_classes)
        elapsed = time.time() - start
        print('SGD took %d ms' % (elapsed * 1000))

        predicted_classes = nn.predict(self.test_features)
        report = sklearn.metrics.classification_report(self.test_classes, predicted_classes)
        print(report)
        plot.ion()
        plot_roc_curves(self.test_classes, nn.predict_proba(self.test_features))
        plot.savefig('sgd_roc_curve')
        print(sklearn.metrics.confusion_matrix(self.test_classes, predicted_classes))

    def find_parameters_random_hill_climbing(self):
        plot.ion()
        def create_nn_restarts(restarts):
            return mlrose.NeuralNetwork(hidden_nodes=[10], algorithm='random_hill_climb', max_iters=1000, restarts=int(restarts), random_state=7106080)
        
        self.__test_parameter_values__(create_nn_restarts, numpy.arange(0,11),'# restarts', 'rhc_restarts_test')

        def create_nn_step_size(step_size):
            return mlrose.NeuralNetwork(hidden_nodes=[10], algorithm='random_hill_climb', max_iters=1000, learning_rate=step_size, random_state=7106080)

        self.__test_parameter_values__(create_nn_step_size, numpy.logspace(-2, 0, num=10),'step size', 'rhc_step_size_test', log_space=True)
        plot.ioff()

    def find_parameters_simulated_annealing(self):
        plot.ion()
        def create_nn_schedule(decay):
            if decay == 'Geom':
                schedule = mlrose.GeomDecay()
            if decay == 'Exp':
                schedule = mlrose.ExpDecay()
            if decay == 'Arith':
                schedule = mlrose.ArithDecay()
            return mlrose.NeuralNetwork(hidden_nodes=[10], algorithm='simulated_annealing', max_iters=1000, schedule=schedule, random_state=7106080)
        
        self.__test_parameter_values__(create_nn_schedule, ['Geom', 'Exp', 'Arith'],'temperature schedule', 'sa_schedule_test')

        def create_nn_step_size(step_size):
            return mlrose.NeuralNetwork(hidden_nodes=[10], algorithm='simulated_annealing', max_iters=1000, learning_rate=step_size, random_state=7106080)

        self.__test_parameter_values__(create_nn_step_size, numpy.logspace(-2, 0, num=10), 'step size', 'sa_step_size_test', log_space=True)
        plot.ioff()

    def find_parameters_genetic_algorithm(self):
        plot.ion()
        def create_nn_pop_size(pop_size):
            return mlrose.NeuralNetwork(hidden_nodes=[10], algorithm='genetic_alg', max_iters=200, pop_size=int(pop_size), random_state=7106080)
        
        self.__test_parameter_values__(create_nn_pop_size, numpy.arange(50, 350, 50), 'population size', 'ga_pop_size_test')

        def create_nn_mutation_prob(prob):
            return mlrose.NeuralNetwork(hidden_nodes=[10], algorithm='genetic_alg', max_iters=200, mutation_prob=prob, random_state=7106080)
        
        self.__test_parameter_values__(create_nn_mutation_prob, numpy.logspace(-2, -0.7, num=5), 'mutation probability', 'ga_mutation_prob_test', log_space=True)

        plot.ioff()

    def optimize_random_hill_climbing(self):
        nn = mlrose.NeuralNetwork(hidden_nodes=[10], algorithm='random_hill_climb', max_iters=25000, learning_rate=0.2, random_state=7106080, curve=True)

        hot_training_classes = to_array(self.training_classes, 10)
        start = time.time()
        nn.fit(self.training_features, hot_training_classes)
        elapsed = time.time() - start
        print('RHC took %d ms and %d iterations' % (elapsed * 1000, len(nn.fitness_curve)))

        predicted_classes = nn.predict(self.test_features)

        plot.clf()
        plot.plot(nn.fitness_curve)
        plot.ylabel('fitness')
        plot.xlabel('iterations')
        plot.pause(0.001)
        plot.savefig('rhc_fitness_curve_nn')

        predicted_classes = nn.predict(self.test_features)
        predicted_classes_not_array = from_array(predicted_classes)
        print(sklearn.metrics.classification_report(self.test_classes, predicted_classes_not_array))
        plot_roc_curves(self.test_classes, nn.predicted_probs)
        plot.savefig('rhc_roc_curve')
        print(sklearn.metrics.confusion_matrix(self.test_classes, predicted_classes_not_array))

    def optimize_genetic_algoirthm(self):
        nn = mlrose.NeuralNetwork(hidden_nodes=[10], algorithm='genetic_alg', pop_size=50, mutation_prob=0.02, max_iters=2500, random_state=7106080, curve=True)

        hot_training_classes = to_array(self.training_classes, 10)
        start = time.time()
        nn.fit(self.training_features, hot_training_classes)
        elapsed = time.time() - start
        print('GA took %d ms and %d iterations' % (elapsed * 1000, len(nn.fitness_curve)))

        plot.clf()
        plot.plot(nn.fitness_curve)
        plot.ylabel('fitness')
        plot.xlabel('iterations')
        plot.pause(0.001)
        plot.savefig('ga_fitness_curve_nn')

        predicted_classes = nn.predict(self.test_features)
        predicted_classes_not_array = from_array(predicted_classes)
        print(sklearn.metrics.classification_report(self.test_classes, predicted_classes_not_array))
        plot_roc_curves(self.test_classes, nn.predicted_probs)
        plot.savefig('ga_roc_curve')
        print(sklearn.metrics.confusion_matrix(self.test_classes, predicted_classes_not_array))

    def optimize_simulated_annealing(self):
        nn = mlrose.NeuralNetwork(hidden_nodes=[10], algorithm='simulated_annealing', max_iters=25000, learning_rate=0.5, random_state=7106080, curve=True)

        hot_training_classes = to_array(self.training_classes, 10)
        start = time.time()
        nn.fit(self.training_features, hot_training_classes)
        elapsed = time.time() - start
        print('SA took %d ms and %d iterations' % (elapsed * 1000, len(nn.fitness_curve)))

        plot.clf()
        plot.plot(nn.fitness_curve)
        plot.ylabel('fitness')
        plot.xlabel('iterations')
        plot.pause(0.001)
        plot.savefig('sa_fitness_curve_nn')

        predicted_classes = nn.predict(self.test_features)
        predicted_classes_not_array = from_array(predicted_classes)
        print(sklearn.metrics.classification_report(self.test_classes, predicted_classes_not_array))
        plot_roc_curves(self.test_classes, nn.predicted_probs)
        plot.savefig('sa_roc_curve')
        print(sklearn.metrics.confusion_matrix(self.test_classes, predicted_classes_not_array))

        # based off of: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        # false_positives = {}
        # true_positives = {}
        # roc_auc = {}
        # matrix_true_classes = numpy.array(to_array(self.test_classes, 10))
        # for i in range(0, 10):
        #     false_positives[i], true_positives[i], _ = sklearn.metrics.roc_curve(matrix_true_classes[:,i], predicted_classes[:,i])
        #     roc_auc[i] = sklearn.metrics.auc(false_positives[i], true_positives[i])

        # plot.title('ROC Curve')
        # plot.xlabel('false positive rate')
        # plot.ylabel('true positive rate')
        # colors = ['lightcoral', 'firebrick', 'orangered', 'chocolate', 'gold', 'lightgreen', 'seagreen', 'deepskyblue', 'lightskyblue', 'navy']
        # plot.plot([0,1], [0, 1], ls='--', lw=2)
        # for i in range(0, 10):
        #     plot.plot(false_positives[i], true_positives[i], color=colors[i], lw=2, label='class %d' % i)
        # plot.legend(loc='best')
        # plot.show()

    def __simulated_annealing__(self, learning_rate):
        total_correct = []
        for fold in range(0, self.folds):
            nn = mlrose.NeuralNetwork(hidden_nodes=[5], algorithm='simulated_annealing', random_state=7106080)

            training_features, validation_features, training_classes, validation_classes = self.__split_fold__(fold)
            nn.fit(training_features, training_classes)

            predicted_classes = nn.predict(validation_features)
            predicted_classes = numpy.reshape(predicted_classes, len(validation_classes))
            correct =  numpy.sum(numpy.array(predicted_classes) == numpy.array(validation_classes)) / len(validation_classes)
            total_correct.append(correct)
        return numpy.mean(total_correct)

    def __split_fold__(self, fold):
        validation_start = self.fold_size * fold
        validation_end = min(validation_start + self.fold_size, len(self.training_classes) - 1)
        if validation_start > 0:
            training_indices = numpy.arange(0, validation_start)
            if validation_end + 1 < len(self.training_classes):
                training_indices = numpy.append(training_indices, numpy.arange(validation_end, len(self.training_classes)))
        else:
            training_indices = numpy.arange(validation_end, len(self.training_classes))
        return (self.training_features[training_indices], self.training_features[validation_start:validation_end],
            self.training_classes[training_indices], self.training_classes[validation_start:validation_end])

    def __test_parameter_values__(self, create_nn, parameter_values, xlabel, save_name, log_space=False):
        validation_size = int(len(self.training_classes) * 0.2)
        training_features, validation_features, training_classes, validation_classes = sklearn_m.train_test_split(self.training_features, self.training_classes, test_size=validation_size, train_size=(len(self.training_features) - validation_size), random_state=7832086)
        hot_training_classes = to_array(training_classes, 10)

        accuracy = []
        for i, value in enumerate(parameter_values):
            print('Testing [%d/%d]' % (i+1, len(parameter_values)))
            nn = create_nn(value)
            nn.fit(training_features, hot_training_classes)

            predicted_classes = nn.predict(validation_features)
            predicted_classes_not_array = from_array(predicted_classes)
            param_accuracy = numpy.sum(numpy.array(predicted_classes_not_array) == numpy.array(validation_classes)) / len(validation_classes)
            print('Accuracy: %f' % param_accuracy)
            accuracy.append(param_accuracy)
        
        plot.clf()
        plot.xlabel(xlabel)
        plot.ylabel('accuracy')
        plot.plot(parameter_values, accuracy)
        if log_space:
            plot.xscale('log')
        plot.pause(0.001)
        plot.savefig(save_name)

def load_digits_data():
    digits = sklearn.datasets.load_digits()
    # flattens the images
    features = digits.images.reshape((len(digits.images), -1))
    classes = digits.target
    return features, classes

def to_array(classes, num_classes):
    array_classes = []
    for sample_class in classes:
        class_array = [0.0] * num_classes
        class_array[sample_class] = 1.0
        array_classes.append(class_array)
    return array_classes

def from_array(array_classes):
    classes = []
    for class_array in array_classes:
        classes.append(numpy.argmax(class_array))
    return classes

def plot_roc_curves(true_classes, predicted_probs):
        # based off of: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        plot.clf()
        false_positives = {}
        true_positives = {}
        roc_auc = {}
        matrix_true_classes = numpy.array(to_array(true_classes, 10))
        for i in range(0, 10):
            false_positives[i], true_positives[i], _ = sklearn.metrics.roc_curve(matrix_true_classes[:,i], predicted_probs[:,i])
            roc_auc[i] = sklearn.metrics.auc(false_positives[i], true_positives[i])

        plot.title('ROC Curve')
        plot.xlabel('false positive rate')
        plot.ylabel('true positive rate')
        colors = ['lightcoral', 'firebrick', 'orangered', 'chocolate', 'gold', 'lightgreen', 'seagreen', 'deepskyblue', 'lightskyblue', 'navy']
        plot.plot([0,1], [0, 1], ls='--', lw=2)
        for i in range(0, 10):
            plot.plot(false_positives[i], true_positives[i], color=colors[i], lw=2, label='class %d (area = %f)' % (i, roc_auc[i]))
        plot.legend(loc='best')
        plot.pause(0.001)
