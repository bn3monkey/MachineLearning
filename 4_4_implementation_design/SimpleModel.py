import tensorflow as tf

class SimpleModel(object) :

    def __init__(self, n_in, n_hiddens, n_out) :
        self.n_in = n_in
        self.n_hiddens = n_hiddens
        self.n_out = n_out
        self.weights = []
        self.biases = []

        self._x = None
        self._t = None
        self._keep_prob = None
        self._sess = None
        self._history = {
            'accuracy' : [],
            'loss' : []
        }

    # f('W'x + b)
    def weight_variable(self, shape) :
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    #f(Wx + 'b')
    def bias_variable(self, shape) :
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    def inference(self, x, keep_prob) :
       y = x
       return y

    def loss(self, y, t) :
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(t*tf.log(y)), reduction_indices=[1])
        return cross_entropy

    def training(self, loss) :
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train_step = optimizer.minimize(loss)
        return train_step

    def accuracy(self, y, t) :
        return y

    def fit(self, X_train, Y_train):
        return self._history

    def evaluate(self, X_test, Y_test):
        return None

