import tensorflow as tf

class SimpleModel(object):

    def __init__(self, n_in, n_hiddens, n_out) :
        self.n_in = n_in
        self.n_hiddens = n_hiddens
        self.n_out = n_out
        self.weights = []
        self.biases = []

    # f('W'x + b)
    def weight_variable(self, shape) :
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    #f(Wx + 'b')
    def bias_variable(self, shape) :
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    def inference(self, x, keep_prob) :
       pass

    def loss(self, y, t) :
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(t*tf.log(y)), reduction_indices=[1])
        return cross_entropy

    def training(self, loss) :
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train_step = optimizer.minimize(loss)
        return train_step

    def accuracy(self, y, t) :
        #softmax
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(t,1))
        accu = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accu

    def fit(self, X_train, Y_train):
        pass

    def evaluate(self, X_test, Y_test):
        pass