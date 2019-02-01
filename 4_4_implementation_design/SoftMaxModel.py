import tensorflow as tf
from sklearn.utils import shuffle
from SimpleModel import *

class SoftMaxModel(SimpleModel) :
    
    def inference(self, x, keep_prob) :
        
        output = []
        for i, n_hidden in enumerate(self.n_hiddens) :
            if i == 0 :
                input = x
                input_dim = self.n_in
            else :
                input = output
                input_dim = self.n_hiddens[i-1]
        
            self.weights.append(self.weight_variable([input_dim, n_hidden]))
            self.biases.append(self.bias_variable([n_hidden]))
            # f( weight * input + biases)
            # last vector of weighs and biases
            h = tf.nn.relu(tf.matmul(input, self.weights[-1]) + self.biases[-1])
            output = tf.nn.dropout(h, keep_prob)
        
        self.weights.append(self.weight_variable([self.n_hiddens[-1], n_out]))
        self.biases.append(self.bias_variable([self.n_out]))
        # f( weight * input + biases)
        # last vector of weighs and biases    
        y = tf.nn.softmax(tf.matmul(output, self.weights[-1]) + self.biases[-1])
        return y
    
    def accuracy(self, y, t) :
        #softmax
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(t,1))
        accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy_

    def fit(self, X_train, Y_train, epochs = 100, batch_size = 100, p_keep = 0.5, verbose = 1) :
        
        x = tf.placeholder(tf.float32, shape=[None, self.n_in])
        t = tf.placeholder(tf.float32, shape=[None, self.n_out])
        keep_prob = tf.placeholder(tf.float32)

        # for evaluation
        self._x = x
        self._t = t
        self._keep_prob = keep_prob

        y = self.inference(x, keep_prob)
        loss = self.loss(y, t)
        train_step = self.training(loss)
        accuracy = self.accuracy(t,f)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # for evaluation
        self._sess = sess

        N_train = len(X_train)
        n_batches = N_train // batch_size

        for epoch in range(epochs) :
            X_, Y_ = shuffle(X_train, Y_train)

            for i in range(n_batches) : 
                start = i * batch_size
                end = start + batch_size
                sess.run(train_step, feed_dict = {
                    x : X_[start:end],
                    t : Y_[start:end],
                    keep_prob : p_keep
                })
            
            loss_ = loss.eval(session = sess, feed_dict = {
                x: X_train,
                y: Y_train,
                keep_prob : 1.0
            })

            accuracy_ = accuracy.eval(session=sess, feed_dict = {
                x: X_train,
                y: Y_train,
                keep_prob : 1.0
            })

            self._history['loss'].append(loss_)
            self._history['accuracy'].append(accuracy_)

            if verbose :
                print('epoch : ', epoch,
                      ' loss : ', loss_,
                      ' accuracy : ', accuracy_)

        return self._history

    def evaluate(self, X_test, Y_test) :
        return self.accuracy.eval(session = self._sess, feed_dict = {
            self._x : X_test,
            self._y : Y_test,
            self._keep_prob : 1.0
        })

