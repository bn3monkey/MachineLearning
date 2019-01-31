import tensorflow as tf
import SimpleModel

class SoftMaxModel(SimpleModel) :
    
    def inference(self, x, keep_prob) :
        
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
            h = tf.relu(tf.matmul(input, self.weights[-1]) + self.biases[-1])
            output = tf.nn.dropout(h, keep_prob)
        
    
if __name__ == '__main__' :
    mModel = SoftMaxModel()