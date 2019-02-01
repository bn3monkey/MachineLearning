from SoftMaxModel import *

if __name__ == '__main__' :
    
    model = SoftMaxModel(n_in = 784, n_hiddens = [200, 200, 200], n_out = 10)
    model.fit(X_train, Y_train)
    accuracy = model.evaluate(X_test, Y_test)
    print('accuracy : ', accuracy)