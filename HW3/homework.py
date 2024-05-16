import numpy as np
import pandas as pd

class NeuralNetwork:
    '''
    Multi-class cassification neural network with following features:
    1. Weights initialization - He initialization method
    2. Optimization method: ADAM
    3. metrics -loss: Categorical cross entropy loss
    4. Activation functions: ReLU, softmax
    5. Hyper parameters: learning_rate, epochs, batch_size, # hidden_layers

    '''
    def __init__(self, input_size, hidden_layer_sizes, output_size,epochs=100, learning_rate=0.01, batch_size=32):
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.num_hidden_layers = len(hidden_layer_sizes)

        self.epochs=epochs
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        
        self.weights = {}
        self.biases = {}
        self.activations = {}
        self.gradients = {}
        self.moments = {}

        self.betas=[0.9,0.999] 
        self.epsilon = 1e-8  
        
        #He Initialization method for initial weights of the neural network
        self.weights['w1'] = np.random.randn(input_size, hidden_layer_sizes[0]) * np.sqrt(2.0 / input_size)
        self.biases['b1'] = np.zeros((1, hidden_layer_sizes[0]))
        
        for i in range(1, self.num_hidden_layers):
            self.weights[f'w{i+1}'] = np.random.randn(hidden_layer_sizes[i-1], hidden_layer_sizes[i]) * np.sqrt(2.0 / hidden_layer_sizes[i-1])
            self.biases[f'b{i+1}'] = np.zeros((1, hidden_layer_sizes[i]))
            
        self.weights[f'w{self.num_hidden_layers+1}'] = np.random.randn(hidden_layer_sizes[-1], output_size) * np.sqrt(2.0 / hidden_layer_sizes[-1])
        self.biases[f'b{self.num_hidden_layers+1}'] = np.zeros((1, output_size))

    
    def forward_propagation(self, X):
        self.activations['A0'] = X
        
        for i in range(self.num_hidden_layers):
            self.activations[f'Z{i+1}'] = np.dot(self.activations[f'A{i}'], self.weights[f'w{i+1}']) + self.biases[f'b{i+1}']
            self.activations[f'A{i+1}'] = self.relu(self.activations[f'Z{i+1}'])
        
        output = np.dot(self.activations[f'A{self.num_hidden_layers}'], self.weights[f'w{self.num_hidden_layers+1}']) + self.biases[f'b{self.num_hidden_layers+1}']
        return self.softmax(output)
    
    def backward_propagation(self, X, y, learning_rate=0.01, batch_size=32):
        m = X.shape[0]
        num_batches = m // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]

            op_proba = self.forward_propagation(X_batch)

            self.gradients['dA' + str(self.num_hidden_layers + 1)] = (op_proba - y_batch) / batch_size
            for l in range(self.num_hidden_layers, 0, -1):
                self.gradients['dZ' + str(l)] = np.dot(self.gradients['dA' + str(l + 1)], self.weights['w' + str(l + 1)].T)
                self.gradients['dA' + str(l)] = self.gradients['dZ' + str(l)] * (self.activations['Z' + str(l)] > 0)
                self.gradients['dW' + str(l)] = np.dot(self.activations['A' + str(l - 1)].T, self.gradients['dA' + str(l)])
                self.gradients['db' + str(l)] = np.sum(self.gradients['dA' + str(l)], axis=0, keepdims=True)

                if 'm_dw' + str(l) not in self.moments:
                    self.moments['m_dw' + str(l)] = np.zeros_like(self.gradients['dW' + str(l)])
                    self.moments['m_db' + str(l)] = np.zeros_like(self.gradients['db' + str(l)])
                    self.moments['v_dw' + str(l)] = np.zeros_like(self.gradients['dW' + str(l)])
                    self.moments['v_db' + str(l)] = np.zeros_like(self.gradients['db' + str(l)])
                
                self.moments['m_dw' + str(l)] = self.betas[0] * self.moments['m_dw' + str(l)] + (1 - self.betas[0]) * self.gradients['dW' + str(l)]
                self.moments['m_db' + str(l)] = self.betas[0] * self.moments['m_db' + str(l)] + (1 - self.betas[0]) * self.gradients['db' + str(l)]
                self.moments['v_dw' + str(l)] = self.betas[1] * self.moments['v_dw' + str(l)] + (1 - self.betas[1]) * np.square(self.gradients['dW' + str(l)])
                self.moments['v_db' + str(l)] = self.betas[1] * self.moments['v_db' + str(l)] + (1 - self.betas[1]) * np.square(self.gradients['db' + str(l)])
                
                m_dw_update = self.moments['m_dw' + str(l)] / (1 - self.betas[0] ** (i + 1))
                m_db_update = self.moments['m_db' + str(l)] / (1 - self.betas[0] ** (i + 1))
                v_dw_update = self.moments['v_dw' + str(l)] / (1 - self.betas[1] ** (i + 1))
                v_db_update = self.moments['v_db' + str(l)] / (1 - self.betas[1] ** (i + 1))

                self.weights['w' + str(l)] -= learning_rate * m_dw_update / (np.sqrt(v_dw_update) + self.epsilon)
                self.biases['b' + str(l)] -= learning_rate * m_db_update / (np.sqrt(v_db_update) + self.epsilon)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return values / np.sum(values, axis=1, keepdims=True)
    
    def categorical_crossentropy_loss(self, y_true, y_pred):
        val = np.sum(y_true, axis=0) > 0
        val_y_true = y_true[:, val]
        val_y_pred = y_pred[:, val]
        return -np.sum(val_y_true * np.log(val_y_pred + 1e-8)) / y_true.shape[0]
    
    def accuracy(self, y_true, y_pred):
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    
    def train(self, X, y):
        for epoch in range(self.epochs):
            self.backward_propagation(X, y, self.learning_rate, self.batch_size)
            y_pred = self.forward_propagation(X)
            loss = self.categorical_crossentropy_loss(y, y_pred)
            acc = self.accuracy(y, y_pred)
            #print(f"Epoch {epoch+1}/{self.epochs} - loss: {loss}, accuracy: {acc}")
    
    def predict(self, X):
        return np.argmax(self.forward_propagation(X), axis=1)
    

if __name__ == '__main__':
    X_train=pd.read_csv("train_data.csv")
    y_train=np.array(pd.read_csv("train_label.csv"))
    
    #onehot encoding y_train
    if y_train.shape and len(y_train.shape) > 1 and y_train.shape[-1] == 1: 
        ip = tuple(y_train.shape[:-1])
    y_train = y_train.reshape(-1)
    r = y_train.shape[0]
    c = 1 + np.max(y_train) 
    y_train_onehot = np.zeros((r, c))
    y_train_onehot[np.arange(r), y_train] = 1
    y_train_onehot = np.reshape(y_train_onehot, ip + (c,))

    #data preprocessing
    sub_locality={"New York": "New York County",
                  "Dumbo": "Kings County", 
                "Snyder Avenue": "Kings County", 
                "Kings County": "Kings County",  
                "Queens County": "Queens County",
                "Queens": "Queens County",
                "Richmond County": "Richmond County", 
                "Brooklyn": "Kings County",
                "Flushing": "Queens County", 
                "Coney Island": "Kings County", 
                "East Bronx": "Bronx County", 
                "Brooklyn Heights": "Kings County", 
                "Jackson Heights": "Queens County",  
                "Rego Park": "Queens County",  
                "Fort Hamilton": "Kings County", 
                "Bronx County": "Bronx County",
                "New York County": "New York County",
                "The Bronx": "Bronx County",
                "Staten Island": "Richmond County",
                "Manhattan": "New York County",
                "Riverdale": "Bronx County",  
                }
    
    X_train["SUBLOCALITY"] = X_train["SUBLOCALITY"].apply(lambda row: sub_locality[row])
    X_train=X_train.drop(["ADDRESS","MAIN_ADDRESS","FORMATTED_ADDRESS", "LONG_NAME", "STREET_NAME", "ADMINISTRATIVE_AREA_LEVEL_2", "BROKERTITLE", "STATE", "LOCALITY"],axis=1)

    #encoding categorical columns
    categorical_columns=["TYPE","SUBLOCALITY"]
    type_unique_train = X_train["TYPE"].unique()
    sublocality_unique_train= X_train["SUBLOCALITY"].unique()

    X_train_encoded_df = pd.DataFrame()

    for value in type_unique_train:
        X_train_encoded_df[f"TYPE_{value}"] = (X_train["TYPE"] == value).astype(int)

    for value in sublocality_unique_train:
        X_train_encoded_df[f"SUBLOCALITY_{value}"] = (X_train["SUBLOCALITY"] == value).astype(int)
    
    X_train = pd.concat([X_train, X_train_encoded_df], axis=1)
    X_train.drop(columns=categorical_columns, inplace=True)
    #print(X_train.shape)

    #Robustscaling
    for col in ['PRICE', 'BATH', 'PROPERTYSQFT', 'LATITUDE','LONGITUDE']:
        median=np.median(X_train[col])
        fq=np.quantile(X_train[col],0.25)
        tq=np.quantile(X_train[col],0.75)
        iqr=tq-fq
        if iqr==0:
            iqr=1
        X_train[col]=(X_train[col]-median)/iqr
    
    input_size = X_train.shape[1]
    output_size = y_train_onehot.shape[1]
    hidden_sizes =[128, 64, 64]
    learning_rate = 0.01
    epochs = 50
    batch_size = 32

    model = NeuralNetwork(input_size, hidden_sizes, output_size, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
    model.train(X_train, y_train_onehot)


    X_test=pd.read_csv("test_data.csv")
    
    X_test["SUBLOCALITY"] = X_test["SUBLOCALITY"].apply(lambda row: sub_locality[row])
    X_test=X_test.drop(["ADDRESS","MAIN_ADDRESS","FORMATTED_ADDRESS", "LONG_NAME", "STREET_NAME", "ADMINISTRATIVE_AREA_LEVEL_2", "BROKERTITLE", "STATE", "LOCALITY"],axis=1)

    X_test_encoded_df = pd.DataFrame()

    for value in type_unique_train:
        X_test_encoded_df[f"TYPE_{value}"] = (X_test["TYPE"] == value).astype(int)

    for value in sublocality_unique_train:
        X_test_encoded_df[f"SUBLOCALITY_{value}"] = (X_test["SUBLOCALITY"] == value).astype(int)
    
    X_test = pd.concat([X_test, X_test_encoded_df], axis=1)
    X_test.drop(columns=categorical_columns, inplace=True)
    #print(X_test.shape)

    #Robustscaling
    for col in ['PRICE', 'BATH', 'PROPERTYSQFT', 'LATITUDE','LONGITUDE']:
        median=np.median(X_test[col])
        fq=np.quantile(X_test[col],0.25)
        tq=np.quantile(X_test[col],0.75)
        iqr=tq-fq
        if iqr==0:
            iqr=1
        X_test[col]=(X_test[col]-median)/iqr

    y_pred = model.predict(X_test)
    y_pred=pd.DataFrame(y_pred,columns=["BEDS"])
    y_pred.to_csv("output.csv",index=False)