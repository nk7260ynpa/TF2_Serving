# TF2_Serving
Here is two tensorflow sample, one for client is with tf, the other is not.

## Install
Install TF serving

    apt-get install tensorflow-model-server
    
## Example 1
### (1)Train Fashion Model
    
    python TF_Model/fashion_mnist_Train.py
    
### (2)Run Server(Don't close terminal)
    
    bash Server/fashion_mnist_Server.sh
    
### (3)Run Client
    
    python Clients/fashion_mnist_Clients.py
    
## Example2
### 



