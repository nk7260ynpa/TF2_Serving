# TF2_Serving
Here is two tensorflow sample, one for client is with tf, the other is not.

## Install
#### Install TF serving 
    apt-get install tensorflow-model-server
## Example 1(Client with tf)
### (1)Train Fashion Model
    python TF_Model/fashion_mnist_Train.py
### (2)Run Server(Don't close this terminal)
    bash Server/fashion_mnist_Server.sh
### (3)Run Client
    python Clients/fashion_mnist_Clients.py
## Example2(Client without tf)
### (1)Train Flowers Model
#### Prepare data    
    cd TF_Model/data
    wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    tar zxvf flower_photos.tgz
    cd ../../
#### Training
    python TF_Model/flowers_Train.py
### (2)Run Server(Don't close this terminal)
    bash Server/flowers_Server.sh
### (3)Run Client
    python Clients/flowers_Clients.py
