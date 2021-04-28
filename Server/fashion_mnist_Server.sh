TFSER_PATH=$PWD
tensorflow_model_server --rest_api_port=8510 --model_name=fashion_mnist_model --model_base_path=$TFSER_PATH/TF_Model/weights/fashion_mnist --rest_api_timeout_in_ms=999999999 
