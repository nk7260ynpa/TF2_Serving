TFSER_PATH=$PWD
tensorflow_model_server --rest_api_port=8510 --model_name=random_model --model_base_path=$TFSER_PATH/weights --rest_api_timeout_in_ms=999999999 
