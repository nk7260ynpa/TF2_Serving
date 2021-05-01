import os
import base64
import json
import requests
import argparse

def argmax(lst):
	return max(range(len(lst)), key=lst.__getitem__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=str, help="Path of Image")
    opt = parser.parse_args()
    
    IMAGE_PATH = opt.image

    with open(IMAGE_PATH, "rb") as img_file:
        encoded_string = str(base64.urlsafe_b64encode(img_file.read()), "utf-8")

    data = json.dumps({"signature_name": "serving", "instances": [{"images": encoded_string}]})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8510/v1/models/flowers_model:predict', 
                                  data=data, headers=headers)
    class_dict= {0: "daisy", 1: "dandelion", 2:"rose", 3:"sunflower", 4:"tulip"}
    print("Result: ", class_dict[argmax(json.loads(json_response.text)["predictions"][0])])
