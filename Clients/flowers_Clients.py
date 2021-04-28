import os
import base64
import json
import requests

with open("Clients/images/4563059851_45a9d21a75.jpg", "rb") as img_file:
    encoded_string = str(base64.urlsafe_b64encode(img_file.read()), "utf-8")

data = json.dumps({"signature_name": "serve", "instances": [{"images": encoded_string}]})

headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8510/v1/models/flowers_model:predict', 
                              data=data, headers=headers)

print(json.loads(json_response.text)["predictions"])
