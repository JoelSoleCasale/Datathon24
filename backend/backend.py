import pandas as pd
from fashion_clip.fashion_clip import FashionCLIP
import pickle as pkl
import torch
from torch import nn

import os

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

        # dropout
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

fclip = FashionCLIP('fashion-clip')

def validate_categorical(data, variable):
    # get actual path
    path = os.path.abspath(__file__)
    # get the path to the models ../models/{cat}_valid_invalid.pkl
    path_to_models = os.path.join(os.path.dirname(path), "../models/{cat}_valid_invalid.pkl".format(cat=variable))
    with open(path_to_models, 'rb') as f:
        model = pkl.load(f)
        
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category')
            
    model.enable_categorical = True
        
    validity = model.predict(data)
    
    path_to_encoder = "../models/{cat}_enc_val.pkl".format(cat=variable)
    with open(path_to_encoder, 'rb') as f:
        encoder = pkl.load(f)
        
    validity = encoder.inverse_transform(validity)
    
    return validity[0]

def predict_categorical(data, variable):
    attri_to_size = {
        "cane_height_type": 6,
        "closure_placement": 6,
        "heel_shape_type": 11,
        "knit_structure": 5,
        "length_type": 12,
        "neck_lapel_type": 32,
        "silhouette_type": 30,
        "sleeve_length_type": 6,
        "toecap_type": 4,
        "waist_type": 4,
        "woven_structure": 4
    }
    
    model = Net(512, attri_to_size[variable])
    # Determine the device: Use GPU if available, else fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("../models/{cat}_classifier.pt".format(cat=variable), device))
    model.eval()
    
    data = torch.tensor(data.values, dtype=torch.float32)
    outputs = model(data)
    _, predicted = torch.max(outputs, 1)
    
    path_to_encoder = "../models/{cat}_enc_cat.pkl".format(cat=variable)
    with open(path_to_encoder, 'rb') as f:
        encoder = pkl.load(f)
        
    classification = encoder.inverse_transform(predicted)
    
    return classification[0]

def get_embeddings(image):
    return fclip.encode_images([image], batch_size=1)


def predict(data, metadata):
    categorical_variables = ['cane_height_type', 
                             'closure_placement',
                             'heel_shape_type',
                             'knit_structure',
                             'length_type',
                             'neck_lapel_type',
                             'silhouette_type',
                             'sleeve_length_type',
                             'toecap_type',
                             'waist_type',
                             'woven_structure'
    ]
    
    categories = {}
    useful_metadata_cols = ["des_product_family", "des_line", "des_fabric"]
    for var in categorical_variables:
        validity = validate_categorical(pd.concat([data, metadata[useful_metadata_cols]], axis=1), var)
        
        if validity == "VALID":
            categories[var] = predict_categorical(data, var)
        else:
            categories[var] = validity
            
    return pd.DataFrame([categories])

def predict_attributes(image, metadata):
    embeddings = get_embeddings(image)
    
    data = pd.DataFrame(embeddings)
 
    return predict(data, metadata)
    
    
    
if __name__ == "__main__":
    image = "../../datathon-fme-mango/archive/images/images/81_1035318_77004377-09_.jpg"
    # load the image
    import PIL
    from PIL import Image
    image = Image.open(image)
    metadata = pd.DataFrame({
        "des_product_family": ["Sweaters and Cardigans"],
        "des_line": ["KIDS"],
        "des_fabric": ["TRICOT"]
    })
    print(predict_attributes(image, metadata))