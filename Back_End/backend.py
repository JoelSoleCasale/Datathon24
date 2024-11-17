import pandas as pd
from fashion_clip.fashion_clip import FashionCLIP
import pickle as pkl
import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np

# Define the Net class
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.fc(x)

fclip = FashionCLIP('fashion-clip')

def validate_categorical(data, variable):
    # Get actual path
    path = os.path.abspath(__file__)
    # Get the path to the models ../models/{cat}_valid_invalid.pkl
    path_to_models = os.path.join(os.path.dirname(path), "../models/{cat}_valid_invalid.pkl".format(cat=variable))
    with open(path_to_models, 'rb') as f:
        model = pkl.load(f)
        
    validity = model.predict(data)
    
    path_to_encoder = "../models/{cat}_enc_val.pkl".format(cat=variable)
    with open(path_to_encoder, 'rb') as f:
        encoder = pkl.load(f)
        
    validity = encoder.inverse_transform(validity)
    
    return validity

def predict_categorical(data, variable):
    attri_to_size = {
        "cane_height_type": 6,
        "closure_placement": 6,
        "heel_shape_type": 11,
        "knit_structure": 5,
        "length_type": 12,
        "neck_lapel_type": 33,
        "silhouette_type": 33,
        "sleeve_length_type": 6,
        "toecap_type": 4,
        "waist_type": 4,
        "woven_structure": 4
    }
    
    model = Net(512, attri_to_size[variable])
    model.load_state_dict(torch.load("../models/{cat}_classifier.pth".format(cat=variable)))
    model.eval()
    
    data = torch.tensor(data.values, dtype=torch.float32)
    outputs = model(data)
    _, predicted = torch.max(outputs, 1)
    
    path_to_encoder = "../models/{cat}_enc_cat.pkl".format(cat=variable)
    with open(path_to_encoder, 'rb') as f:
        encoder = pkl.load(f)
        
    classification = encoder.inverse_transform(predicted)
    
    return classification

def preprocess_image(image):
    """
    Ensures the input image is in the correct format for FashionCLIP.
    Converts the image to a PIL Image if necessary.
    """
    if isinstance(image, str):
        # If the image is a file path
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        # If the image is a NumPy array
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        # Convert torch tensor to PIL image
        image = Image.fromarray(image.numpy())
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    return image

def get_embeddings(image):
    """
    Encodes an image using FashionCLIP and returns its embeddings.
    """
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # FashionCLIP expects a batch; wrap the image in a list
    embeddings = fclip.encode_images([processed_image], batch_size=1)
    return embeddings

def predict(data):
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
    
    for var in categorical_variables:
        validity = validate_categorical(data, var)
        
        if validity == "Valid":
            categories[var] = predict_categorical(data, var)
        else:
            categories[var] = validity

def predict_attributes(image, metadata):
    embeddings = get_embeddings(image)
    data = pd.DataFrame(embeddings).T
    # data = pd.concat([data, metadata], axis=1)
    return predict(data)
