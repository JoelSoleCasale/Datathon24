"""
General class for managing image and text embeddings
"""

import pandas as pd
import numpy as np
import pickle as pkl
from typing import Literal



def add_embeddings_to_df(
        df: pd.DataFrame,
        embedding_kind: Literal['fclip', 'resnet50'] = 'fclip',
        id_col: str = 'cod_modelo_color'
    ) -> pd.DataFrame:
    """
    Add embeddings to a dataframe
    """
    # load the image embeddings
    embeddings_path = f'embeddings/embeddings_{embedding_kind}.pkl'

    with open(embeddings_path, 'rb') as f:
        embeddings = pkl.load(f)

    embeddings_df = pd.DataFrame(embeddings.items(), columns=[id_col, 'embedding'])

    embeddings_df['cod_modelo_color'] = embeddings_df['cod_modelo_color'].apply(lambda x: '_'.join(x.split('_')[:2]))
    embeddings_df.head()

    return df.merge(embeddings_df, on='cod_modelo_color')

# with open('text_embeddings/long_attr_embeddings.pkl', 'wb') as f:
#     pkl.dump(long_attr_embeddings, f)

# with open('text_embeddings/short_attr_embeddings.pkl', 'wb') as f:
#     pkl.dump(short_attr_embeddings, f)

# with open('text_embeddings/long_subattr_embeddings.pkl', 'wb') as f:
#     pkl.dump(long_subattr_embeddings, f)

# with open('text_embeddings/short_subattr_embeddings.pkl', 'wb') as f:
#     pkl.dump(short_subattr_embeddings, f)

def add_attr_sim(
        df: pd.DataFrame,
        embedding_kind: Literal['short', 'long'] = 'short'
    ) -> pd.DataFrame:
    """
    Adds the similarity of the attribute with the image embeddings to the dataframe
    in an attr_sim column
    """
    # load the text embeddings
    embeddings_path = f'text_embeddings/{embedding_kind}_attr_embeddings.pkl'
    with open(embeddings_path, 'rb') as f:
        attr_embeddings = pkl.load(f)

    # for all the column in the dataframe in attr_embeddings, add the
    # cosine similarity with the "embedding" column in the dataframe

    # attr_embeddings is a dict with attr: embedding

    
    for attr, embedding in attr_embeddings.items():
        if attr not in df.columns:
            continue

        emb_norm = embedding/(np.linalg.norm(embedding))
        df[f'{attr}_sim'] = df['embedding'].apply(lambda x: np.dot(x, emb_norm)/(np.linalg.norm(x)))
    
    return df


def add_subattr_sim(
        df: pd.DataFrame,
        attr: str,
        embedding_kind: Literal['short', 'long'] = 'short'
    ) -> pd.DataFrame:
    """
    Adds the similarity of the  all the subattribute with the image embeddings to the dataframe
    in an subattr_sim column
    """
    # load the text embeddings
    embeddings_path = f'text_embeddings/{embedding_kind}_subattr_embeddings.pkl'
    with open(embeddings_path, 'rb') as f:
        subattr_embeddings = pkl.load(f)

    for subattr, embedding in subattr_embeddings[attr].items():
        emb_norm = embedding/(np.linalg.norm(embedding))
        df[f'{subattr}_sim'] = df['embedding'].apply(lambda x: np.dot(x, emb_norm)/(np.linalg.norm(x)))
    
    return df

