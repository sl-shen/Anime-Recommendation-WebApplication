import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


df_user_score=pd.read_csv('./users-score-2023.csv')
df_user_score.head()
df_anime=pd.read_csv('./anime-dataset-2023.csv')
df_anime.head()

# merge the Type col into user_score dataset
merged_dataset = pd.merge(df_user_score, df_anime[['anime_id', 'Type']], on='anime_id', how='left')
merged_dataset.head()

# Grouping the dataset by genre
grouped = merged_dataset.groupby('Type')

# Creating a dictionary of DataFrames for each tpye
type_datasets = {Type: group.copy() for Type, group in grouped}

unique_types = merged_dataset['Type'].unique()
print("unique_type:", unique_types)

# seperate the dataset based on Type
TV_dataset = type_datasets['TV']
Movie_dataset = type_datasets['Movie']
OVA_dataset = type_datasets['OVA']
ONA_dataset = type_datasets['ONA']
Special_dataset = type_datasets['Special']
Music_dataset = type_datasets['Music'] # weird type

combined_TV_dataset = pd.concat([TV_dataset, OVA_dataset, Special_dataset, ONA_dataset]) # include all episodes into tv category
combined_TV_dataset.head()


# deal with TV dataset
# shuffle the data since was ordered by the user id
combined_TV_dataset = shuffle(combined_TV_dataset, random_state=66)
# do the encoding
# Create a MinMaxScaler object
scaler = MinMaxScaler(feature_range=(0, 1))
# Scale the 'score' column between 0 and 1
combined_TV_dataset['scaled_score'] = scaler.fit_transform(combined_TV_dataset[['rating']])
user_encoder = LabelEncoder()
anime_encoder = LabelEncoder()
combined_TV_dataset['anime_id_encoded'] = anime_encoder.fit_transform(combined_TV_dataset['anime_id'])
combined_TV_dataset['user_id_encoded'] = user_encoder.fit_transform(combined_TV_dataset['user_id'])
# drop tv col
del combined_TV_dataset['Type']
combined_TV_dataset.head()

# set X and y for tv
# X is the features used for prediction
# y is the target
X_tv = combined_TV_dataset[['user_id_encoded','anime_id_encoded']].values
y_tv = combined_TV_dataset['scaled_score'].values
print("Shape of X_tv:", X_tv.shape)
print("Shape of y_tv:", y_tv.shape)



# deal with Movie dataset
# shuffle the data since was ordered by the user id
Movie_dataset = shuffle(Movie_dataset, random_state=66)
# do the encoding
# Create a MinMaxScaler object
scaler = MinMaxScaler(feature_range=(0, 1))
# Scale the 'score' column between 0 and 1
Movie_dataset['scaled_score'] = scaler.fit_transform(Movie_dataset[['rating']])
user_encoder = LabelEncoder()
anime_encoder = LabelEncoder()
Movie_dataset['anime_id_encoded'] = anime_encoder.fit_transform(Movie_dataset['anime_id'])
Movie_dataset['user_id_encoded'] = user_encoder.fit_transform(Movie_dataset['user_id'])
# drop Movie col
del Movie_dataset['Type']
Movie_dataset.head()

# set X and y for Movie
# X is the features used for prediction
# y is the target
X_movie = Movie_dataset[['user_id_encoded','anime_id_encoded']].values
y_movie = Movie_dataset['scaled_score'].values
print("Shape of X_movie:", X_movie.shape)
print("Shape of y_movie:", y_movie.shape)


# split training data and validation data for tv dataset
X_tv_train, X_tv_test, y_tv_train, y_tv_test = train_test_split(X_tv, y_tv, test_size=10000/20894155)

X_tv_train = X_tv_train.astype('int32')
X_tv_test = X_tv_test.astype('int32')
print("Number of samples in the training set for tv:", len(y_tv_train))
print("Number of samples in the test set for tv:", len(y_tv_test))

# split training data and validation data for movie dataset
X_movie_train, X_movie_test, y_movie_train, y_movie_test = train_test_split(X_movie, y_movie, test_size=10000/3346546)

X_movie_train = X_movie_train.astype('int32')
X_movie_test = X_movie_test.astype('int32')
print("Number of samples in the training set for movie:", len(y_movie_train))
print("Number of samples in the test set for movie:", len(y_movie_test))


class CollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users, num_animes, embedding_size):
        super(CollaborativeFilteringModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.anime_embedding = nn.Embedding(num_animes, embedding_size)
        self.dense = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.activation = nn.Sigmoid()  # or nn.ReLU()

    def forward(self, user_input, anime_input):
        user_embedded = self.user_embedding(user_input)
        anime_embedded = self.anime_embedding(anime_input)
        dot_product = torch.mul(user_embedded, anime_embedded).sum(dim=-1)
        flattened = self.flatten(dot_product.unsqueeze(-1))
        activated = self.activation(flattened)
        dense_output = self.dense(activated)
        output = self.sigmoid(dense_output)
        return output
    

num_users_tv = combined_TV_dataset['user_id_encoded'].max() + 1
num_animes_tv = combined_TV_dataset['anime_id_encoded'].max() + 1
num_users_movie = Movie_dataset['user_id_encoded'].max() + 1
num_animes_movie = Movie_dataset['anime_id_encoded'].max() + 1
embedding_size = 128

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Create an instance of the model for tv
model_tv = CollaborativeFilteringModel(num_users_tv, num_animes_tv, embedding_size).to(device)

# Create an instance of the model for movie
model_movie = CollaborativeFilteringModel(num_users_movie, num_animes_movie, embedding_size).to(device)

print(num_users_tv)
print(num_animes_tv)
print(num_users_movie)
print(num_animes_movie)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer_tv = optim.Adam(model_tv.parameters())
optimizer_movie = optim.Adam(model_movie.parameters())



# Training loop for tv model
num_epochs = 20
batch_size = 10000

for epoch in range(num_epochs):
    for i in range(0, len(X_tv_train), batch_size):
        batch_users = torch.LongTensor(X_tv_train[i:i+batch_size, 0]).to(device)
        batch_animes = torch.LongTensor(X_tv_train[i:i+batch_size, 1]).to(device)
        batch_ratings = torch.FloatTensor(y_tv_train[i:i+batch_size]).to(device)

        # Forward pass
        outputs = model_tv(batch_users, batch_animes)
        loss = criterion(outputs.squeeze(), batch_ratings)

        # Backward pass and optimization
        optimizer_tv.zero_grad()
        loss.backward()
        optimizer_tv.step()

    # Print the loss for every epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model on the test set
with torch.no_grad():
    test_users = torch.LongTensor(X_tv_test[:, 0]).to(device)
    test_animes = torch.LongTensor(X_tv_test[:, 1]).to(device)
    test_ratings = torch.FloatTensor(y_tv_test).to(device)

    test_outputs = model_tv(test_users, test_animes)
    test_loss = criterion(test_outputs.squeeze(), test_ratings)

print(f"Test Loss: {test_loss.item():.4f}")

torch.save(model_tv.state_dict(), 'tv_model_weights.pth')


# Training loop for movie model
num_epochs = 20
batch_size = 10000

for epoch in range(num_epochs):
    for i in range(0, len(X_movie_train), batch_size):
        batch_users = torch.LongTensor(X_movie_train[i:i+batch_size, 0]).to(device)
        batch_animes = torch.LongTensor(X_movie_train[i:i+batch_size, 1]).to(device)
        batch_ratings = torch.FloatTensor(y_movie_train[i:i+batch_size]).to(device)

        # Forward pass
        outputs = model_movie(batch_users, batch_animes)
        loss = criterion(outputs.squeeze(), batch_ratings)

        # Backward pass and optimization
        optimizer_movie.zero_grad()
        loss.backward()
        optimizer_movie.step()

    # Print the loss for every epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model on the test set
with torch.no_grad():
    test_users = torch.LongTensor(X_movie_test[:, 0]).to(device)
    test_animes = torch.LongTensor(X_movie_test[:, 1]).to(device)
    test_ratings = torch.FloatTensor(y_movie_test).to(device)

    test_outputs = model_movie(test_users, test_animes)
    test_loss = criterion(test_outputs.squeeze(), test_ratings)

print(f"Test Loss: {test_loss.item():.4f}")

torch.save(model_movie.state_dict(), 'movie_model_weights.pth')



# use model to predict
def extract_weights(layer):
    # Get the weights from the PyTorch layer
    weights = layer.weight.data
    
    # Convert to numpy for normalization
    weights = weights = weights.cpu().numpy()
    
    # Normalize the weights
    weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    
    return weights

anime_weights_tv = extract_weights(model_tv.anime_embedding)
user_weights_tv = extract_weights(model_tv.user_embedding)
anime_weights_movie = extract_weights(model_movie.anime_embedding)
user_weights_movie = extract_weights(model_movie.user_embedding)

# Grouping the dataset by genre
grouped_anime = df_anime.groupby('Type')

# Creating a dictionary of DataFrames for each tpye
type_anime = {Type: group.copy() for Type, group in grouped_anime}


# seperate the dataset based on Type
TV_anime = type_anime['TV']
Movie_anime = type_anime['Movie']
OVA_anime = type_anime['OVA']
ONA_anime = type_anime['ONA']
Special_anime = type_anime['Special']
Music_anime = type_anime['Music'] # weird type

combined_TV_anime = pd.concat([TV_anime, OVA_anime, Special_anime, ONA_anime]) # include all episodes into tv category


# since the same encoder was used to fit movie again, need to refit to tv
combined_TV_dataset['anime_id_encoded'] = anime_encoder.fit_transform(combined_TV_dataset['anime_id'])
combined_TV_dataset['user_id_encoded'] = user_encoder.fit_transform(combined_TV_dataset['user_id'])

def find_similar_animes_tv(name, n=10, return_dist=False, neg=False):
    try:
        anime_row = combined_TV_anime[combined_TV_anime['Name'].str.lower() == name.lower()].iloc[0]
        #print(anime_row)
        index = anime_row['anime_id']
        #print(index)
        encoded_index = anime_encoder.transform([index])[0]
        weights = anime_weights_tv
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        n = n + 1            
        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]
        print('Animes closest to {}'.format(name))
        if return_dist:
            return dists, closest
        
        SimilarityArr = []
        
        for close in closest:
            decoded_id = anime_encoder.inverse_transform([close])[0]
            anime_frame = combined_TV_anime[combined_TV_anime['anime_id'] == decoded_id]
            
            anime_name = anime_frame['Name'].values[0]
            english_name = anime_frame['English name'].values[0]
            name = english_name if english_name != "UNKNOWN" else anime_name
            genre = anime_frame['Genres'].values[0]
            Synopsis = anime_frame['Synopsis'].values[0]
            similarity = dists[close]
            similarity = "{:.2f}%".format(similarity * 100)
            SimilarityArr.append({"Name": name, "Similarity": similarity, "Genres": genre, "Synopsis":Synopsis})
        Frame = pd.DataFrame(SimilarityArr).sort_values(by="Similarity", ascending=False)
        return Frame[Frame.Name != name]
    except Exception as e:
        print('{} not found in Anime list. Error: {}'.format(name, str(e)))

pd.set_option('display.max_colwidth', None)


