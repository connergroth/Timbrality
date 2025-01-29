from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
from data import (
    fetch_user_artists,
    fetch_user_songs,
    fetch_user_albums,
    fetch_user_liked_songs,
    fetch_user_playlists,
    fetch_recently_played,
)
import pandas as pd


class APIRecommender:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = AlternatingLeastSquares(factors=100, iterations=20, regularization=0.1)

    def create_combined_dataset(self, spotify_data, lastfm_data):
        """Combine Spotify and Last.fm data into a single dataset."""
        if spotify_data.empty:
            print("No Spotify data found. Using Last.fm data only.")
            return lastfm_data

        if lastfm_data.empty:
            print("No Last.fm data found. Using Spotify data only.")
            return spotify_data

        # Combine both datasets
        combined_data = pd.concat([spotify_data, lastfm_data], ignore_index=True)

        # Normalize playcount across sources
        combined_data["playcount"] = (combined_data["playcount"] - combined_data["playcount"].min()) / \
                                     (combined_data["playcount"].max() - combined_data["playcount"].min()) + 1e-6

        return combined_data

    def fetch_user_matrix(self, users: list, data_type: str, sp=None):
        print("fetch_user_matrix called with:", users, data_type)
        spotify_data = pd.DataFrame()
        lastfm_data = pd.DataFrame()

        # Fetch Spotify data if available
        if sp:
            try:
                if data_type == "songs":
                    spotify_data = fetch_user_liked_songs(sp)
                elif data_type == "playlists":
                    spotify_data = fetch_user_playlists(sp)
                elif data_type == "recently_played":
                    spotify_data = fetch_recently_played(sp)
                else:
                    raise ValueError("Invalid data_type. Choose 'songs', 'playlists', or 'recently_played'.")
            except Exception as e:
                print(f"Error fetching Spotify data: {e}")

        # Fetch Last.fm data for all users
        for user in users:
            try:
                if data_type == "artists":
                    user_data = fetch_user_artists(user)
                elif data_type == "songs":
                    user_data = fetch_user_songs(user)
                elif data_type == "albums":
                    user_data = fetch_user_albums(user)
                else:
                    raise ValueError("Invalid data_type. Choose 'artists', 'songs', or 'albums'.")
                if not user_data.empty:
                    user_data["user_id"] = user
                    lastfm_data = pd.concat([lastfm_data, user_data], ignore_index=True)
            except Exception as e:
                print(f"Error fetching data for user '{user}': {e}")

        # Combine datasets
        combined_data = self.create_combined_dataset(spotify_data, lastfm_data)

        # Check combined data
        if combined_data.empty:
            print("Error: Combined dataset is empty.")
            return None, None

        print("Combined DataFrame:")
        print(combined_data.head())

        # Create mappings
        user_map = {user: idx for idx, user in enumerate(combined_data["user_id"].unique())}
        item_map = {item: idx for idx, item in enumerate(combined_data["name"].unique())}

        # Map interactions
        combined_data["user_idx"] = combined_data["user_id"].map(user_map)
        combined_data["item_idx"] = combined_data["name"].map(item_map)

        user_ids = combined_data["user_idx"]
        item_ids = combined_data["item_idx"]
        interactions = combined_data["playcount"]

        coo = coo_matrix((interactions, (user_ids, item_ids)), shape=(len(user_map), len(item_map)))
        return coo.tocsr(), combined_data

    def train_model(self, user_matrix):
        """Train the model with fetched data."""
        self.model.fit(user_matrix)

    def recommend(self, user_id: int, user_matrix, user_data_df, n=5):
        recommended_ids, scores = self.model.recommend(
            user_id,
            user_matrix,
            N=n,
            filter_already_liked_items=True  # Exclude already interacted items
        )

        recommendations = pd.DataFrame({
            "id": recommended_ids,
            "score": scores
        })

        # Map back to item names
        recommendations = recommendations.merge(
            user_data_df[["item_idx", "name"]].drop_duplicates(),
            left_on="id",
            right_on="item_idx",
            how="inner"
        )

        return recommendations[["name", "score"]].to_dict(orient="records")
