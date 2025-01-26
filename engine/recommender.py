from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
from data import fetch_user_artists, fetch_user_songs, fetch_user_albums
import pandas as pd

class APIRecommender:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = AlternatingLeastSquares(factors=100, iterations=10, regularization=0.01)



    def fetch_user_matrix(self, users: list, data_type: str):
        all_user_data = []
        for user in users:
            try:
                if data_type == "artists":
                    user_data_df = fetch_user_artists(user)
                elif data_type == "songs":
                    user_data_df = fetch_user_songs(user)
                elif data_type == "albums":
                    user_data_df = fetch_user_albums(user)
                else:
                    raise ValueError("Invalid data_type. Choose 'artists', 'songs', or 'albums'.")

                if not user_data_df.empty:
                    user_data_df["user_id"] = user
                    all_user_data.append(user_data_df)
                else:
                    print(f"No data found for user '{user}'.")
            except Exception as e:
                print(f"Error fetching data for user '{user}': {e}")

        if not all_user_data:
            print("Error: No valid data for any user.")
            return None, None

        combined_data = pd.concat(all_user_data, ignore_index=True)
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
        recommended_ids, scores = self.model.recommend(user_id, user_matrix, N=n)

        recommendations = user_data_df.iloc[recommended_ids].copy()
        recommendations["score"] = scores
        return recommendations[["name", "score"]].to_dict(orient="records")

