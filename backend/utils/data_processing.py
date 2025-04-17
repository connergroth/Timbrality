# utils/data_processing.py
import pandas as pd

def create_combined_dataset(spotify_data, lastfm_data):
    """Combine Spotify and Last.fm data into a single dataset."""
    if spotify_data.empty:
        return lastfm_data
    if lastfm_data.empty:
        return spotify_data

    combined_data = pd.concat([spotify_data, lastfm_data], ignore_index=True)
    combined_data["playcount"] = (combined_data["playcount"] - combined_data["playcount"].min()) / \
                                 (combined_data["playcount"].max() - combined_data["playcount"].min())
    return combined_data
