import os
import urllib.request
import zipfile
import pandas as pd
from django.core.management.base import BaseCommand
from recommend.models import Movie

class Command(BaseCommand):
    help = 'Loads the MovieLens Latest Small dataset'

    def handle(self, *args, **kwargs):
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        zip_path = "ml-latest-small.zip"
        extract_dir = "ml_data"

        self.stdout.write("Downloading MovieLens Latest Small dataset...")
        urllib.request.urlretrieve(url, zip_path)

        self.stdout.write("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        movies_file = os.path.join(extract_dir, "ml-latest-small", "movies.csv")
        links_file = os.path.join(extract_dir, "ml-latest-small", "links.csv")

        self.stdout.write("Reading CSVs with Pandas...")
        movies_df = pd.read_csv(movies_file)
        # We load tmdbId as a string so Pandas doesn't turn it into a weird decimal
        links_df = pd.read_csv(links_file, dtype={'tmdbId': str}) 

        # Merge the two files so every movie gets its TMDB ID
        movies_df = pd.merge(movies_df, links_df, on='movieId', how='left')

        self.stdout.write("Clearing old database...")
        Movie.objects.all().delete()

        self.stdout.write("Loading ~9,000 Modern Movies...")
        movies_to_create = []
        for index, row in movies_df.iterrows():
            # Catch missing TMDB IDs safely
            tmdb_id = row['tmdbId'] if pd.notna(row['tmdbId']) else None
            
            movies_to_create.append(
                Movie(
                    id=row['movieId'],
                    title=row['title'],
                    genres=row['genres'],
                    tmdb_id=tmdb_id
                )
            )
        
        # bulk_create is insanely fast for SQLite
        Movie.objects.bulk_create(movies_to_create, batch_size=1000)
        self.stdout.write(self.style.SUCCESS("SUCCESS! 9,000 modern movies and TMDB links are now in your database."))