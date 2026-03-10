import os
import requests
import concurrent.futures
from django.core.management.base import BaseCommand, CommandError
from recommend.models import Movie

def fetch_movie_synopsis(movie_id, tmdb_id, api_key):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            overview = data.get('overview')
            if overview:
                return movie_id, overview
    except requests.exceptions.RequestException:
        pass
    return movie_id, None

class Command(BaseCommand):
    help = 'Fetches movie synopses from TMDB API using Multi-threading'

    def handle(self, *args, **kwargs):
        API_KEY = os.environ.get("TMDB_API_KEY")
        if not API_KEY:
            raise CommandError("ERROR: TMDB_API_KEY environment variable is not set.")

        movies = Movie.objects.filter(tmdb_id__isnull=False, synopsis__isnull=True)
        total_movies = movies.count()

        if total_movies == 0:
            self.stdout.write(self.style.SUCCESS("All movies already have a synopsis!"))
            return

        self.stdout.write(f"Found {total_movies} movies needing a synopsis. Downloading...")

        movie_dict = {m.id: m for m in movies}
        updates = []
        count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(fetch_movie_synopsis, m.id, m.tmdb_id, API_KEY): m.id for m in movies}

            for future in concurrent.futures.as_completed(futures):
                movie_id, overview = future.result()
                count += 1

                if overview:
                    movie = movie_dict[movie_id]
                    movie.synopsis = overview
                    updates.append(movie)

                if count % 100 == 0:
                    self.stdout.write(f"Processed {count} / {total_movies} movies...")
                    if updates:
                        Movie.objects.bulk_update(updates, ['synopsis'])
                        updates = []

        if updates:
            Movie.objects.bulk_update(updates, ['synopsis'])

        self.stdout.write(self.style.SUCCESS("SUCCESS! All synopses have been added."))