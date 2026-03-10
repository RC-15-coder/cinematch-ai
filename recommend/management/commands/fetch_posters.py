import os
import requests
import concurrent.futures
from django.core.management.base import BaseCommand, CommandError
from recommend.models import Movie

# We move the actual fetching logic into a separate worker function
def fetch_poster(movie_id, tmdb_id, api_key):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return movie_id, f"https://image.tmdb.org/t/p/w500{poster_path}"
    except requests.exceptions.RequestException:
        pass
    
    return movie_id, None

class Command(BaseCommand):
    help = 'Fetches high-resolution movie posters from TMDB API using Multi-threading'

    def handle(self, *args, **kwargs):
        API_KEY = os.environ.get("TMDB_API_KEY")
        if not API_KEY:
            raise CommandError("ERROR: TMDB_API_KEY environment variable is not set.")

        # This will automatically skip the 900 movies you already processed!
        movies = Movie.objects.filter(tmdb_id__isnull=False, poster_url__isnull=True)
        total_movies = movies.count()

        if total_movies == 0:
            self.stdout.write(self.style.SUCCESS("All movies already have posters!"))
            return

        self.stdout.write(f"Found {total_movies} movies needing posters. Initiating Multi-threaded download...")

        # Store movie objects in a dictionary for quick lookup by ID
        movie_dict = {m.id: m for m in movies}
        updates = []
        count = 0

        # Open 20 simultaneous threads (Our "20 Checkout Lanes")
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            # Submit all tasks to the thread pool
            futures = {
                executor.submit(fetch_poster, m.id, m.tmdb_id, API_KEY): m.id 
                for m in movies
            }

            # As soon as ANY thread finishes a poster, process it
            for future in concurrent.futures.as_completed(futures):
                movie_id, poster_url = future.result()
                count += 1

                if poster_url:
                    movie = movie_dict[movie_id]
                    movie.poster_url = poster_url
                    updates.append(movie)

                # Batch save every 100 movies to keep the database fast
                if count % 100 == 0:
                    self.stdout.write(f"Processed {count} / {total_movies} movies...")
                    if updates:
                        Movie.objects.bulk_update(updates, ['poster_url'])
                        updates = []

        # Save the final batch
        if updates:
            Movie.objects.bulk_update(updates, ['poster_url'])

        self.stdout.write(self.style.SUCCESS("SUCCESS! All posters have been linked blazingly fast."))