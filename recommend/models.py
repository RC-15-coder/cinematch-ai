from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from django.contrib.auth.models import User


# Create your models here.

class Movie(models.Model):
    # We let Django use the dataset's MovieID as the primary key
    id = models.IntegerField(primary_key=True) 
    title = models.CharField(max_length=255)
    genres = models.CharField(max_length=255)
    
    # NEW FIELDS FOR THE TMDB API
    tmdb_id = models.CharField(max_length=50, null=True, blank=True)
    poster_url = models.URLField(max_length=500, null=True, blank=True)
    synopsis = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.title

class Myrating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    rating = models.IntegerField(default=0, validators=[MaxValueValidator(5), MinValueValidator(0)])

class MyList(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    watch = models.BooleanField(default=False)
