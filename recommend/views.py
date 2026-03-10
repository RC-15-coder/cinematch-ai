from .ml_service import get_pytorch_recommendations
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, get_object_or_404, redirect
from django.http import Http404, HttpResponseRedirect, JsonResponse
from django.db.models import Q, Case, When
from django.contrib import messages
from django.core.paginator import Paginator

from .forms import SignUpForm
from .models import Movie, Myrating, MyList
from .services_cf_knn import topn_knn_for_user
from .services_content import has_user_history
from .services import get_recommendations_for_user

# Home / listing with search
def index(request):
    # 1. Grab ALL perfect movies, but DO NOT slice with [:48] yet
    movies_list = Movie.objects.exclude(
        poster_url__isnull=True
    ).exclude(
        poster_url='None'
    ).exclude(
        poster_url=''
    ).order_by('-id')
    
    q = request.GET.get('q')
    if q:
        movies_list = movies_list.filter(Q(title__icontains=q)).distinct()
        
    # 2. Set up the Paginator: Tell it to show 48 movies per page
    paginator = Paginator(movies_list, 48)
    
    # 3. Get the current page number from the URL (e.g., /?page=2)
    page_number = request.GET.get('page')
    
    # 4. Grab only the 48 movies for that specific page
    movies = paginator.get_page(page_number)
        
    return render(request, 'recommend/list.html', {'movies': movies})

# Movie details + rate / add to watch list (login required)
def detail(request, movie_id):
    if not request.user.is_authenticated:
        return redirect("login")
    if not request.user.is_active:
        raise Http404

    movies = get_object_or_404(Movie, id=movie_id)
    movie = movies  # alias

    # current watch flag in user's MyList
    row = MyList.objects.filter(movie_id=movie_id, user=request.user).values('watch').first()
    update = row['watch'] if row else False

    if request.method == "POST":
        # MyList toggle
        if 'watch' in request.POST:
            update = (request.POST['watch'] == 'on')
            obj, _ = MyList.objects.get_or_create(user=request.user, movie=movie, defaults={'watch': update})
            if not _:
                MyList.objects.filter(pk=obj.pk).update(watch=update)
            messages.success(request, "Movie added to your list!" if update else "Movie removed from your list!")
        # Rating submit
        else:
            rate = request.POST['rating']
            obj, created = Myrating.objects.get_or_create(user=request.user, movie=movie, defaults={'rating': rate})
            if not created:
                Myrating.objects.filter(pk=obj.pk).update(rating=rate)
            messages.success(request, "Rating has been submitted!")

        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))

    # user’s existing rating for this movie (if any)
    user_rating = Myrating.objects.filter(user=request.user, movie=movie).values_list('rating', flat=True).first()
    movie_rating = user_rating or 0
    rate_flag = user_rating is not None

    context = {'movies': movies, 'movie_rating': movie_rating, 'rate_flag': rate_flag, 'update': update}
    return render(request, 'recommend/detail.html', context)

# My List (login required)
def watch(request):
    if not request.user.is_authenticated:
        return redirect("login")
    if not request.user.is_active:
        raise Http404

    movies = Movie.objects.filter(mylist__watch=True, mylist__user=request.user)
    q = request.GET.get('q')
    if q:
        movies = Movie.objects.filter(Q(title__icontains=q)).distinct()
    return render(request, 'recommend/watch.html', {'movies': movies})

def recommend(request):
    if not request.user.is_authenticated:
        return redirect("login")
    if not request.user.is_active:
        raise Http404

    # 1. Find the IDs of movies the user actually rated 4 or 5 stars
    liked_movie_ids = list(Myrating.objects.filter(
        user=request.user, 
        rating__gte=4
    ).values_list('movie_id', flat=True))
    
    # FORCE DATA COLLECTION ---
    if len(liked_movie_ids) < 15:
        return render(request, 'recommend/recommend.html', {
            'movies': [],
            'liked_movies': [],
            'cold_start_warning': f"You have only highly rated {len(liked_movie_ids)} movies. Our AI requires at least 15 high ratings to build an accurate mathematical taste profile."
        })

        # 2. Fetch the actual Movie objects
    liked_movies = Movie.objects.filter(id__in=liked_movie_ids)
    
    # WEIGHTED FREQUENCY ---
    # Count EXACTLY how many times the user likes each genre
    liked_genre_counts = {}
    for movie in liked_movies:
        if movie.genres and movie.genres != "None":
            genres = [g.strip() for g in movie.genres.replace('|', ',').split(',')]
            for g in genres:
                liked_genre_counts[g] = liked_genre_counts.get(g, 0) + 1

    # 3. Ask PyTorch for a large pool of recommendations (Top 100)
    predicted_movie_ids = get_pytorch_recommendations(liked_movie_ids, top_k=100)
    
    if predicted_movie_ids:
        preserved = Case(*[When(id=pk, then=pos) for pos, pk in enumerate(predicted_movie_ids)])
        movie_queryset = Movie.objects.filter(id__in=predicted_movie_ids).exclude(
            poster_url__isnull=True).exclude(poster_url='None')
        
        movie_list = list(movie_queryset.order_by(preserved))
        
        # 4. Score PyTorch's choices using the Weighted Genre Counts
        if liked_genre_counts:
            def genre_score(m):
                if not m.genres or m.genres == "None": 
                    return 0
                m_genres = [g.strip() for g in m.genres.replace('|', ',').split(',')]
                # Sum the frequency points for matching genres
                return sum([liked_genre_counts.get(g, 0) for g in m_genres])
            
            # Sort by the weighted score
            movie_list.sort(key=lambda m: genre_score(m), reverse=True)

        # 5. Slice exactly the top 12 best-matching, highest-quality movies
        final_movies = movie_list[:12]
    else:
        final_movies = []
        
    return render(request, 'recommend/recommend.html', {
        'movies': final_movies,
        'liked_movies': liked_movies
    })

# Auth
def signUp(request):
    form = SignUpForm(request.POST or None)
    if form.is_valid():
        user = form.save()
        user = authenticate(username=form.cleaned_data['username'], password=form.cleaned_data['password1'])
        if user and user.is_active:
            login(request, user)
            return redirect("index")
    return render(request, 'recommend/signUp.html', {'form': form})

def Login(request):
    if request.method == "POST":
        user = authenticate(username=request.POST.get('username',''), password=request.POST.get('password',''))
        if user:
            if user.is_active:
                login(request, user); return redirect("index")
            return render(request, 'recommend/login.html', {'error_message': 'Your account is disabled'})
        return render(request, 'recommend/login.html', {'error_message': 'Invalid login'})
    return render(request, 'recommend/login.html')

def Logout(request):
    logout(request)
    return redirect("login")

# JSON API (personalizes when logged in)
def recommendations_api(request):
    k = int(request.GET.get("k", 10))
    if request.user.is_authenticated and has_user_history(request.user):
        ids = topn_knn_for_user(request.user, k=k, min_overlap=3, blend_pop=0.30)
        if not ids:
            items = get_recommendations_for_user(request.user.id, k)
            return JsonResponse({"user_id": request.user.id, "k": k, "items": items})
        preserved = Case(*[When(pk=pk, then=pos) for pos, pk in enumerate(ids)])
        movies = Movie.objects.filter(id__in=ids).order_by(preserved)
        items = [{"id": m.id, "title": m.title} for m in movies]
        return JsonResponse({"user_id": request.user.id, "k": k, "items": items})
    else:
        items = get_recommendations_for_user(None, k)
        return JsonResponse({"user_id": None, "k": k, "items": items})
