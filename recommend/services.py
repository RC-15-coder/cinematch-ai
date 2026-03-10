from django.db.models import Avg, Count
from .models import Movie, Myrating

def get_recommendations_for_user(user_id: int | None, k: int = 10):
    """
    Simple popularity baseline:
    - average rating (then count) per movie
    - if user_id is given, exclude movies that user has already rated
    """
    qs = Movie.objects.all()

    if user_id:
        try:
            qs = qs.exclude(myrating__user_id=user_id)  # default reverse name if set
        except Exception:
            qs = qs.exclude(myrating_set__user_id=user_id)  # Django default reverse

    # Annotate popularity stats (handle both reverse names)
    try:
        qs = qs.annotate(
            num=Count("myrating"),
            avg=Avg("myrating__rating"),
        )
    except Exception:
        qs = qs.annotate(
            num=Count("myrating_set"),
            avg=Avg("myrating_set__rating"),
        )

    qs = qs.filter(num__gte=1).order_by("-avg", "-num")[:k]

    data = []
    for m in qs:
        data.append({
            "id": m.id,
            "title": getattr(m, "title", str(m)),
            "score": float(getattr(m, "avg", 0.0) or 0.0),
        })
    return data
