"""
Stream Minds — AI Movie Recommendation App
==========================================
Backend: Flask REST API
Author: Keratilwe
Stack: Flask · PostgreSQL · pandas · scikit-learn · SQLAlchemy
"""

from flask import Flask, request, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from functools import wraps
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", os.getenv("FRONTEND_URL", "*")])

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL", "postgresql://postgres:password@localhost:5432/stream_minds"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "stream-minds-secret-change-me")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=7)

db = SQLAlchemy(app)
jwt = JWTManager(app)


# ─────────────────────────────────────────────
# SUBSCRIPTION PLANS (ZAR)
# ─────────────────────────────────────────────
PLANS = {
    "basic":   {"name": "Basic",   "price": 99,  "daily_limit": 20},
    "premium": {"name": "Premium", "price": 149, "daily_limit": None},
    "pro":     {"name": "Pro",     "price": 199, "daily_limit": None},
}


# ─────────────────────────────────────────────
# DATABASE MODELS
# ─────────────────────────────────────────────
class User(db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(120), nullable=False)
    email         = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    plan          = db.Column(db.String(20), default="basic")
    subscribed    = db.Column(db.Boolean, default=False)
    trial_end     = db.Column(db.DateTime, default=lambda: datetime.utcnow() + timedelta(days=14))
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    ratings       = db.relationship("Rating", backref="user", lazy=True)

    def set_password(self, pw):  self.password_hash = generate_password_hash(pw)
    def check_password(self, pw): return check_password_hash(self.password_hash, pw)

    def is_active(self):
        return self.subscribed or (self.trial_end and self.trial_end > datetime.utcnow())

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "email": self.email,
            "plan": self.plan, "subscribed": self.subscribed,
            "trial_active": self.trial_end > datetime.utcnow() if self.trial_end else False,
            "trial_end": self.trial_end.isoformat() if self.trial_end else None,
        }


class Movie(db.Model):
    __tablename__ = "movies"
    id          = db.Column(db.Integer, primary_key=True)
    title       = db.Column(db.String(200), nullable=False)
    year        = db.Column(db.Integer)
    genres      = db.Column(db.String(200))   # comma-separated
    director    = db.Column(db.String(120))
    cast        = db.Column(db.Text)
    description = db.Column(db.Text)
    imdb_rating = db.Column(db.Float)
    poster_url  = db.Column(db.String(300))
    ratings     = db.relationship("Rating", backref="movie", lazy=True)

    def genre_list(self): return self.genres.split(",") if self.genres else []

    def to_dict(self):
        return {
            "id": self.id, "title": self.title, "year": self.year,
            "genres": self.genre_list(), "director": self.director,
            "imdb_rating": self.imdb_rating, "poster_url": self.poster_url,
        }


class Rating(db.Model):
    __tablename__ = "ratings"
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    movie_id   = db.Column(db.Integer, db.ForeignKey("movies.id"), nullable=False)
    score      = db.Column(db.Float, nullable=False)   # 1–10
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint("user_id", "movie_id"),)


class Subscription(db.Model):
    __tablename__ = "subscriptions"
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    plan        = db.Column(db.String(20), nullable=False)
    amount_zar  = db.Column(db.Integer, nullable=False)
    payment_ref = db.Column(db.String(100))
    status      = db.Column(db.String(20), default="pending")   # pending | active | cancelled
    start_date  = db.Column(db.DateTime, default=datetime.utcnow)
    end_date    = db.Column(db.DateTime)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────
# DECORATORS
# ─────────────────────────────────────────────
def subscription_required(f):
    """Block endpoints if user has no active plan."""
    @wraps(f)
    @jwt_required()
    def wrapper(*args, **kwargs):
        uid  = get_jwt_identity()
        user = User.query.get(uid)
        if not user or not user.is_active():
            return jsonify({"error": "Active subscription required.", "code": "NO_SUBSCRIPTION"}), 403
        return f(*args, **kwargs)
    return wrapper


# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
class HybridRecommender:
    """
    Hybrid engine:
    - Collaborative filtering via user-movie rating matrix + cosine similarity
    - Content-based filtering via TF-IDF on genre + director + cast + description
    """

    def __init__(self):
        self.cf_weight  = 0.6
        self.cb_weight  = 0.4

    # ── COLLABORATIVE ──
    def _build_rating_matrix(self):
        rows = db.session.query(Rating.user_id, Rating.movie_id, Rating.score).all()
        if not rows:
            return None, None
        df = pd.DataFrame(rows, columns=["user_id", "movie_id", "score"])
        matrix = df.pivot_table(index="user_id", columns="movie_id", values="score").fillna(0)
        return matrix, cosine_similarity(matrix)

    def _cf_scores(self, user_id, matrix, sim_matrix):
        if user_id not in matrix.index:
            return {}
        idx       = list(matrix.index).index(user_id)
        sim_row   = sim_matrix[idx]
        rated_ids = set(matrix.loc[user_id][matrix.loc[user_id] > 0].index)
        scores    = {}
        for mid in matrix.columns:
            if mid in rated_ids:
                continue
            weighted = sum(
                sim_row[i] * matrix.iloc[i][mid]
                for i in range(len(matrix.index))
                if matrix.iloc[i][mid] > 0
            )
            denom = sum(abs(sim_row[i]) for i in range(len(matrix.index)) if matrix.iloc[i][mid] > 0)
            scores[mid] = weighted / denom if denom else 0
        return scores

    # ── CONTENT-BASED ──
    def _cb_scores(self, user_id):
        liked_ids = [
            r.movie_id for r in
            Rating.query.filter_by(user_id=user_id).filter(Rating.score >= 7).all()
        ]
        if not liked_ids:
            return {}

        all_movies = Movie.query.all()
        if not all_movies:
            return {}

        corpus = [
            f"{m.genres or ''} {m.director or ''} {m.cast or ''} {m.description or ''}"
            for m in all_movies
        ]
        idx_map = {m.id: i for i, m in enumerate(all_movies)}

        tfidf   = TfidfVectorizer(stop_words="english")
        tf_mat  = tfidf.fit_transform(corpus)
        sim_mat = cosine_similarity(tf_mat)

        liked_indices = [idx_map[lid] for lid in liked_ids if lid in idx_map]
        profile_vec   = np.mean(sim_mat[liked_indices], axis=0)

        rated_ids = {r.movie_id for r in Rating.query.filter_by(user_id=user_id).all()}
        return {
            m.id: float(profile_vec[i])
            for i, m in enumerate(all_movies)
            if m.id not in rated_ids
        }

    # ── HYBRID MERGE ──
    def recommend(self, user_id, top_n=10):
        matrix, sim_matrix = self._build_rating_matrix()
        cf  = self._cf_scores(user_id, matrix, sim_matrix) if matrix is not None else {}
        cb  = self._cb_scores(user_id)

        all_ids = set(cf) | set(cb)
        hybrid  = {
            mid: self.cf_weight * cf.get(mid, 0) + self.cb_weight * cb.get(mid, 0)
            for mid in all_ids
        }
        ranked = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:top_n]

        results = []
        for mid, score in ranked:
            movie = Movie.query.get(mid)
            if not movie:
                continue
            match_pct = min(99, int(50 + score * 50))
            reason    = self._explain(user_id, mid)
            results.append({**movie.to_dict(), "match_pct": match_pct, "reason": reason})
        return results

    def _explain(self, user_id, movie_id):
        """Generate a plain-English explanation for a recommendation."""
        top_rated = (
            db.session.query(Rating, Movie)
            .join(Movie, Rating.movie_id == Movie.id)
            .filter(Rating.user_id == user_id, Rating.score >= 8)
            .order_by(Rating.score.desc())
            .first()
        )
        if top_rated:
            _, liked_movie = top_rated
            return f"Recommended because you liked {liked_movie.title}"
        return "Recommended based on your taste profile"


recommender = HybridRecommender()


# ─────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────
@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data or not all(k in data for k in ("name", "email", "password")):
        return jsonify({"error": "name, email, and password are required"}), 400

    if User.query.filter_by(email=data["email"].lower()).first():
        return jsonify({"error": "Email already registered"}), 409

    user = User(name=data["name"], email=data["email"].lower(), plan="basic")
    user.set_password(data["password"])
    db.session.add(user)
    db.session.commit()

    token = create_access_token(identity=user.id)
    return jsonify({"token": token, "user": user.to_dict()}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data.get("email", "").lower()).first()
    if not user or not user.check_password(data.get("password", "")):
        return jsonify({"error": "Invalid credentials"}), 401

    token = create_access_token(identity=user.id)
    return jsonify({"token": token, "user": user.to_dict()})


@app.route("/api/auth/me", methods=["GET"])
@jwt_required()
def me():
    user = User.query.get(get_jwt_identity())
    return jsonify(user.to_dict())


# ─────────────────────────────────────────────
# MOVIES ROUTES
# ─────────────────────────────────────────────
@app.route("/api/movies", methods=["GET"])
@subscription_required
def get_movies():
    page    = request.args.get("page", 1, type=int)
    per_pg  = request.args.get("per_page", 20, type=int)
    genre   = request.args.get("genre", "")
    query   = Movie.query
    if genre:
        query = query.filter(Movie.genres.ilike(f"%{genre}%"))
    paginated = query.paginate(page=page, per_page=per_pg, error_out=False)
    return jsonify({
        "movies": [m.to_dict() for m in paginated.items],
        "total": paginated.total, "pages": paginated.pages, "page": page,
    })


@app.route("/api/movies/<int:movie_id>", methods=["GET"])
@subscription_required
def get_movie(movie_id):
    movie = Movie.query.get_or_404(movie_id)
    return jsonify(movie.to_dict())


@app.route("/api/movies/trending", methods=["GET"])
@subscription_required
def trending():
    """Top movies by average rating from all users in last 30 days."""
    cutoff = datetime.utcnow() - timedelta(days=30)
    rows = (
        db.session.query(Rating.movie_id, db.func.avg(Rating.score).label("avg_score"))
        .filter(Rating.created_at >= cutoff)
        .group_by(Rating.movie_id)
        .order_by(db.desc("avg_score"))
        .limit(10)
        .all()
    )
    result = []
    for mid, avg in rows:
        movie = Movie.query.get(mid)
        if movie:
            result.append({**movie.to_dict(), "avg_score": round(float(avg), 1)})
    return jsonify(result)


# ─────────────────────────────────────────────
# RECOMMENDATIONS ROUTE
# ─────────────────────────────────────────────
@app.route("/api/recommendations", methods=["GET"])
@subscription_required
def recommendations():
    uid    = get_jwt_identity()
    top_n  = request.args.get("top_n", 10, type=int)
    recs   = recommender.recommend(user_id=uid, top_n=top_n)
    return jsonify({"recommendations": recs, "count": len(recs)})


# ─────────────────────────────────────────────
# RATINGS ROUTES
# ─────────────────────────────────────────────
@app.route("/api/ratings", methods=["POST"])
@subscription_required
def rate_movie():
    uid  = get_jwt_identity()
    data = request.get_json()
    if not data or "movie_id" not in data or "score" not in data:
        return jsonify({"error": "movie_id and score required"}), 400

    score = float(data["score"])
    if not 1 <= score <= 10:
        return jsonify({"error": "Score must be between 1 and 10"}), 400

    existing = Rating.query.filter_by(user_id=uid, movie_id=data["movie_id"]).first()
    if existing:
        existing.score = score
    else:
        db.session.add(Rating(user_id=uid, movie_id=data["movie_id"], score=score))
    db.session.commit()
    return jsonify({"message": "Rating saved", "score": score})


@app.route("/api/ratings", methods=["GET"])
@subscription_required
def get_ratings():
    uid = get_jwt_identity()
    ratings = (
        db.session.query(Rating, Movie)
        .join(Movie, Rating.movie_id == Movie.id)
        .filter(Rating.user_id == uid)
        .all()
    )
    return jsonify([
        {**movie.to_dict(), "score": rating.score, "rated_at": rating.created_at.isoformat()}
        for rating, movie in ratings
    ])


# ─────────────────────────────────────────────
# SUBSCRIPTION ROUTES
# ─────────────────────────────────────────────
@app.route("/api/subscription/plans", methods=["GET"])
def get_plans():
    return jsonify(PLANS)


@app.route("/api/subscription/subscribe", methods=["POST"])
@jwt_required()
def subscribe():
    uid  = get_jwt_identity()
    data = request.get_json()
    plan = data.get("plan", "").lower()
    if plan not in PLANS:
        return jsonify({"error": f"Invalid plan. Choose: {', '.join(PLANS)}"}), 400

    user            = User.query.get(uid)
    user.plan       = plan
    user.subscribed = True

    sub = Subscription(
        user_id    = uid,
        plan       = plan,
        amount_zar = PLANS[plan]["price"],
        status     = "active",
        end_date   = datetime.utcnow() + timedelta(days=30),
    )
    db.session.add(sub)
    db.session.commit()
    return jsonify({
        "message": f"Subscribed to {PLANS[plan]['name']} plan",
        "plan": plan, "amount_zar": PLANS[plan]["price"],
    })


@app.route("/api/subscription/cancel", methods=["POST"])
@jwt_required()
def cancel():
    uid             = get_jwt_identity()
    user            = User.query.get(uid)
    user.subscribed = False
    active_sub = Subscription.query.filter_by(user_id=uid, status="active").first()
    if active_sub:
        active_sub.status = "cancelled"
    db.session.commit()
    return jsonify({"message": "Subscription cancelled"})


# ─────────────────────────────────────────────
# ANALYTICS ROUTE (Premium / Pro only)
# ─────────────────────────────────────────────
@app.route("/api/analytics/dashboard", methods=["GET"])
@subscription_required
def analytics():
    uid  = get_jwt_identity()
    user = User.query.get(uid)
    if user.plan not in ("premium", "pro"):
        return jsonify({"error": "Analytics requires Premium or Pro plan"}), 403

    total_ratings  = Rating.query.filter_by(user_id=uid).count()
    avg_score      = db.session.query(db.func.avg(Rating.score)).filter_by(user_id=uid).scalar()
    top_genre_rows = (
        db.session.query(Movie.genres, db.func.count(Rating.id).label("cnt"))
        .join(Rating, Movie.id == Rating.movie_id)
        .filter(Rating.user_id == uid)
        .group_by(Movie.genres)
        .order_by(db.desc("cnt"))
        .limit(5).all()
    )
    return jsonify({
        "total_ratings": total_ratings,
        "avg_score": round(float(avg_score), 2) if avg_score else 0,
        "top_genres": [{"genre": g, "count": c} for g, c in top_genre_rows],
        "plan": user.plan,
    })


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "app": "Stream Minds",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    })


# ─────────────────────────────────────────────
# SEED SAMPLE DATA (dev only)
# ─────────────────────────────────────────────
def seed_movies():
    if Movie.query.count() > 0:
        return
    samples = [
        ("The Algorithm",   2024, "Sci-Fi,Thriller",   "Chris Nolan",  9.1),
        ("Dark Horizon",    2023, "Drama,Mystery",      "Denis V.",     8.7),
        ("Neural Storm",    2024, "Action,Sci-Fi",      "J.J. Abrams",  8.4),
        ("Echoes of Time",  2023, "Drama",              "A. Iñárritu",  8.9),
        ("The Last Signal", 2024, "Sci-Fi,Thriller",    "Ridley Scott", 9.3),
        ("Crimson Web",     2023, "Crime,Action",       "D. Fincher",   7.8),
        ("Zero Gravity",    2025, "Sci-Fi",             "A. Cuarón",    9.4),
        ("The Override",    2025, "Thriller,Drama",     "Park C.W.",    8.8),
        ("Blood Circuit",   2024, "Action,Crime",       "G. del Toro",  8.2),
        ("Frozen Signal",   2025, "Sci-Fi,Mystery",     "D. Villeneuve",9.1),
    ]
    for title, year, genres, director, rating in samples:
        db.session.add(Movie(
            title=title, year=year, genres=genres,
            director=director, imdb_rating=rating,
            description=f"A gripping {genres.split(',')[0]} film rated {rating}/10.",
        ))
    db.session.commit()
    print("✅ Sample movies seeded.")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        seed_movies()
    app.run(debug=True, host="0.0.0.0", port=5000)
