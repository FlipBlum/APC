-- SQLite
CREATE TABLE rated_image (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path String,
    rating STRING,
    date_rated DATETIME DEFAULT CURRENT_TIMESTAMP
);