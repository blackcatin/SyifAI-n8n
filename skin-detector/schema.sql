CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    image_path TEXT,
    processed_image_path TEXT,
    predicted_class VARCHAR(255),
    confidence DOUBLE PRECISION,
    blur_score NUMERIC(10,2),
    brightness_score NUMERIC(10,2),
    quality_score NUMERIC(10,2),
    is_valid BOOLEAN,
    used_for_retraining BOOLEAN,
    model_version VARCHAR(50),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    is_hard_example BOOLEAN,
    confidence_score DOUBLE PRECISION
);