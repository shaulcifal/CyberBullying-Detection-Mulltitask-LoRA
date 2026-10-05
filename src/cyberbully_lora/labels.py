"""Label vocabularies shared between data prep, datasets, and models."""

BULLY_LABELS = ["Non_bully", "Bully"]

SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

TARGET_LABELS = [
    "Not Applicable",
    "Miscellaneous",
    "Attacking Relatives and Friends",
    "Community",
    "Gender",
    "Organization",
    "Profession",
    "Race",
    "Religion",
    "Sexual Oriantation",  # spelling as annotated in the source dataset
]

CLASSIFICATION_LABELS = {
    "bully": BULLY_LABELS,
    "sentiment": SENTIMENT_LABELS,
    "target": TARGET_LABELS,
}
