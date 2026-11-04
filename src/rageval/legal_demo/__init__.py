"""The legal-demo evaluation set: fictional sources, labelled cases, frozen splits.

Kept apart from the historical benchmark analysis in ``rageval.report``. See
``data/legal-demo/DATA_CARD.md`` for what the data is and is not.
"""

from .model import (
    CATEGORIES,
    OUTCOMES,
    SPLITS,
    Case,
    Dataset,
    DatasetError,
    Ref,
    Source,
    canonical_sha256,
    ineligibility,
)
from .validate import load_dataset

__all__ = [
    "CATEGORIES",
    "OUTCOMES",
    "SPLITS",
    "Case",
    "Dataset",
    "DatasetError",
    "Ref",
    "Source",
    "canonical_sha256",
    "ineligibility",
    "load_dataset",
]
