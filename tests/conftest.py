import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from ensemble.config import Settings
from ensemble.data import generate_dataset, train_test_split
from ensemble.evaluation import train_and_evaluate_all
from ensemble.learners import default_learners
from ensemble.platform import EnsemblePlatform
from ensemble.registry import EnsembleRegistry


@pytest.fixture()
def settings():
    return Settings()


@pytest.fixture()
def dataset():
    return generate_dataset(n=600, seed=42)


@pytest.fixture()
def split(dataset):
    return train_test_split(dataset, test_fraction=0.25, seed=42)


@pytest.fixture()
def members():
    return default_learners()


@pytest.fixture()
def fitted_members(split, members):
    return [m.clone().fit(split.X_train, split.y_train) for m in members]


@pytest.fixture(scope="module")
def reports():
    """Module-scoped: building all three ensembles is the slow step, and
    no test mutates the reports (gates only read them)."""
    settings = Settings()
    data = generate_dataset(n=600, seed=42)
    split = train_test_split(data, test_fraction=0.25, seed=42)
    return train_and_evaluate_all(
        default_learners(),
        split.X_train, split.y_train,
        split.X_test, split.y_test,
        split.segment_test,
        settings,
    )


@pytest.fixture()
def platform(settings):
    return EnsemblePlatform(settings)


@pytest.fixture()
def registry(settings):
    return EnsembleRegistry(settings)
