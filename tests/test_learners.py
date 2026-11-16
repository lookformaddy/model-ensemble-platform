from ensemble.data import generate_dataset, kfold_indices, train_test_split
from ensemble.learners import (
    DecisionTreeLearner,
    GaussianNaiveBayesLearner,
    KNearestNeighborsLearner,
    LogisticRegressionLearner,
    MajorityLearner,
    default_learners,
    make_learner,
)


def test_dataset_shape_and_balance(dataset):
    assert dataset.n == 600
    assert dataset.n_features == 8
    assert len(dataset.X) == 600
    assert all(len(row) == 8 for row in dataset.X)
    assert set(dataset.y) == {0, 1}
    # Roughly balanced by construction.
    assert 0.35 < sum(dataset.y) / dataset.n < 0.65
    assert set(dataset.segment) == {0, 1}


def test_dataset_deterministic():
    a = generate_dataset(n=200, seed=7)
    b = generate_dataset(n=200, seed=7)
    assert a.X == b.X and a.y == b.y
    c = generate_dataset(n=200, seed=8)
    assert c.X != a.X


def test_train_test_split_sizes(split):
    assert len(split.X_train) + len(split.X_test) == 600
    assert len(split.X_test) == 150
    assert len(split.segment_test) == 150


def test_kfold_covers_all_rows():
    folds = kfold_indices(100, 5, seed=3)
    assert len(folds) == 5
    flat = sorted(i for fold in folds for i in fold)
    assert flat == list(range(100))
    assert all(18 <= len(f) <= 22 for f in folds)


def test_each_learner_fits_and_predicts(fitted_members, split):
    for member in fitted_members:
        probs = member.predict_proba(split.X_test)
        assert len(probs) == len(split.X_test)
        assert all(0.0 <= p <= 1.0 for p in probs)
        labels = member.predict(split.X_test)
        assert set(labels) <= {0, 1}
        acc = sum(1 for p, t in zip(labels, split.y_test) if p == t) / len(split.y_test)
        # Every learner must beat chance on this planted-signal data.
        assert acc > 0.55, f"{member.name} accuracy {acc:.3f} is at chance"


def test_logistic_learns_linear_signal():
    # Purely linear regime (regime indicator positive): logistic
    # regression should be near the Bayes rate there.
    data = generate_dataset(n=600, seed=11)
    rows = [(row, label) for row, label, seg in zip(data.X, data.y, data.segment) if seg == 0]
    X = [row for row, _ in rows]
    y = [label for _, label in rows]
    learner = LogisticRegressionLearner().fit(X, y)
    acc = sum(1 for p, t in zip(learner.predict(X), y) if p == t) / len(y)
    assert acc > 0.85


def test_tree_learns_interaction_signal():
    # XOR-style data: a tree should beat a linear model.
    import random

    rng = random.Random(5)
    X, y = [], []
    for _ in range(400):
        a, b = rng.gauss(0, 1), rng.gauss(0, 1)
        X.append([a, b])
        y.append(1 if a * b > 0 else 0)
    tree = DecisionTreeLearner(max_depth=4).fit(X, y)
    logreg = LogisticRegressionLearner().fit(X, y)
    tree_acc = sum(1 for p, t in zip(tree.predict(X), y) if p == t) / len(y)
    log_acc = sum(1 for p, t in zip(logreg.predict(X), y) if p == t) / len(y)
    assert tree_acc > 0.85
    assert tree_acc > log_acc + 0.2


def test_knn_probability_is_vote_share():
    learner = KNearestNeighborsLearner(k=3).fit(
        [[0.0], [0.1], [10.0], [10.1]], [0, 0, 1, 1]
    )
    probs = learner.predict_proba([[0.05], [10.05]])
    assert probs[0] < 0.4
    assert probs[1] > 0.6


def test_naive_bayes_separates_gaussians():
    learner = GaussianNaiveBayesLearner().fit(
        [[-2.0], [-1.9], [2.0], [1.9]], [0, 0, 1, 1]
    )
    probs = learner.predict_proba([[-2.0], [2.0]])
    assert probs[0] < 0.1
    assert probs[1] > 0.9


def test_majority_learner_predicts_majority():
    learner = MajorityLearner().fit([[0.0]] * 10, [1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    assert learner.predict([[0.0], [1.0]]) == [1, 1]
    assert learner.predict_proba([[0.0]])[0] == 0.7


def test_clone_is_fresh_and_independent(members, split):
    original = members[0]
    clone = original.clone()
    assert clone is not original
    assert clone.kind == original.kind
    # Fitting the clone does not touch the original.
    clone.fit(split.X_train, split.y_train)
    assert clone.predict_proba(split.X_test) != [0.5] * len(split.X_test)


def test_default_learners_cover_four_kinds(members):
    assert [m.kind for m in members] == [
        "logistic", "tree", "knn", "naive_bayes",
    ]


def test_make_learner_unknown_kind():
    try:
        make_learner("transformer")
    except ValueError as exc:
        assert "unknown learner kind" in str(exc)
    else:
        raise AssertionError("expected ValueError")
