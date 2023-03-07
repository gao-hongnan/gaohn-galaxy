import random
from dataclasses import dataclass, fields
from typing import Any, List, Mapping, Optional, Tuple
from urllib.request import urlopen

import numpy as np
import PIL
from matplotlib.colors import ListedColormap
from PIL.Image import Image
from rich import print  # pylint: disable=redefined-builtin

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.base.estimator import BaseEstimator
from src.fundamentals.decision_boundary.decision_boundary import plot_decision_regions


def run_classifier(
    estimator: BaseEstimator,  # this type hint is the same name as scikit-learn
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 1992,
    class_names: Optional[List[str]] = None,
) -> BaseEstimator:
    """Run a generic classifier on a dataset and returns the fitted classifier."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    estimator.fit(X_train, y_train)
    y_preds_train = estimator.predict(X_train)
    y_preds_test = estimator.predict(X_test)

    train_report = classification_report(
        y_train,
        y_preds_train,
        labels=np.unique(y_train),
        target_names=class_names if class_names else np.unique(y_train),
        output_dict=False,
    )

    test_report = classification_report(
        y_test,
        y_preds_test,
        labels=np.unique(y_test),
        target_names=class_names if class_names else np.unique(y_test),
        output_dict=False,
    )

    print(f"Train Classification report: \n{train_report}")
    print()
    print_mislabeled_points(y_train, y_preds_train)
    print()
    print(f"Test Classification report: \n{test_report}")
    print_mislabeled_points(y_test, y_preds_test)

    return estimator


def plot_classifier_decision_boundary(
    estimator: BaseEstimator, X: np.ndarray, y: np.ndarray
) -> None:
    """Plot the decision boundary of a classifier."""
    assert X.shape[1] == 2, "Can only plot decision boundary for 2 features."

    estimator.fit(X, y)
    y_preds = estimator.predict(X)
    print(f"Train Classification report: \n{classification_report(y, y_preds)}")
    print_mislabeled_points(y, y_preds)

    # setup marker generator and color map
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])
    plot_decision_regions(
        X,
        y,
        classifier=estimator,
        markers=markers,
        colors=colors,
        contourf=True,
        alpha=0.3,
        cmap=cmap,
    )


def print_mislabeled_points(y_trues: np.ndarray, y_preds: np.ndarray) -> None:
    """Print the mislabeled points."""
    mislabeled_points = (y_trues != y_preds).sum()
    print(f"Mislabeled points: {mislabeled_points}/{y_trues.shape[0]}")


def url_to_numpy_and_pil(url: str) -> Tuple[np.ndarray, Image]:
    """Convert a url to a numpy array and a PIL image."""
    image_pil = PIL.Image.open(urlopen(url)).convert("RGB")
    image_numpy = np.array(image_pil)
    return image_numpy, image_pil


def seed_all(seed: int = 1992) -> None:
    """Seed all random number generators."""
    print(f"Using Seed Number {seed}")

    # os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var
    np.random.seed(seed)  # numpy pseudo-random generator
    random.seed(seed)  # built-in pseudo-random generator


class ChainableDict(dict):
    """Container object exposing keys as attributes.

    State objects extend dictionaries by enabling values to be accessed by key,
    `state["value_key"]`, or by an attribute, `state.value_key`.

    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        """Setattr method.

        Args:
            key: The input key.
            value: The corresponding value to the key.
        """
        self[key] = value

    def __dir__(self):
        """Method to return all the keys."""
        return self.keys()

    def __getattr__(self, key):
        """Method to access value associated with the key.

        Args:
            key: The input key.
        """
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError from exc


def dataclass_from_dict(cls: dataclass, src: Mapping[str, Any]) -> dataclass:
    """Create a new instance of the class from a dictionary.

    Reference: https://stackoverflow.com/questions/53376099/python-dataclass-from-a-nested-dict

    Usage:
    ```
    @dataclass
    class BaseHyperParams(ABC):
        @classmethod
        def from_dict(
            cls: Type[BaseHyperParams], src: Mapping[str, Any]
        ) -> BaseHyperParams:
            return dataclass_from_dict(cls, src)
    ```
    """
    field_types_lookup = {field.name: field.type for field in fields(cls)}

    constructor_inputs = {}
    for field_name, value in src.items():
        try:
            # recursive call to dataclass_from_dict
            constructor_inputs[field_name] = dataclass_from_dict(
                field_types_lookup[field_name], value
            )
        except TypeError:
            # type error from fields() call in recursive call
            # indicates that field is not a dataclass, this is how we are
            # breaking the recursion. If not a dataclass - no need for loading
            constructor_inputs[field_name] = value
        except KeyError:
            # similar, field not defined on dataclass, pass as plain field value
            constructor_inputs[field_name] = value
    return cls(**constructor_inputs)


if __name__ == "__main__":
    state1 = ChainableDict(a=1, b=2)
    state2 = ChainableDict(**{"a": 1, "b": 2})
    print(state2.a)
