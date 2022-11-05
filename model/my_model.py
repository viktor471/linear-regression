from typing import Callable
import numpy as np


class NonParametricRegression:
    def __init__(self,
                 distance_func: Callable,
                 kernel_func: Callable,
                 static_window_width: float = None,
                 neighbours_count: int = None):
        """Модель непараметрической регрессии

        Args:
            distance_func (Callable): функция вычисления дистанции
            kernel_func (Callable): функция вычисления ядра
            window_width (float, optional): ширина окна сглаживания. По-умолчанию не задано.
            neighbours_count (int, optional): количество соседей. По-умолчанию не задано.
        """

        # проверка того, что либо была задана ширина окна, либо количество соседей
        assert (static_window_width is None and neighbours_count is not None or
                static_window_width is not None and neighbours_count is None)

        assert isinstance(distance_func, Callable)
        assert isinstance(kernel_func, Callable)

        self._distance_func = distance_func
        self._kernel_func = kernel_func
        self.static_window_width = static_window_width
        self.neighbours_count = neighbours_count

    @property
    def static_window_width(self) -> float:
        return self._window_width

    @static_window_width.setter
    def static_window_width(self, new_width: float) -> float:
        assert isinstance(new_width, float) or new_width is None
        self._window_width = new_width
        return new_width

    @property
    def neighbours_count(self) -> int:
        return self._neighbours_count

    @neighbours_count.setter
    def neighbours_count(self, new_count: int) -> int:
        assert isinstance(new_count, int) or new_count is None

        self._neighbours_count = new_count

        return new_count

    @property
    def distances(self):
        return [self._distance_func(self._current_selection, x_train_value) for x_train_value in self.x_train]

    @property
    def window_width(self) -> float:
        if self._window_width is not None:
            return self._window_width

        elif self._neighbours_count is not None:
            return sorted(self.distances)[self._neighbours_count - 1]

    @property
    def kernels(self) -> np.ndarray:
        return np.array([self._kernel_func(distance / self.window_width) for distance in self.distances])

    def fit(self, x: np.ndarray, y: np.ndarray):
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

        self.x_train = x
        self.y_train = y

        return self

    def predict(self, x: np.ndarray):
        if len(x.shape) > 1:
            y_predicted = [self._predict(x_value) for x_value in x]
        else:
            y_predicted = self._predict(x)

        return np.array(y_predicted)

    def _predict(self, x: np.ndarray):
        self._current_selection = x

        numenator = 0.0
        denominator = 0.0

        for y_train_value, kernel in zip(self.y_train, self.kernels):
            numenator += y_train_value * kernel
            denominator += kernel

        # numenator = np.sum([(y_train_value * kernel) for (y_train_value, kernel) in zip(self.y_train, self.kernels)], axis=0)

        return numenator / denominator
