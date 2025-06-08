from __future__ import annotations


class RunningReward:
    def __init__(self, keep_factor: float, initial_value: float = 0) -> None:
        """Args:
        keep_factor: How much of the last value to keep when a new one is added.
        initial_value: Initial reward

        """
        if not 0.0 <= keep_factor <= 1.0:
            raise ValueError("keep_factor must be between 0 and 1")

        self._keep_factor = keep_factor
        self._reward = initial_value
        self.last_added = initial_value

    @property
    def value(self) -> float:
        """Get the current running reward."""
        return self._reward

    def update(self, reward: float) -> None:
        """Update the running reward with a new value."""
        self._reward *= self._keep_factor
        self._reward += reward * (1.0 - self._keep_factor)
        self.last_added = reward
