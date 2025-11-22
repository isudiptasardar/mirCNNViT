from typing import Literal
class EarlyStop:
    def __init__(self, patience: int, delta: float, mode: Literal["min", "max"]) -> None:
        self.patience = patience
        self.delta = delta
        self.mode = mode

        self.counter = 0
        
        # Initialize the score based on the mode
        self.best_score = float('inf') if mode == 'min' else -float('inf')
        self.is_improved = False

    def __call__(self, score: float):
        self.is_improved = (
            (self.mode == "min" and score < self.best_score - self.delta) or (self.mode == "max" and score > self.best_score + self.delta)
        )
        if self.is_improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience