import numpy as np
import matplotlib.pyplot as plt


class ActiveLearner:
    """
    An active learning class that uses  a query strategy to select
    the most informative samples from a pool and train the model.
    """

    def __init__(self, model, query_strategy, dataset):
        """Initialize with a model, a query strategy, and dataset."""
        self.model = model
        self.query_strategy = query_strategy
        self.dataset = dataset

    def load_dataset(self):
        """Load dataset from the provided dictionary."""
        self.x_pool = self.dataset["x_pool"].copy()
        self.y_pool = self.dataset["y_pool"].copy()
        self.x_test = self.dataset["x_test"].copy()
        self.y_test = self.dataset["y_test"].copy()

    def fit(self, plot=False, **kwargs):
        """Train the model on the current training data."""
        losses, val_losses = self.model.fit(
            self.x_train, self.y_train, self.x_test, self.y_test, **kwargs
        )
        if plot:
            plt.plot(losses, label="Training Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.legend()
            plt.show()

    def query(self, batch_size, **kwargs):
        """
        Use the query strategy to select the most informative
        samples from the pool.
        """
        query_idx, query_samples = self.query_strategy(
            self.model.model,
            self.x_pool,
            batch_size,
            self.x_train,
            self.y_train,
            **kwargs,
        )
        return query_idx, query_samples

    def score(self, **kwargs):
        """Return the score of the model on given test data."""
        return self.model.score(self.x_test, self.y_test, **kwargs)

    def learn(
        self,
        n_loop,
        batch_size,
        init_index=[],
        model_name="",
        verbose=True,
        **kwargs,
    ):
        """
        Perform active learning by iteratively querying the pool, appending the
        new samples to the training set (and removing them from the pool),
        re-fitting the model, and evaluating performance on the validation set.
        """
        # Load dataset
        self.load_dataset()
        # To record the pool index
        # keep a map from “current pool” -> “original pool index”
        pool_map = np.arange(len(self.x_pool))
        picks_history = []

        # If init_index is not provided, use random index
        if not init_index:
            init_index = np.random.choice(
                len(self.x_pool), batch_size, replace=False
            ).tolist()
        if len(init_index) != batch_size:
            raise ValueError(f"init_index must be of length {batch_size}")
        # Record the initial pool index
        picks_history.append(pool_map[init_index])

        # Initialize training data with initial index / initial set
        self.x_train = self.x_pool[init_index]
        self.y_train = self.y_pool[init_index]
        # remove the initial data from the pool
        self.x_pool = np.delete(self.x_pool, init_index, axis=0)
        self.y_pool = np.delete(self.y_pool, init_index, axis=0)
        pool_map = np.delete(pool_map, init_index)

        # Initial fit
        self.fit(**kwargs)
        if model_name:
            self.save(model_name)
        scores = [self.score(**kwargs)]
        for i in range(n_loop):

            # Query the pool
            query_idx, _ = self.query(batch_size=batch_size, **kwargs)
            # Record the picked index
            picks_history.append(pool_map[query_idx])

            # Append to the training set
            new_x = self.x_pool[query_idx]
            new_y = self.y_pool[query_idx]
            self.x_train = np.concatenate([self.x_train, new_x])
            self.y_train = np.concatenate([self.y_train, new_y])
            # Remove from the pool
            self.x_pool = np.delete(self.x_pool, query_idx, axis=0)
            self.y_pool = np.delete(self.y_pool, query_idx, axis=0)
            pool_map = np.delete(pool_map, query_idx)

            # Refit the model on the updated training data
            self.fit(**kwargs)
            if model_name:
                self.save(model_name)

            # Evaluate the model on the validation set
            score_val = self.score(**kwargs)
            if verbose and (i + 1) % 2 == 0:
                print(f"Loop {i+1}: Score {score_val:0.4f}")
            scores.append(score_val)

        picks_history = np.concatenate(picks_history, axis=0).tolist()
        return scores, picks_history

    def save(self, model_name):
        """Save the model with the number of training samples"""
        self.model.save(f"{model_name}_{len(self.x_train)}.pth")
