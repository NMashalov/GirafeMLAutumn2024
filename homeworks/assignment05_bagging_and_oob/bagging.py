import numpy as np


class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob

    def _generate_splits(self, data: np.ndarray):
        """
        Generate indices for every bag and store in self.indices_list list
        """
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            # Your Code Here
            self.indices_list.append(
                np.random.choice(range(data_length), size=data_length, replace=True)
            )

    def fit(self, model_constructor, data, target):
        """
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.

        example:

        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        """
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert (
            len(set(list(map(len, self.indices_list)))) == 1
        ), "All bags should be of the same length!"
        assert list(map(len, self.indices_list))[0] == len(
            data
        ), "All bags should contain `len(data)` number of elements!"
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            data_bag, target_bag = (
                data[self.indices_list[bag]],
                target[self.indices_list[bag]],
            )  # Your Code Here
            self.models_list.append(
                model.fit(data_bag, target_bag)
            )  # store fitted models here
        if self.oob:
            self.data = data
            self.target = target

    def predict(self, data):
        """
        Get average prediction for every object from passed dataset
        """
        # Your code here
        return np.mean([model.predict(data) for model in self.models_list], axis=0)

    def _get_oob_predictions_from_every_model(self):
        """
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        """
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        # Your Code Here
        list_of_predictions_lists = [
            [
                self.models_list[model_index].predict(self.data[j].reshape(1, -1))
                for model_index in range(self.num_bags)
                if j not in self.indices_list[model_index]
            ]
            for j in range((len(self.data)))
        ]
        self.list_of_predictions_lists = np.array(
            list_of_predictions_lists, dtype=object
        )

    def _get_averaged_oob_predictions(self):
        """
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        """
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = [
            sum(prediction_list) / len(prediction_list)
            if len(prediction_list)
            else None
            for prediction_list in self.list_of_predictions_lists
        ]  # Your Code Here

    def OOB_score(self):
        """
        Compute mean square error for all objects, which have at least one prediction
        """
        self._get_averaged_oob_predictions()
        return sum(
            (self.target[i] - prediction) ** 2
            for i, prediction in enumerate(self.oob_predictions)
            if prediction is not None
        ) / sum(
            1 for prediction in self.oob_predictions if prediction is not None
        )  # Your Code Here
