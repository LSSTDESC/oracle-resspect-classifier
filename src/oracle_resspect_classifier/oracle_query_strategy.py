import numpy as np
from resspect.query_strategies import QueryStrategy


class ExampleQueryStrategy(QueryStrategy):
    """Minimal example of an external query strategy class."""

    def __init__(
        self,
        queryable_ids: np.array,
        test_ids: np.array,
        batch: int = 1,
        query_threshold: float = 1.0,
        screen: bool = False,
        **kwargs,
    ):
        """The parameters shown are the default set that will be passed to all
        query strategies. If there are additional parameters required, pass them
        in via `**kwargs`.

        Parameters
        ----------
        queryable_ids : np.array
            Set of ids for objects available for querying.
        test_ids : np.array
            Set of ids for objects in the test sample.
        batch : int, optional
            Number of objects to be chosen in each batch query, by default 1
        query_threshold : float, optional
            Threshold where a query is considered worth it, by default 1.0 (no limit)
        screen : bool, optional
            If True display on screen the shift in index and
            the difference in estimated probabilities of being Ia
            caused by constraints on the sample available for querying, by default False
        **kwargs: dict
            Any additional parameters required by the query strategy.
        """

        # The call to `super().__init__` will set the instance variables as follows:
        # self.queryable_ids = queryable_ids
        # self.test_ids = test_ids
        # self.batch = batch
        # self.query_threshold = query_threshold
        # self.screen = screen
        super().__init__(queryable_ids, test_ids, batch, query_threshold, screen, **kwargs)

        # If there are additional parameters, they can be set here. e.g.:
        # self.additional_parameters = kwargs["additional_parameters"]

    def sample(self, probability: np.array) -> list:
        """Search for the sample with highest anomaly certainty in predicted class.

        Parameters
        ----------
        probability : np.array
            Classification probability. One value per class per object.

        Returns
        -------
        list
            List of indexes identifying the objects from the test sample
            to be queried in decreasing order of importance.
            If there are less queryable objects than the required batch
            it will return only the available objects -- so the list of
            objects to query can be smaller than 'batch'.
        """

        # Note - the following are guidelines, but are not enforced in code.
        # 1) After all calculations are complete, the list of returned_indexes is
        #    expected to have length <= self.batch.
        # 2) The list of returned_indexes should only include values that are in
        #    self.queryable_ids.
        returned_indexes = []

        return returned_indexes
