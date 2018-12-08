

class Observer():
    r"""Base class for all learners.
    """

    def __init__(self):
        pass

    def observe(self, transition):
        r"""Observe a transition.

        Arguments:
            transition (Transition): the observered transition
        """
        raise NotImplementedError

class BufferedObserver(Observer):
    """Implements Deterministic Policy Gradient.
    """

    def __init__(self, buffer):
        super(BufferedObserver, self).__init__()
        self.buffer = buffer

    def observe(self, transition):
        r"""Observe a transition.

        Arguments:
            transition (Transition): the observered transition
        """
        self.buffer.append(transition)

class PreprocessingObserver(Observer):
    """Implements Deterministic Policy Gradient.
    """

    def __init__(self, base_observer, preprocessing_fn):
        super(PreprocessingObserver, self).__init__()
        self.base_observer = base_observer
        self.preprocessing_fn = preprocessing_fn

    def observe(self, transition):
        r"""Observe a transition.

        Arguments:
            transition (Transition): the observered transition
        """
        self.base_observer.observe(self.preprocessing_fn(transition))


class StackedObserver(Observer):
    r"""Observer where each transition decomposed such that 
        each component is observed respectively"""

    def __init__(self, observers):
        super(StackedObserver, self).__init__()
        self.observers = observers

    def observe(self, transition):
        r"""Observe a transition.

        Arguments:
            transition (Transition): the observered transition
        """
        for t, o in zip(zip(*transition), self.observers):
            o.observe(t)
