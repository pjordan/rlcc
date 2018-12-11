r"""Transition observation functions"""

class Observer():
    r"""Base class for all observers.

    Your observer should also subclass this class.
    """
    def __init__(self):
        pass

    def observe(self, transition):
        r"""Observe a transition.

        :param transition: the transition
        :type transition: :class:`rlcc.Transition`
        """
        raise NotImplementedError

class BufferedObserver(Observer):
    r"""Observer that places the transitions in a buffer.

    :param buffer: the buffer
    :type buffer: an appendable object
    """
    def __init__(self, buffer):
        super(BufferedObserver, self).__init__()
        self.buffer = buffer

    def observe(self, transition):
        r"""Observe a transition.

        :param transition: the transition
        :type transition: :class:`rlcc.Transition`
        """
        self.buffer.append(transition)

class StackedObserver(Observer):
    r"""Stacked observers
    
    :param observers: list of observers
    :type observers: list of :class:`rlcc.observe.Observer`
    """
    def __init__(self, observers):
        super(StackedObserver, self).__init__()
        self.observers = observers

    def observe(self, transition):
        r"""Observe a transition.

        :param transition: the transition
        :type transition: :class:`rlcc.Transition`
        """
        for t, o in zip(zip(*transition), self.observers):
            o.observe(t)
