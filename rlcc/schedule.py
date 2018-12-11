class Schedule():
    r"""Base class for all schedules.
    
    Your schedules should also subclass this class.
    """
    def __init__(self):
        pass

    def reset(self):
        r"""Resets the schedule."""
        raise NotImplementedError

    def update(self):
        r"""Updates the schedule."""
        raise NotImplementedError
        
    def value(self):
        r"""Get the value."""
        raise NotImplementedError
        
class LinearSchedule(Schedule):
    r"""Linearly updating schedule
    
    :param initial: the initial value
    :type initial: float
    :param delta: the additive delta
    :type delta: float
    """
    def __init__(self, initial, delta):
        super(LinearSchedule, self).__init__()
        self.initial = initial
        self.current = initial
        self.delta = delta

    def reset(self):
        r"""Resets the schedule."""
        self.current = self.initial

    def update(self):
        r"""Updates the schedule."""
        self.current += self.delta
        
    def value(self):
        r"""Get the value."""
        return self.current
    
class ExponentialSchedule(Schedule):
    r"""Exponentially updating schedule
    
    :param initial: the initial value
    :type initial: float
    :param delta: the additive delta
    :type delta: float
    """
    def __init__(self, initial, decay):
        super(ExponentialSchedule, self).__init__()
        self.initial = initial
        self.current = initial
        self.decay = decay

    def reset(self):
        r"""Resets the schedule."""
        self.current = self.initial
    
    def update(self):
        r"""Updates the schedule."""
        self.current *= self.decay
        
    def value(self):
        r"""Get the value."""
        return self.current
    
class BoundedSchedule(Schedule):
    r"""Bounded schedule
    
    :param schedule: the delegate scheule
    :type schedule: :class:`rlcc.schedule.Schedule`
    :param min: the additive delta
    :type min: float, optional
    :param max: the additive delta
    :type max: float, optional
    """
    def __init__(self, schedule, min=None, max=None):
        super(BoundedSchedule, self).__init__()
        self.schedule = schedule
        self.min = min
        self.max = max

    def update(self):
        r"""Updates the schedule."""
        self.schedule.update()
        
    def value(self):
        r"""Get the value."""
        v = self.schedule.value()
        if self.min:
            v = max(v, self.min)
        if self.max:
            v = min(v, self.max)
        return v
