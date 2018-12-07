class Schedule():
    r"""Base class for all schedules.
    """
    
    def __init__(self):
        pass

    def update(self):
        r"""Updates the schedule."""
        raise NotImplementedError
        
    def value(self):
        r"""Get the value."""
        raise NotImplementedError
        
class LinearSchedule(Schedule):
    r"""Linearly updating schedule"""
    
    def __init__(self, initial, delta):
        super(LinearSchedule, self).__init__()
        self.current = initial
        self.delta = delta

    def update(self):
        r"""Updates the schedule."""
        self.current += self.delta
        
    def value(self):
        r"""Get the value."""
        return self.current
    
class ExponentialSchedule(Schedule):
    r"""Linearly updating schedule"""
    
    def __init__(self, initial, decay):
        super(ExponentialSchedule, self).__init__()
        self.current = initial
        self.decay = decay

    def update(self):
        r"""Updates the schedule."""
        self.current *= self.decay
        
    def value(self):
        r"""Get the value."""
        return self.current
    
class BoundedSchedule(Schedule):
    r"""Linearly updating schedule"""
    
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
