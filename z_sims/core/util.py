from __future__ import annotations


def get_next_time(start_time: float, dt: float, cur_time: float):
    """
    :param start_time: time when this clock started ticking
    :param dt: timestep of globally ticking clock
    :param cur_time: the current timestamp
    :return: the time of the next tick of a global clock that ticked from start_time at intervals of dt. for example,
    get_next_time(0, 3, 7) returns 9 because the global clock ticks at 3, 6, 9, ..., and 9 is the next tick after 7.
    the returned timestamp excludes cur_time, so get_next_time(0, 2, 4) returns 6 and not 4
    """
    return start_time + ((((cur_time - start_time) // dt) + 1) * dt)
