"""
ray_map provides a convenient way to parallelize function execution across an iterable of arguments using Ray.

It is syntactic sugar for calling ray.get([f.remote(x) for x in iterable]), yielding results as they complete.

Example:
    >>> @ray.remote
    ... def f(x):
    ...     return x * x
    ...
    >>> list(ray_map(f, [1, 2, 3]))
    [1, 4, 9]
"""

from typing import Any, Iterable, Sequence, Optional
import ray


def ray_map(
    f: ray.remote_function.RemoteFunction,
    *input_iterators: Iterable[Any],  # one input iterator per argument in the mapped-over function/generator
    kwargs: Optional[dict] = None,  # any extra keyword arguments for the function
    order_outputs: bool = True,  # return outputs in order
):
    """Map a Ray remote function over multiple iterables of arguments, yielding results as they complete.

    Example:
        >>> @ray.remote
        ... def square(x):
        ...     return x * x
        ...
        >>> # Process results in order
        >>> list(ray_map(square, [1, 2, 3]))
        [1, 4, 9]
        ...
        >>> # Multiple arguments
        >>> @ray.remote
        ... def add(x, y):
        ...     return x + y
        ...
        >>> list(ray_map(add, [1, 2, 3], [4, 5, 6]))
        [5, 7, 9]
        ...
        >>> # With keyword arguments
        >>> @ray.remote
        ... def power(x, exp=2):
        ...     return x ** exp
        ...
        >>> list(ray_map(power, [1, 2, 3], kwargs={'exp': 3}))
        [1, 8, 27]
    
    Args:
        f: A Ray remote function to execute
        *input_iterators: One input iterator per argument in the mapped-over function/generator
        kwargs: Any extra keyword arguments for the function
        order_outputs: Whether to return outputs in order. Note this will reduce performance, since you 
          will have to wait for all tasks to complete before getting any results.
        
    Yields:
        Results from the remote function executions as they complete
    """
    if isinstance(f, ray.actor.ActorHandle):
        raise ValueError("ray_map does not support Ray actors.")

    if kwargs is None:
        kwargs = {}

    futures = [f.remote(*args, **kwargs) for args in zip(*input_iterators)]
    
    if order_outputs:  # Process futures in order
        for result in ray.get(futures):  # blocks until all futures are complete
            yield result
    else:  # Process futures as they complete
        while futures:
            ready_futures, futures = ray.wait(futures)
            for done_future in ready_futures:
                yield ray.get(done_future)


def ray_starmap(
    f: ray.remote_function.RemoteFunction,
    input_iterator: Iterable[Sequence[Any] | Any],
    *,
    kwargs: Optional[dict] = None,
    order_outputs: bool = True
):
    """Like ray_map but accepts a single iterator of argument sequences instead of separate iterators.
    
    Examples:
        >>> # Process results in order
        >>> def add(x, y): return x + y
        >>> list(ray_starmap(add, [(1,4), (2,5), (3,6)]))
        [5, 7, 9]
        
        >>> # With keyword arguments
        >>> def power(x, exp=2): return x ** exp
        >>> list(ray_starmap(power, [(1,), (2,), (3,)], kwargs={'exp': 3}))
        [1, 8, 27]
    
    Args:
        f: A Ray remote function to execute
        input_iterator: An iterator yielding sequences of arguments to pass to f
        kwargs: Any extra keyword arguments for the function
        order_outputs: Whether to return outputs in order. Note this will reduce performance, since you 
          will have to wait for all tasks to complete before getting any results.
        
    Yields:
        Results from the remote function executions as they complete
    """
    if isinstance(f, ray.actor.ActorHandle):
        raise ValueError("ray_starmap does not support Ray actors.")

    if kwargs is None:
        kwargs = {}

    # Create futures for all args
    futures = [f.remote(*args, **kwargs) for args in input_iterator]
    
    if order_outputs:  # Process futures in order
        for result in ray.get(futures):  # blocks until all futures are complete
            yield result
    else:  # Process futures as they complete
        while futures:
            ready_futures, futures = ray.wait(futures)
            for done_future in ready_futures:
                yield ray.get(done_future)



