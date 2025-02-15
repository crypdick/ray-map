import pytest
import ray
from ray_map import ray_map, ray_starmap

@pytest.fixture(scope="session", autouse=True)
def ray_init():
    """Initialize Ray for all tests if not already initialized."""
    if not ray.is_initialized():
        ray.init()

# Ray Map Tests
def test_basic_map():
    @ray.remote 
    def square(x):
        return x * x
    
    result = list(ray_map(square, [1, 2, 3]))
    assert result == [1, 4, 9]

def test_iterator_behavior():
    @ray.remote 
    def square(x):
        return x * x
    
    results = []
    for x in ray_map(square, [1, 2, 3]):
        results.append(x)
    assert results == [1, 4, 9]

def test_kwargs():
    @ray.remote
    def power(x, exp=2):
        return x ** exp
    
    results = list(ray_map(power, [1, 2, 3], kwargs={'exp': 3}))
    assert results == [1, 8, 27]

def test_unordered_output():
    @ray.remote 
    def square(x):
        return x * x
    
    results = set()
    for result in ray_map(square, [1, 2, 3], order_outputs=False):
        results.add(result)
    assert results == {1, 4, 9}

def test_multiple_arguments():
    @ray.remote
    def add(x, y):
        return x + y
    
    results = list(ray_map(add, [1, 2, 3], [4, 5, 6]))
    assert results == [5, 7, 9]

# Ray Starmap Tests
def test_basic_starmap():
    @ray.remote
    def add(x, y):
        return x + y
    
    result = list(ray_starmap(add, [(1,4), (2,5), (3,6)]))
    assert result == [5, 7, 9]

def test_starmap_iterator_behavior():
    @ray.remote
    def add(x, y):
        return x + y
    
    results = []
    for x in ray_starmap(add, [(1,4), (2,5), (3,6)]):
        results.append(x)
    assert results == [5, 7, 9]

def test_starmap_kwargs():
    @ray.remote
    def power(x, exp=2):
        return x ** exp
    
    results = list(ray_starmap(power, [(1,), (2,), (3,)], kwargs={'exp': 3}))
    assert results == [1, 8, 27]

def test_starmap_unordered_output():
    @ray.remote
    def add(x, y):
        return x + y
    
    results = set()
    for result in ray_starmap(add, [(1,4), (2,5), (3,6)], order_outputs=False):
        results.add(result)
    assert results == {5, 7, 9}