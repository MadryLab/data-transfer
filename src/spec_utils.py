from copy import deepcopy
import numpy as np

COMPLETED = '_completed'
NUM_MODELS = 'num_models'
SCHEMA = 'schema'
DTYPE = 'dtype'
SHAPE = 'shape'

transfer_datasets = [
        "aircraft",
        "birdsnap",
        # "chestxray14",
        "caltech101",
        "caltech256",
        "cifar10",
        "cifar10_0.1",
        "cifar10_0.25",
        # "cifar10_0.5",
        "cifar100",
        "flowers",
        "food",
        "pets",
        "stanford_cars",
        "sun397"
]

SCHEMA_ENTRIES = {
    'softmax_logits': 'float16',
    'raw_logits': 'float16',
    'softmax_margins': 'float16',
    'raw_margins': 'float16',
    'is_corrects': 'bool_',
    'predictions': 'int64',
    'acc': 'float16'
}

def get_schema(test_size):
    assert isinstance(test_size, int)
    spec = {
        "schema": {},
        "num_models": 71753 # num imagenet models
    }
    for name, dtype in SCHEMA_ENTRIES.items():
        spec["schema"][name] = {
                "dtype": dtype,
                "shape": [test_size]
            }
    spec["schema"]["acc"]['shape'] = []
    return spec

def _add_completed(spec):
    spec = deepcopy(spec)
    spec[SCHEMA][COMPLETED] = {
        DTYPE:"bool_",
        SHAPE:[]
    }

    return spec

def preprocess_spec(spec):
    verify_spec(spec)
    spec = _add_completed(spec)
    return spec

def verify_spec(spec):
    assert NUM_MODELS in spec, f'you need a {NUM_MODELS} attribute'
    assert SCHEMA in spec,  f'you need a {SCHEMA} attribute'
    assert not COMPLETED in spec[SCHEMA], f'no schema dtypes called {COMPLETED} allowed'

    schema = spec[SCHEMA]
    for _, v in schema.items():
        assert DTYPE in v, 'you need a dtype'
        assert SHAPE in v, 'you need a shape'

        this_dtype = v[DTYPE]
        this_shape = v[SHAPE]
        assert type(this_shape) is list, "your shape must be a list"
        assert hasattr(np, this_dtype), f"your dtype {this_dtype} is not a numpy dtype"
