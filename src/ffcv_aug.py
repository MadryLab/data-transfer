from dataclasses import replace
from typing import Callable, Optional, Tuple

import numpy as np
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import AllocationQuery, Operation
from ffcv.pipeline.state import State


class SelectLabel(Operation):
    """Select label from multiple labels of specified images.
    Parameters
    ----------
    indices : Sequence[int] / list
        The indices of labels to select.
    """

    def __init__(self, indices):
        super().__init__()

        assert isinstance(indices, list) or isinstance(indices, int), f"required dtype: int/list(int). received {type(indices)}"
        if isinstance(indices, int):
            indices = [indices]
        assert len(indices) > 0, "Number of labels to select must be > 0"
        self.indices = np.sort(indices)

    def generate_code(self) -> Callable:

        to_select = self.indices
        my_range = Compiler.get_iterator()

        def select_label(labels, temp_array, indices):
            new_shape = (labels.shape[0], len(to_select))
            labels_subset = np.zeros(shape=new_shape, dtype=labels.dtype)
            for i in my_range(labels.shape[0]):
                labels_subset[i] = labels[i][to_select]
            return labels_subset

        select_label.is_parallel = True
        select_label.with_indices = True

        return select_label

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        new_shape = (len(self.indices),)
        new_state = replace(previous_state, shape=new_shape)
        mem_allocation = AllocationQuery(new_shape, previous_state.dtype)
        # We do everything in place
        return (new_state, mem_allocation)
