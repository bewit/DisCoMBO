from ..search_spaces import SearchSpace
import numpy as np

DTYPES_TO_TRANSFORM = ['str', 'int', 'bool', 'cat']

class ConfigurationNumericalTransform:

    def __init__(self, search_sapce: SearchSpace) -> None:
        self.search_space = search_sapce
        self._transform_meta_data = {}
        self._variables = list(self.search_space.get_search_space_definition())

        # initialize metadata and store which variables have to be transformed (with corresponding index)
        for key_idx, (key, val) in enumerate(self.search_space.get_search_space_definition().items()):
            if val['dtype'] in DTYPES_TO_TRANSFORM:
                self._transform_meta_data[key_idx] = (key, True)
            else:
                self._transform_meta_data[key_idx] = (key, False)
    
    def transform_configuration(self, configuration: dict, trgt_col_order=None):
        vector_repr = []

        for key_idx, (key, val) in enumerate(self.search_space.get_search_space_definition().items()):
            if val['dtype'] in DTYPES_TO_TRANSFORM:
                choices = val['allowed']
                idx = choices.index(configuration[key])
                vector_repr.append(idx)
            else:
                if val['is_log']:
                    vector_repr.append(np.log(configuration[key]))
                else:
                    vector_repr.append(configuration[key])

        if trgt_col_order is not None:
            vector_repr = reorder_vector_columns(vector_repr, self._variables, trgt_col_order)
    
        return vector_repr

    def inv_transform_configuration(self, config_vec: list, is_col_order=None):
        dict_repr = {}
        search_space_def = self.search_space.get_search_space_definition()

        if is_col_order is not None:
            config_vec = reorder_vector_columns(config_vec, is_col_order, self._variables)

        for idx, val in enumerate(config_vec):
            key, transformed = self._transform_meta_data[idx]
            if transformed:
                choices = search_space_def[key]['allowed']
                config_val = choices[int(val)]
                dict_repr[key] = config_val
            else:
                if search_space_def[key]['is_log']:
                    val = np.exp(val)
                dict_repr[key] = val
        return dict_repr
    
def reorder_vector_columns(vector: list, is_col_order: list, target_col_order: list):
    """Given a configuration vector over a set of variables ordered as in is_col_order, reorder its entries 
        according to target_col_order.

    Example:
        Given a vector [1, 0, 0] with column order [A, B, C],  the vector will be reordered into [0, 1, 0] 
        to match with the column order of target_col_order=[B, A, C].

    Args:
        vector (list): Vector representing a configuration
        is_col_order (list): Column order of vector
        target_col_order (list): Target order of columns of vector

    Returns:
        list: Reordered vector.
    """
    new_vector = []
    for name in target_col_order:
        vec_col_idx = is_col_order.index(name)
        new_vector.append(vector[vec_col_idx])
    return new_vector