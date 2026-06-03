try:
    import orjson as json
except ImportError:  # pragma: no cover
    import json

import numpy as np
import pandas as pd


def util_to_json_from_pandas(data):
    """
    explanation:
        将pandas数据转换成json格式

    params:
        * data ->:
            meaning: pandas数据
            type: null
            optional: [null]

    return:
        dict

    demonstrate:
        Not described

    output:
        Not described
    """

    """需要对于datetime 和date 进行转换, 以免直接被变成了时间戳"""
    if 'datetime' in data.columns:
        data.datetime = data.datetime.apply(str)
    if 'date' in data.columns:
        data.date = data.date.apply(str)
    return json.loads(data.to_json(orient='records'))


def util_to_json_from_numpy(data):
    pass


def util_to_json_from_list(data):
    pass


def util_to_list_from_pandas(data):
    """
    explanation:
         将pandas数据转换成列表

    params:
        * data ->:
            meaning: pandas数据
            type: null
            optional: [null]

    return:
        list

    demonstrate:
        Not described

    output:
        Not described
    """

    return np.asarray(data).tolist()


def util_to_list_from_numpy(data):
    """
    explanation:
        将numpy数据转换为列表

    params:
        * data ->:
            meaning: numpy数据
            type: null
            optional: [null]

    return:
        None

    demonstrate:
        Not described

    output:
        Not described
    """

    return data.tolist()


def util_to_pandas_from_json(data):
    """
    explanation:
        将json数据载入为pandas数据

    params:
        * data ->:
            meaning: json数据
            type: null
            optional: [null]

    return:
        DataFrame

    demonstrate:
        Not described

    output:
        Not described
    """
    if isinstance(data, dict):
        return pd.DataFrame(data=[data, ])
    else:
        return pd.DataFrame(data=[{'value': data}])


def util_to_pandas_from_list(data):
    """
    explanation:
        将列表数据转换为pandas

    params:
        * data ->:
            meaning: 列表数据
            type: list
            optional: [null]

    return:
        DataFrame

    demonstrate:
        Not described

    output:
        Not described
    """

    if isinstance(data, list):
        return pd.DataFrame(data=data)
