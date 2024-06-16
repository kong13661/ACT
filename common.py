import inspect
import typing


def get_type(typing_value):
    if typing_value is str:
        return 'text'
    if typing_value is int:
        return 'integer'
    if typing_value is float:
        return 'double precision'
    if typing_value is bool:
        return 'boolean'
    if typing.get_origin(typing_value) is list or typing.get_origin(typing_value) is tuple:
        return get_type(typing.get_args(typing_value)[0]) + ' []'
    if typing_value is inspect._empty:
        raise TypeError('function signature is needed.')


def get_database_meta_from_funcion_signature(config, function_signature: inspect.Signature, program_meta_argument=None):
    program_meta_argument = program_meta_argument or []
    all_keys = function_signature.parameters.keys()

    database_meta = {}
    for key in all_keys:
        if key not in program_meta_argument and function_signature.parameters[key].annotation is not inspect._empty:
            database_meta[key.lower()] = [
                get_type(function_signature.parameters[key].annotation), config[key]]

    return database_meta


def get_params_to_record_from_funcion_signature(config, function_signature: inspect.Signature, program_meta_argument=None):
    program_meta_argument = program_meta_argument or []
    all_keys = function_signature.parameters.keys()

    params_to_record = {}
    for key in all_keys:
        if key not in program_meta_argument and function_signature.parameters[key].annotation is not inspect._empty:
            params_to_record[key] = config[key]
    return params_to_record
