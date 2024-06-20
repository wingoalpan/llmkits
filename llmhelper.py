
import os
import torch
from torch import nn
from typing import Dict, Union
import transformers.modeling_utils as modeling_utils
from safetensors.torch import (
    save_file,
    load_file,
)
import json as js
import hashlib
from gguf import GGUFReader
# from gguf.constants import LlamaFileType
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAFE_WEIGHTS_NAME = 'model.safetensors'
SAFE_WEIGHTS_INDEX_NAME = 'model.safetensors.index.json'

from enum import IntEnum
class LlamaFileType(IntEnum):
    ALL_F32              = 0
    MOSTLY_F16           = 1   # except 1d tensors
    MOSTLY_Q4_0          = 2   # except 1d tensors
    MOSTLY_Q4_1          = 3   # except 1d tensors
    MOSTLY_Q4_1_SOME_F16 = 4   # tok_embeddings.weight and output.weight are F16
    # MOSTLY_Q4_2        = 5   # support has been removed
    # MOSTLY_Q4_3        = 6   # support has been removed
    MOSTLY_Q8_0          = 7   # except 1d tensors
    MOSTLY_Q5_0          = 8   # except 1d tensors
    MOSTLY_Q5_1          = 9   # except 1d tensors
    MOSTLY_Q2_K          = 10  # except 1d tensors
    MOSTLY_Q3_K_S        = 11  # except 1d tensors
    MOSTLY_Q3_K_M        = 12  # except 1d tensors
    MOSTLY_Q3_K_L        = 13  # except 1d tensors
    MOSTLY_Q4_K_S        = 14  # except 1d tensors
    MOSTLY_Q4_K_M        = 15  # except 1d tensors
    MOSTLY_Q5_K_S        = 16  # except 1d tensors
    MOSTLY_Q5_K_M        = 17  # except 1d tensors
    MOSTLY_Q6_K          = 18  # except 1d tensors
    MOSTLY_IQ2_XXS       = 19  # except 1d tensors
    MOSTLY_IQ2_XS        = 20  # except 1d tensors
    MOSTLY_Q2_K_S        = 21  # except 1d tensors
    MOSTLY_IQ3_XS        = 22  # except 1d tensors
    MOSTLY_IQ3_XXS       = 23  # except 1d tensors
    MOSTLY_IQ1_S         = 24  # except 1d tensors
    MOSTLY_IQ4_NL        = 25  # except 1d tensors
    MOSTLY_IQ3_S         = 26  # except 1d tensors
    MOSTLY_IQ3_M         = 27  # except 1d tensors
    MOSTLY_IQ2_S         = 28  # except 1d tensors
    MOSTLY_IQ2_M         = 29  # except 1d tensors
    MOSTLY_IQ4_XS        = 30  # except 1d tensors
    MOSTLY_IQ1_M         = 31  # except 1d tensors
    MOSTLY_BF16          = 32  # except 1d tensors

    GUESSED              = 1024  # not specified in the model file


# load parameters from safetensors files to nn model
def load_safetensors(model: nn.Module, model_path):
    safe_tensors_index_file = os.path.join(model_path, SAFE_WEIGHTS_INDEX_NAME)
    if os.path.exists(safe_tensors_index_file):
        modeling_utils.load_sharded_checkpoint(model, model_path, strict=True, prefer_safe=True)
    else:
        # if the safetensors is not sharded, the parameter can be the safetensors file or directory.
        safe_tensors_data_file = model_path if os.path.isfile(model_path) else os.path.join(model_path, SAFE_WEIGHTS_NAME)
        if os.path.split(safe_tensors_data_file)[-1] == SAFE_WEIGHTS_NAME:
            state_dict = load_file(safe_tensors_data_file)
            model.load_state_dict(state_dict, strict=True)


# load parameters from safetensors files to state_dict variable
def load_safetensors_to_state_dict(model_path):
    state_dict = {}
    safe_tensors_index_file = os.path.join(model_path, SAFE_WEIGHTS_INDEX_NAME)
    if os.path.exists(safe_tensors_index_file):
        with open(safe_tensors_index_file, "r", encoding="utf-8") as f:
            index = js.load(f)
        shard_files = list(set(index["weight_map"].values()))
        for shard_file in shard_files:
            partial_state_dict = load_file(os.path.join(model_path, shard_file))
            for k, v in partial_state_dict.items():
                state_dict[k] = v
    else:
        # if the safetensors is not sharded, the parameter can be the safetensors file or directory.
        safe_tensors_data_file = model_path if os.path.isfile(model_path) else os.path.join(model_path, SAFE_WEIGHTS_NAME)
        if os.path.split(safe_tensors_data_file)[-1] == SAFE_WEIGHTS_NAME:
            state_dict = load_file(safe_tensors_data_file)
    return state_dict


# save state_dict to safetensors files
def save_safetensors(state_dict: Dict[str, torch.Tensor], save_directory: Union[str, os.PathLike],
                     max_shard_size: Union[int, str] = "45MB"):
    if os.path.isfile(save_directory):
        log("Provided path ({save_directory}) should be a directory, not a file")
        return

    os.makedirs(save_directory, exist_ok=True)
    shards, index = modeling_utils.shard_checkpoint(state_dict,
                                                    max_shard_size=max_shard_size,
                                                    weights_name=SAFE_WEIGHTS_NAME)
    # Save the model
    for shard_file, shard in shards.items():
        save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})

    if index is not None:
        save_index_file = SAFE_WEIGHTS_INDEX_NAME
        save_index_file = os.path.join(save_directory, save_index_file)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = js.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)


# transform pytorch format state_dict file to safetensors files
def save_as_safetensors(state_dict_file, save_directory='models',
                        max_shard_size: Union[int, str] = "45MB"):
    state_dict = torch.load(state_dict_file, map_location=device)
    save_safetensors(state_dict, save_directory, max_shard_size=max_shard_size)


# save pytorch model's parameters to safetensors files
def save_pretrained(model, save_directory: Union[str, os.PathLike],
                    max_shard_size: Union[int, str] = "45MB"):
    state_dict = model.state_dict()
    save_safetensors(state_dict, save_directory, max_shard_size=max_shard_size)


def _id2dtype(file_type):
    file_type_name = LlamaFileType(file_type).name
    parts = file_type_name.split('_')
    dtype_name = '_'.join(parts[1:3])
    scope = parts[0].lower()
    if dtype_name == 'F32':
        return scope, 'float32'
    elif dtype_name == 'F16':
        return scope, 'float16'
    elif dtype_name == 'BF16':
        return scope, 'bfloat16'
    else:
        return scope, dtype_name.lower()


# load parameters from gguf files to state_dict variable
def load_gguf_to_state_dict(model_path):
    state_dict = {}
    state_meta = {}
    data = GGUFReader(model_path)
    info = data.fields
    scope = 'mostly'
    dtype = None
    for key, value in info.items():
        if key == 'general.file_type':
            file_type = value.parts[value.data[0]][0].item()
            scope, dtype = _id2dtype(file_type)
            break
    tensors = data.tensors
    for tensor in tensors:
        param = torch.from_numpy(tensor.data)
        shape = list(tensor.shape)
        # now ALL_* pattern only support ALL_F32.
        # if more ALL_* are supported in the future, the following code should be optimized
        tensor_dtype = dtype if scope == 'mostly' and len(shape) > 1 else None
        state_dict[tensor.name] = param  # .view(shape)
        state_meta[tensor.name] = {'shape': shape, 'dtype': tensor_dtype}
    return state_dict, state_meta


def load_to_state_dict(model_path):
    state_meta = None
    if os.path.isfile(model_path) and model_path.endswith('.gguf'):
        state_dict, state_meta = load_gguf_to_state_dict(model_path)
    else:
        state_dict = load_safetensors_to_state_dict(model_path)
    return state_dict, state_meta


def sha256(tensor: torch.Tensor):
    sha_obj = hashlib.sha256()
    if tensor.dtype == torch.bfloat16:
        sha_obj.update(tensor.to(torch.float).numpy())
    else:
        sha_obj.update(tensor.numpy())
    return sha_obj.hexdigest()


# 从state_dict 获取模型参数的schema (参数名称、数据类型、形状、参数值sha256、参数device)
def model_meta_params(state_dict, state_meta=None):
    params = []
    for k, v in state_dict.items():
        # 预处理: 获取该parameter tensor的 meta数据，如果有的话
        meta = state_meta.get(k, None) if state_meta else None
        data_type = meta['dtype'] if meta and meta.get('dtype', None) else str(v.dtype).split('.')[-1]
        shape = meta['shape'] if meta else list(v.shape)
        sha = sha256(v)
        device_ = str(v.device)
        params.append({'name': k, 'size': shape, 'dtype': data_type, 'sha256': sha, 'device': device_})
    return params


# 将模型参数的schema (参数名称、数据类型、形状、参数值sha256、参数device) 保存到excel中
def save_meta_params(state_dict, save_excel_file, state_meta=None):
    params = model_meta_params(state_dict, state_meta)
    params_df = pd.DataFrame(params)
    params_df.to_excel(save_excel_file, index=False)


# 比较两套模型参数，将比较结果保存到 excel中
def diff_params(src, dest, save_excel_file, ignore_fields=None):
    if ignore_fields is None:
        ignore_fields = []
    fields = ['name', 'size', 'dtype', 'sha256', 'device']
    key_field = 'name'
    src_dict = {p[key_field]: p for p in src}
    dest_dict = {p[key_field]: p for p in dest}
    diffs = []
    for param in src:
        key = param[key_field]
        param_cmp = {'src_' + k: v for k, v in param.items() if k not in ignore_fields}
        param_dest = dest_dict.get(key, None)
        if not param_dest:
            param_cmp.update({'dest_' + k: None for k, _ in param.items() if k not in ignore_fields})
            param_cmp['flag'] = '-'  # dest中被删除了
            param_cmp['different_field'] = None
        else:
            param_cmp.update({'dest_' + k: v for k, v in param_dest.items() if k not in ignore_fields})
            different_field = None
            for k in fields:
                if param[k] != param_dest[k] and k not in ignore_fields:
                    different_field = k
                    break
            param_cmp['flag'] = 'M' if different_field else None
            param_cmp['different_field'] = different_field
        diffs.append(param_cmp)

    for param_dest in dest:
        key = param_dest[key_field]
        param = src_dict.get(key, None)
        if not param:
            param_cmp = {'src_' + k: None for k, _ in param_dest.items() if k not in ignore_fields}
            param_cmp.update({'dest_' + k: v for k, v in param_dest.items() if k not in ignore_fields})
            param_cmp['flag'] = '+'  # dest中新增的
            diffs.append(param_cmp)

    params_df = pd.DataFrame(diffs)
    # adjust the order of the columns
    columns = (['flag'] + ['src_' + k for k in fields if k not in ignore_fields]
               + ['dest_' + k for k in fields if not k in ignore_fields]
               + ['different_field'])
    params_df = params_df[columns]
    params_df.to_excel(save_excel_file, index=False)


def gguf_meta_hyperparams(model_path):
    data = GGUFReader(model_path)
    info = data.fields
    hyperparams = {}
    for key, value in info.items():
        if value.types[0] == 8:
            v_data = value.parts[value.data[0]][:].tobytes().decode('utf-8')
        elif value.types[0] == 9:
            alen = len(value.data)
            idx = value.data[0]
            val_list = []
            for i in range(min(10, alen)):
                memmap_val_list = value.parts[idx + i].tolist()
                if len(memmap_val_list) == 1:
                    val_list.append(memmap_val_list[0])
                else:
                    val_list.append(memmap_val_list)
            v_data = {'len': alen, 'first_10_values': val_list}
        else:
            v_data = value.parts[value.data[0]][0].item()
        hyperparams[key] = {'name': key, 'value': v_data, 'value_type': repr(value.types[0]), 'offset': value.offset}
    return hyperparams
