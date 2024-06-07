
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
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAFE_WEIGHTS_NAME = 'model.safetensors'
SAFE_WEIGHTS_INDEX_NAME = 'model.safetensors.index.json'


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


# load parameters from gguf files to state_dict variable
def load_gguf_to_state_dict(model_path):
    state_dict = {}
    data = GGUFReader(model_path)
    # info = data.fields
    tensors = data.tensors
    for tensor in tensors:
        param = torch.from_numpy(tensor.data)
        shape = list(tensor.shape)
        state_dict[tensor.name] = param #.view(shape)
    return state_dict


def sha256(tensor: torch.Tensor):
    sha_obj = hashlib.sha256()
    sha_obj.update(tensor.numpy())
    return sha_obj.hexdigest()


# 从state_dict 获取模型参数的schema (参数名称、数据类型、形状、参数值sha256、参数device)
def model_meta_params(state_dict):
    params = []
    for k, v in state_dict.items():
        data_type = str(v.dtype).split('.')[-1]
        shape = list(v.shape)
        sha = sha256(v)
        device_ = str(v.device)
        params.append({'name': k, 'size': shape, 'dtype': data_type, 'sha256': sha, 'device': device_})
    return params


# 将模型参数的schema (参数名称、数据类型、形状、参数值sha256、参数device) 保存到excel中
def save_meta_params(state_dict, save_excel_file):
    params = model_meta_params(state_dict)
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
        else:
            param_cmp.update({'dest_' + k: v for k, v in param_dest.items() if k not in ignore_fields})
            is_different = any([param[k] != param_dest[k] for k in fields if k not in ignore_fields])
            param_cmp['flag'] = 'M' if is_different else None
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
               + ['dest_' + k for k in fields if not k in ignore_fields])
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
