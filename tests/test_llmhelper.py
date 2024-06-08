
import sys, os
import argparse
import time
import torch
import torch.nn as nn
import json as js

from wingoal_utils.common import (
    set_log_file,
    log,
    logs
)
sys.path.insert(0, '..\\..\\..\\ml-learning\\transformer')
sys.path.append('..\\..\\..\\ml-learning')
from transformer import Transformer, validate
from dataset_P3n9W31 import TP3n9W31Data

sys.path.append('..')
import llmhelper

set_log_file(os.path.split(__file__)[-1], timestamp=True)


def main():
    test_load_safetensors()


# load multiple sharded safetensors files / single safetensors file
def test_load_safetensors():
    # multiple sharded safetensors files are stored in model_save_dir
    model_save_dir = '../../../temp_ml_models/transformer/models/transformer-P3n9W31_0.1-800'
    save_directory = 'models/transformer-P3n9W31_0.1-800'
    device = 'cpu'
    corpora, name, batch_size, num_epochs = TP3n9W31Data(), 'classical', 48, 800
    _dropout = 0.1
    model = Transformer(corpora, name, dropout=_dropout).to(device)
    llmhelper.load_safetensors(model, model_save_dir)
    # save the models as single safetensors file in save_directory
    llmhelper.save_pretrained(model, save_directory, '200MB')
    log('reload the single safetensors file...')

    # reload the single safetensors file and evaluate the model
    llmhelper.load_safetensors(model, save_directory)

    log('validating model ...')
    result = validate(model, 100)
    if result['success']:
        log('validate result: OK')
    else:
        log('validate result: Failed (failed = %s, passed = %s)' % (result['failed'], result['passed']))
        logs('validate detail:', js.dumps(result['detail'], indent=2, ensure_ascii=False))


def test_load_to_state_dict():
    # case 1. multiple sharded safetensors files or single safetensors file are stored in model_path directory
    # model_path = '../../../temp_ml_models/transformer/models/transformer-P3n9W31_0.1-800'
    # case 2. model_path is the single safetensors file
    model_path = 'models/transformer-P3n9W31_0.1-800/model.safetensors'
    device = 'cpu'
    corpora, name, batch_size, num_epochs = TP3n9W31Data(), 'classical', 48, 800
    model = Transformer(corpora, name).to(device)
    # first load multiple sharded safetensors to state_dict
    state_dict = llmhelper.load_safetensors_to_state_dict(model_path)
    # then load state_dict to model
    model.load_state_dict(state_dict)

    log('validating model ...')
    result = validate(model, 100)
    if result['success']:
        log('validate result: OK')
    else:
        log('validate result: Failed (failed = %s, passed = %s)' % (result['failed'], result['passed']))
        logs('validate detail:', js.dumps(result['detail'], indent=2, ensure_ascii=False))


# test the llmhelper.load_gguf_to_state_dict(...) function
def test_load_gguf_to_state_dict():
    gguf_file = '../../models/out_Meta-Llama-3-8B-gguf/llama-3-8b-model.gguf'
    state_dict, state_meta = llmhelper.load_gguf_to_state_dict(gguf_file)
    params = llmhelper.model_meta_params(state_dict)
    print(js.dumps(params, indent=2))


def test_sha256():
    a = torch.rand(3, 5)
    b = a.clone()
    sha_a = llmhelper.sha256(a)
    sha_b = llmhelper.sha256(b)
    print(sha_a, sha_b)
    assert sha_a == sha_b, "Sha256 of a and b should be equal."

    b[1, 1] = 0.0
    sha_b = llmhelper.sha256(b)
    print(sha_a, sha_b)
    assert sha_a != sha_b, "Sha256 of a and b should be different."

    b[1, 1] = a[1, 1]
    sha_b = llmhelper.sha256(b)
    print(sha_a, sha_b)
    assert sha_a == sha_b, "Sha256 of a and b should be equal after b is restored."


def test_model_meta_params():
    model_save_dir = '../../../temp_ml_models/transformer/models/transformer-P3n9W31_0.1-800'
    state_dict = llmhelper.load_safetensors_to_state_dict(model_save_dir)
    params = llmhelper.model_meta_params(state_dict)
    print(js.dumps(params, indent=2))


# test these 3 functions:
#    1. llmhelper.load_gguf_to_state_dict(...) function
#    2. llmhelper.model_meta_params(...) function (this function was invoked inside load_gguf_to_state_dict(...))
#    3. llmhelper.save_meta_params(...) function
def test_save_meta_params():
    gguf_file = '../../models/Llama-3-Chinese-8B-Instruct-v2-GGUF/ggml-model-q4_0.gguf'
    save_excel_file = 'logs/Llama-3-Chinese-8B-Instruct-v2-GGUF-q4_0.xlsx'
    save_raw_excel_file = 'logs/Llama-3-Chinese-8B-Instruct-v2-GGUF-q4_0-raw.xlsx'
    state_dict, state_meta = llmhelper.load_gguf_to_state_dict(gguf_file)
    llmhelper.save_meta_params(state_dict, save_excel_file, state_meta)
    llmhelper.save_meta_params(state_dict, save_raw_excel_file)


def test_diff_params():
    # 1. test the gguf format model file
    src_gguf_file = '../../models/Llama-3-Chinese-8B-Instruct-v2-GGUF/ggml-model-f16.gguf'
    dest_gguf_file = '../../models/Llama-3-Chinese-8B-Instruct-v2-GGUF/ggml-model-q4_0.gguf'
    save_excel_file = 'logs/Llama-3-Chinese-8B-Instruct-v2-GGUF-diffs-f16-vs-q4_0.xlsx'
    src_state_dict, src_state_meta = llmhelper.load_gguf_to_state_dict(src_gguf_file)
    dest_state_dict, dest_state_meta = llmhelper.load_gguf_to_state_dict(dest_gguf_file)
    src_params = llmhelper.model_meta_params(src_state_dict, src_state_meta)
    dest_params = llmhelper.model_meta_params(dest_state_dict, dest_state_meta)
    llmhelper.diff_params(src_params, dest_params, save_excel_file, ignore_fields=['device'])
    # 2. test the safetensors format model file
    src_safe_file = '../../models/Llama-3-Chinese-8B-Instruct-v2'
    dest_safe_file = '../../models/Llama-3-Chinese-8B-Instruct-v3'
    save_excel_file = 'logs/Llama-3-Chinese-8B-Instruct-diffs-v2-vs-v3.xlsx'
    src_state_dict = llmhelper.load_safetensors_to_state_dict(src_safe_file)
    dest_state_dict = llmhelper.load_safetensors_to_state_dict(dest_safe_file)
    src_params = llmhelper.model_meta_params(src_state_dict)
    dest_params = llmhelper.model_meta_params(dest_state_dict)
    llmhelper.diff_params(src_params, dest_params, save_excel_file, ignore_fields=['device'])


def test_gguf_meta_hyperparams():
    gguf_file = '../../models/out_Meta-Llama-3-8B-gguf/llama-3-8b-model-q4_0.gguf'
    hyperparams = llmhelper.gguf_meta_hyperparams(gguf_file)
    print(js.dumps(hyperparams, indent=2))
    #print(hyperparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("params", nargs="*")
    args = parser.parse_args()
    if len(args.params) == 0:
        log('executing function [main] ...')
        main()
    else:
        func = args.params[0]
        if func != 'main':
            set_log_file(os.path.split(__file__)[-1], suffix=func, timestamp=True)
        param_list = args.params[1:]
        log('executing function [%s] ...' % func)
        eval(func)(*param_list)
    log('finish executing function!')

