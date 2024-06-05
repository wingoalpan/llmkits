
from torch import nn


# 检查一个 nn.Module是否有子模块
# 输入: module -- 必须是一个 nn.Module类型的实例
# 输出: bool
#          True  -- module有包含子模块
#          False -- module没有子模块
def _has_children(module: nn.Module):
    for _ in module.children():
        return True
    return False


# 检查一个 nn.Module是否有注册的buffer
# 输入: module -- 必须是一个 nn.Module类型的实例
# 输出: bool
#          True  -- module有包含子模块
#          False -- module没有子模块
def _has_buffers(module: nn.Module):
    for _ in module.buffers():
        return True
    return False


def check_net(model: nn.Module, x, model_name='-', level=0, as_singles=None):
    lines = []
    if level == 0:
        lines.append('input data shape: {}'.format(x.shape))
    indent_str = ' ' * level * 2
    if as_singles is None:
        as_singles = []

    class_name = model.__class__.__name__
    if class_name in as_singles:
        lines.append('%s(%s): %s' % (indent_str, model_name, model.__class__.__name__))
        x = model(x)
        lines.append('{indent}{output_flag} {shape}'.format(indent=' ' * 10, output_flag='=' * 4, shape=x.shape))
    elif not _has_children(model):
        lines.append('%s(%s): %s' % (indent_str, model_name, model))
        x = model(x)
        lines.append('{indent}{output_flag} {shape}'.format(indent=' ' * 10, output_flag='=' * 4, shape=x.shape))
    else:
        children_lines = []
        for name, child in model.named_children():
            x, child_info = check_net(child, x, name, level + 1, as_singles)
            children_lines.append(child_info)
        lines.append('%s(%s): %s (' % (indent_str, model_name, model.__class__.__name__))
        lines.extend(children_lines)
        lines.append('%s)' % indent_str)

    if level == 0:
        lines.append('output data shape: {}'.format(x.shape))
    return x, '\n'.join(lines)


def get_net_detail(model: nn.Module, model_name='-', show_param_shape=False, level=0):
    lines = []
    param_count = 0
    indent_str = ' ' * level * 2
    if not _has_children(model) and not _has_buffers(model):
        param_count, _ = _count_params(model, show_param_shape=False)
        if param_count > 0:
            lines.append('%s(%s): %s [params: %s]' % (indent_str, model_name, model, param_count))
            if show_param_shape:
                _, params_shape = _count_params(model, show_param_shape=True)
                lines.append(params_shape)
        else:
            lines.append('%s(%s): %s' % (indent_str, model_name, model))
    else:
        children_lines = []
        for name, child in model.named_children():
            p_count, child_info = get_net_detail(child, name, show_param_shape, level + 1)
            param_count += p_count
            children_lines.append(child_info)
        for name, buf in model.named_buffers(recurse=False):
            p_count, buffer_info = get_buffer_info(buf, name, show_param_shape, level + 1)
            param_count += p_count
            children_lines.append(buffer_info)
        lines.append('%s(%s): %s [params: %s] (' % (indent_str, model_name, model.__class__.__name__, param_count))
        lines.extend(children_lines)
        lines.append('%s)' % indent_str)
    return param_count, '\n'.join(lines)


def _count_params(model: nn.Module, show_param_shape=False):
    p_count = 0
    lines = []
    for p in model.parameters():
        shape = p.data.shape
        if show_param_shape:
            lines.append('{indent}{param_flag} {shape}'.format(indent=' ' * 10, param_flag='#' * 4, shape=shape))
        if len(shape) > 1:
            cnt = shape[0]
            for i in range(1, len(shape)):
                cnt = cnt * shape[i]
            p_count += cnt
        else:
            p_count += shape[0]
    return p_count, '\n'.join(lines)


def get_buffer_info(buf, name, show_param_shape, level):
    p_count = 0
    lines = []

    indent_str = ' ' * level * 2
    shape = buf.shape
    if len(shape) > 1:
        cnt = shape[0]
        for i in range(1, len(shape)):
            cnt = cnt * shape[i]
        p_count += cnt
    else:
        p_count += shape[0]
    lines.append('%s(%s): %s [params: %s]' % (indent_str, name, '<BUFFER>', p_count))
    if show_param_shape:
        lines.append('{indent}{param_flag} {shape}'.format(indent=' ' * 10, param_flag='#' * 4, shape=shape))
    return p_count, '\n'.join(lines)

