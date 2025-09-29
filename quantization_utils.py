import torch

def quantize(weights, group_size, num_bits, along_column=False, method='rtn'):
    if method == 'bin':
        assert num_bits==1, 'must quantize to 1 bit if using binary quantization'
        return one_bit_quantize(weights, group_size, along_column)
    elif method == 'rtn':
        return RTN(weights, num_bits, group_size, along_column)
    
    
    
def one_bit_quantize(weights, group_size, along_column=False):
    if along_column:
        weights = weights.T
    shape = weights.shape
    flattened = weights.contiguous().view(-1)  # Flatten the tensor for easy processing
    num_groups = (flattened.numel() + group_size - 1) // group_size  # Ensure all elements are covered
    
    # Pad tensor if needed
    pad_size = num_groups * group_size - flattened.numel()
    if pad_size > 0:
        flattened = torch.cat([flattened, torch.zeros(pad_size, device=weights.device)])
    
    # Reshape into groups
    groups = flattened.view(num_groups, group_size)
    
    # Compute scaling factors per group
    # zero_points = groups.mean(dim=1, keepdim=True)
    groups = groups #- zero_points
    scaling_factors = groups.abs().mean(dim=1, keepdim=True)    
    # Dequantize
    dequantized = torch.sign(groups) * scaling_factors # + zero_points
    
    # Reshape back to original shape (removing padding if any)
    dequantized = dequantized.view(-1)[:shape.numel()].view(shape)
    
    return dequantized if not along_column else dequantized.T

def RTN(w, num_bits=8,  group_size=-1, along_column=False):
    if along_column:
        w = w.T

    org_w_shape = w.shape
    if group_size > 0:
        # assert org_w_shape[-1] % group_size == 0
        w = w.reshape(-1, group_size)
    assert w.dim() == 2

    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**num_bits - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    w = (
        torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
    ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    return w if not along_column else w.T