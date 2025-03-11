import torch 
from evoformer import TritonEvoformer
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention


### setup data
batch_size = 2
n_seq = 18
n_res = 20
c_hidden = 32
no_heads = 4
eps = 2e-2
inf=1e9

dtype = torch.bfloat16
q = torch.rand(
    batch_size, n_seq, n_res,no_heads, c_hidden, requires_grad=False
).type(dtype).cuda()

kv = torch.rand(
    batch_size, n_seq, n_res, no_heads, c_hidden, requires_grad=False
).type(dtype).cuda()

mask = torch.randint(
        0, 2, (batch_size, n_seq, 1, 1, n_res),requires_grad=False
    ).type(dtype).cuda()

z_bias = torch.rand(
    batch_size, 1, no_heads, n_res, n_res, requires_grad=False
).type(dtype).cuda()

mask_bias = inf * (mask - 1)
biases = [mask_bias, z_bias]

### run triton kernel 
triton_out = TritonEvoformer(
    q, kv, kv, mask, biases[1]
).cpu()
print(triton_out.shape)
print("\n\n")

#### run deepspeed kernel 
ds_out = DS4Sci_EvoformerAttention(
    q, kv,kv, biases
).cpu()
print(ds_out.shape)


err = torch.max(torch.abs(ds_out - triton_out))
print(err)

print("Done")