import time
from collections import defaultdict
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x

class LatencyMeasurementMode(TorchDispatchMode):
    def __init__(self, module = None):
        self.parents = ['Global']
        if module is not None:
            for name, module in dict(module.named_children()).items():
                module.register_forward_pre_hook(self.enter_module(name))
                module.register_forward_hook(self.exit_module(name))

    def enter_module(self, name):
        def f(module, inputs):
            self.parents.append(name)
            inputs = normalize_tuple(inputs)
            out = self.create_backwards_pop(name)(*inputs)
            return out

        return f

    def exit_module(self, name):
        def f(module, inputs, outputs):
            assert(self.parents[-1] == name)
            self.parents.pop()
            outputs = normalize_tuple(outputs)
            return self.create_backwards_push(name)(*outputs)
        return f

    def create_backwards_push(self, name):
        class PushState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                self.parents.append(name)
                return grad_outs

        return PushState.apply

    def create_backwards_pop(self, name):
        class PopState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                assert(self.parents[-1] == name)
                self.parents.pop()
                return grad_outs

        return PopState.apply


    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        start_time = time.time()
        out = func(*args, **kwargs)
        end_time = time.time()
        latency = end_time - start_time
        func_packet = func._overloadpacket
        print(f"{func} Latency: {latency:.6f} seconds")

        return out

if __name__ == "__main__":
    import torchvision.models as models

    inp = torch.randn(1, 3, 224, 224, device='cpu')
    mod = models.resnet50()
    optimizer = torch.optim.Adam(mod.parameters(), lr=0.001)

    latency_counter = LatencyMeasurementMode(mod)
    
    optimizer.zero_grad()
    outputs = mod(inp)
    loss = outputs.sum()
    with latency_counter:
        
        loss.backward()
    optimizer.step()
