from torch.nn import ReLU, ELU, Identity, Tanh
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN

def MLP(channels, batch_norm=True, act='relu'):
    
    if isinstance(act, str) or act is None:
        act = [act]*(len(channels)-1)
    
    activation = []
    for a in act:
        if a is None:
            activation.append(Identity)
        elif a == 'relu':
            activation.append(ReLU)
        elif a == 'elu':
            activation.append(ELU)
        elif a == 'tanh':
            activation.append(Tanh)
        else:
            raise ValueError("unrecognized keyword: {}".format(act))
    
    if batch_norm:
        return Seq(*[
                Seq(Lin(channels[i-1], channels[i]), activation[i-1](), BN(channels[i]))
                for i in range(1, len(channels))
            ])
    else:
        return Seq(*[
                Seq(Lin(channels[i-1], channels[i]), activation[i-1]())
                for i in range(1, len(channels))
            ])
