import torch
import torch.nn as nn
import torch.nn.functional as F# 纯计算就用functional这个库 如果是需要参数在训练的时候就使用nn.moudle

from .conv import Conv, Conv2d, autopad

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):#数据流 一个张量进入 1*1卷积 分离 一部分直接到底部 一部分经过bottlenect 经过bottlenect时候都会复制一份用于底部的融合

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2*self.c, 1, 1)
        self.cv2 = Conv((2+n)*self.c, c2, 1)
        self.m = nn.ModuleList([Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)])  # module list

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))  # split channels
        y.extend([m(y[-1]) for m in self.m])  # apply bottlenecks
        return self.cv2(torch.cat(y, 1))  # concatenate and apply final conv

    def forward_split(self, x:torch.Tensor) -> torch.Tensor:
        y = self.cv1(x).split((self.c, self.c), 1)  # split channels
        y=[y[0],y[1]]
        y.extend(m(y[-1]) for m in self.m)  # apply bottlenecks
        return self.cv2(torch.cat(y, 1))  # concatenate and apply final conv

class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):

        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n=n
        self.add=shortcut and c1 == c2

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        y=[self.cv1(x)]
        y.extend([self.m(y[-1]) for _ in range(getattr(self,'n',3))])
        y= self.cv2(torch.cat(y, 1))
        return x+y if getattr(self,'add',False) else y

