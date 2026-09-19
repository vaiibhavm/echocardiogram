import torch
from echo_research.losses import SegmentationLoss, BoundaryDiceLoss, LVFunctionalConsistencyLoss


def test_binary_losses_finite():
    led=torch.randn(2,1,64,64,requires_grad=True); les=torch.randn(2,1,64,64,requires_grad=True)
    yed=(torch.rand(2,1,64,64)>0.6).float(); yes=(torch.rand(2,1,64,64)>0.7).float()
    s,_=SegmentationLoss("binary")(led,yed); b=BoundaryDiceLoss("binary")(led,yed); f,_=LVFunctionalConsistencyLoss("binary")(led,les,yed,yes)
    loss=s+b+f
    assert torch.isfinite(loss)
    loss.backward()
    assert led.grad is not None


def test_multiclass_losses_finite():
    led=torch.randn(2,4,32,32,requires_grad=True); les=torch.randn(2,4,32,32,requires_grad=True)
    yed=torch.randint(0,4,(2,32,32)); yes=torch.randint(0,4,(2,32,32))
    s,_=SegmentationLoss("multiclass",4)(led,yed); b=BoundaryDiceLoss("multiclass",4)(led,yed); f,_=LVFunctionalConsistencyLoss("multiclass")(led,les,yed,yes)
    loss=s+b+f
    assert torch.isfinite(loss)
    loss.backward()
