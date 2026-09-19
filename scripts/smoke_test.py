#!/usr/bin/env python3
"""No-data smoke test: model forward/backward and all proposed losses."""
import torch
from echo_research.models.pix2pix import Pix2PixResearchModel
from echo_research.losses import SegmentationLoss, BoundaryDiceLoss, LVFunctionalConsistencyLoss


def run(task="binary"):
    n=1; h=w=256
    model=Pix2PixResearchModel(task=task,num_classes=4,base_channels=8)
    xed=torch.rand(n,1,h,w); xes=torch.rand(n,1,h,w)
    if task=="binary":
        yed=(torch.rand(n,1,h,w)>0.7).float(); yes=(torch.rand(n,1,h,w)>0.75).float()
    else:
        yed=torch.randint(0,4,(n,h,w)); yes=torch.randint(0,4,(n,h,w))
    led=model.generator(xed); les=model.generator(xes)
    seg,_=SegmentationLoss(task,4)(led,yed)
    bd=BoundaryDiceLoss(task,4)(led,yed)
    fn,_=LVFunctionalConsistencyLoss(task)(led,les,yed,yes)
    fake=model.mask_representation(led,True)
    d=model.discriminator(xed,fake)
    loss=seg+bd+fn+d.mean()*0.0
    loss.backward()
    print(task, "PASS", tuple(led.shape), tuple(d.shape), float(loss.detach()))


if __name__ == "__main__":
    run("binary")
    run("multiclass")
