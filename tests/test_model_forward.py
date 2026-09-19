import torch
from echo_research.models import Pix2PixResearchModel


def test_binary_forward():
    m=Pix2PixResearchModel("binary",base_channels=8)
    x=torch.randn(1,1,256,256)
    y=m.generator(x)
    assert y.shape==(1,1,256,256)
    d=m.discriminator(x,m.mask_representation(y,True))
    assert d.ndim==4 and d.shape[0]==1


def test_multiclass_forward():
    m=Pix2PixResearchModel("multiclass",num_classes=4,base_channels=8)
    x=torch.randn(1,1,256,256)
    y=m.generator(x)
    assert y.shape==(1,4,256,256)
