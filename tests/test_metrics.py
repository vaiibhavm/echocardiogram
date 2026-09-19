import numpy as np
from echo_research.metrics.segmentation import segmentation_metrics
from echo_research.metrics.clinical import estimate_biplane_volume_ml,ejection_fraction


def test_identical_mask_metrics():
    x=np.zeros((64,64),bool); x[10:40,20:50]=1
    m=segmentation_metrics(x,x,(0.3,0.3))
    assert abs(m["dice"]-1)<1e-8 and m["hd95"]==0


def test_biplane_positive_volume():
    a=np.zeros((100,100),bool); a[20:80,35:65]=1
    v=estimate_biplane_volume_ml(a,(0.3,0.3),a,(0.3,0.3))
    assert v>0
    assert 0 < ejection_fraction(v, v*0.4) < 1
