import pytest
from echo_research.data.camus import validate_split_safety


def test_split_safety_accepts_disjoint():
    validate_split_safety(["p1"],["p2"],["p3"])


def test_split_safety_rejects_leakage():
    with pytest.raises(RuntimeError):
        validate_split_safety(["p1"],["p1"],["p3"])

from echo_research.data.camus import load_split_ids


def test_automatic_split_is_disjoint(tmp_path):
    data=tmp_path/'database_nifti'; data.mkdir()
    for i in range(30): (data/f'patient{i:04d}').mkdir()
    tr=load_split_ids(tmp_path,'train',seed=7)
    va=load_split_ids(tmp_path,'val',seed=7)
    te=load_split_ids(tmp_path,'test',seed=7)
    validate_split_safety(tr,va,te)
    assert len(set(tr)|set(va)|set(te))==30


def test_explicit_test_never_enters_auto_validation(tmp_path):
    data=tmp_path/'database_nifti'; data.mkdir()
    ids=[f'patient{i:04d}' for i in range(20)]
    for pid in ids: (data/pid).mkdir()
    s=tmp_path/'database_split'; s.mkdir()
    (s/'subgroup_training.txt').write_text('\n'.join(ids[:18]))
    (s/'subgroup_testing.txt').write_text('\n'.join(ids[18:]))
    tr=load_split_ids(tmp_path,'train',seed=3)
    va=load_split_ids(tmp_path,'val',seed=3)
    te=load_split_ids(tmp_path,'test',seed=3)
    validate_split_safety(tr,va,te)
    assert set(te)==set(ids[18:])
