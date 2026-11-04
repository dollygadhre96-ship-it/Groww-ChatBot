from dataclasses import dataclass
import pytest
pytest.importorskip("chromadb")
from src.vector_store import ChromaStore

@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict

def test_upsert_count_query(tmp_path):
    s=ChromaStore(tmp_path/"db","test_col")
    s.upsert_chunks([Chunk("a","large cap fund",{"source":"groww"})],[[1.,0.,0.]])
    assert s.count()==1
    r=s.query([1.,0.,0.],1)[0]
    assert r.id=="a" and r.text=="large cap fund" and r.metadata["source"]=="groww"
    assert r.distance==pytest.approx(0,abs=1e-5)

def test_upsert_is_idempotent(tmp_path):
    s=ChromaStore(tmp_path/"db","test_col")
    s.upsert_chunks([Chunk("same","old",{})],[[1.,0.]])
    s.upsert_chunks([Chunk("same","new",{})],[[0.,1.]])
    assert s.count()==1 and s.query([0.,1.],1)[0].text=="new"

def test_persistence(tmp_path):
    p=tmp_path/"persist"; ChromaStore(p,"persist_col").upsert_chunks([Chunk("x","stored",{})],[[1.,0.]])
    s=ChromaStore(p,"persist_col")
    assert s.count()==1 and s.query([1.,0.],1)[0].text=="stored"

def test_validation_empty_reset(tmp_path):
    s=ChromaStore(tmp_path/"db","test_col")
    assert s.upsert_chunks([],[])==0 and s.query([1.],1)==[]
    with pytest.raises(ValueError): s.upsert_chunks([Chunk("a","x",{})],[])
    with pytest.raises(ValueError): s.query([1.],0)
    s.upsert_chunks([Chunk("a","x",{})],[[1.,0.]])
    s.reset(); assert s.count()==0
