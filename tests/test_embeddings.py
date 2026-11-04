import pytest
from src.embeddings.service import EmbeddingService
class FakeModel:
    def encode(self, texts, **kwargs): return [[float(len(t)), 1.0] for t in texts]
def service():
    obj = EmbeddingService.__new__(EmbeddingService); obj.model_name="fake"; obj.model=FakeModel(); return obj
def test_embed_texts(): assert service().embed_texts(["hello","world!"]) == [[5.0,1.0],[6.0,1.0]]
def test_empty_batch(): assert service().embed_texts([]) == []
def test_blank_rejected():
    with pytest.raises(ValueError): service().embed_texts([" "])
def test_query_embedding(): assert service().embed_query("hello") == [5.0,1.0]
