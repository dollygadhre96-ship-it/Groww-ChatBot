from src.retrieval import Retriever
from src.vector_store.store import SearchResult

class FakeEmbeddings:
    def embed_query(self, q):
        assert q == 'What is expense ratio?'
        return [0.1, 0.2]

class FakeStore:
    def query(self, vector, top_k, where=None):
        assert vector == [0.1, 0.2] and top_k == 3
        return [
            SearchResult('a', 'Expense ratio is the annual fee.', {'title':'Fund page','url':'https://groww.in/x'}, .2),
            SearchResult('b', 'Expense ratio is the annual fee.', {'title':'Fund page','url':'https://groww.in/x'}, .3),
            SearchResult('c', 'Unrelated content.', {'title':'Other','url':'https://groww.in/y'}, .8),
        ]

def test_retrieval_filters_deduplicates_and_assembles_sources():
    r = Retriever(FakeStore(), FakeEmbeddings(), top_k=3, relevance_threshold=.45)
    result = r.retrieve('What is expense ratio?')
    assert len(result.chunks) == 1
    assert result.chunks[0].distance == .2
    assert result.chunks[0].relevance == .8
    assert 'Fund page' in result.context and 'https://groww.in/x' in result.context
    assert len(result.sources) == 1 and result.no_evidence is False

def test_no_evidence_when_all_distances_exceed_threshold():
    class EmptyEvidenceStore:
        def query(self, *args, **kwargs):
            return [SearchResult('x','text',{},.9)]
    result = Retriever(EmptyEvidenceStore(), FakeEmbeddings(), relevance_threshold=.45).retrieve('What is expense ratio?')
    assert result.no_evidence and result.context == '' and result.sources == []

def test_blank_query_rejected():
    import pytest
    with pytest.raises(ValueError): Retriever(FakeStore(), FakeEmbeddings()).retrieve('  ')

def test_context_length_cap():
    class LongStore:
        def query(self,*args,**kwargs): return [SearchResult('x','A'*500,{'title':'T'},.1)]
    result=Retriever(LongStore(),FakeEmbeddings(),max_context_chars=100).retrieve('What is expense ratio?')
    assert len(result.context) <= 100
