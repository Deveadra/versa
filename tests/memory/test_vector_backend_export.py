import pytest

from base.memory import vector_backend as vb


def test_qdrant_backend_export_is_defined():
    # This test mainly ensures QdrantMemoryBackend is defined at module scope.
    assert hasattr(vb, "QdrantMemoryBackend")


def test_qdrant_backend_behavior_without_dependency():
    if vb.HAVE_QDRANT:
        # We do NOT instantiate (would require a live Qdrant service).
        # We only assert that the export points at the implementation class.
        assert vb.QdrantMemoryBackend is vb._QdrantMemoryBackendImpl
    else:
        with pytest.raises(ImportError):
            vb.QdrantMemoryBackend(embedder=None, dim=384)  # type: ignore[arg-type]
