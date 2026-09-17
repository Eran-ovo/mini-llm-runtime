from mini_llm_runtime.environment import collect_environment


def test_environment_contains_reproducibility_fields() -> None:
    result = collect_environment()
    assert result["python"]["version"]
    assert result["torch"]["version"]
    assert "cuda_available" in result["torch"]
    assert "nvcc" in result["tools"]
    assert "transformers" in result["packages"]
