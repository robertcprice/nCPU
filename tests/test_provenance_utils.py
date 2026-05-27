from pathlib import Path

from ncpu.utils.provenance import collect_provenance, file_record


ROOT = Path(__file__).resolve().parents[1]


def test_collect_provenance_has_core_sections():
    data = collect_provenance(ROOT, argv=["benchmarks/ablation_study.py", "--trials", "1"])

    assert "timestamp_utc" in data
    assert data["argv"] == ["benchmarks/ablation_study.py", "--trials", "1"]
    assert "python" in data
    assert "platform" in data
    assert "ncpu_git" in data
    assert "mog" in data


def test_file_record_captures_hash_and_size(tmp_path: Path):
    target = tmp_path / "sample.txt"
    target.write_text("ncpu\n")

    record = file_record(target, root=tmp_path)

    assert record["path"] == "sample.txt"
    assert record["size_bytes"] == len("ncpu\n")
    assert len(record["sha256"]) == 64
