import json
import sys
from zipfile import ZipFile
import pytest
from scripts import prepare_data


def test_download_respects_custom_raw_dir_and_retains_checksum_on_rerun(tmp_path, monkeypatch):
    cached = tmp_path / "data/raw/ml-1m.zip"
    cached.parent.mkdir(parents=True)
    with ZipFile(cached, "w") as archive:
        archive.writestr("ml-1m/users.dat", "1::F::25::1::12345\n")
        archive.writestr("ml-1m/movies.dat", "1::Movie (2000)::Drama\n")
        archive.writestr("ml-1m/ratings.dat", "1::1::5::100\n")
        archive.writestr("ml-1m/README", "Dataset attribution")
        archive.writestr("../outside.txt", "not a dataset file")
    raw = tmp_path / "custom"
    processed = tmp_path / "processed"
    monkeypatch.setattr(prepare_data, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(prepare_data, "make_eda_figures", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--raw-dir", str(raw), "--processed-dir", str(processed), "--figures-dir", str(tmp_path / "figures"), "--download"])
    for _ in range(2):
        prepare_data.main()
        summary = json.loads((processed / "dataset_summary.json").read_text())
        assert summary["archive_sha256"] == prepare_data.sha256(cached)
    assert (raw / "README").read_text() == "Dataset attribution"
    assert (raw / "ratings.dat").is_file()
    assert not (tmp_path / "outside.txt").exists()


def test_incomplete_archive_fails_before_writing_data(tmp_path):
    archive = tmp_path / "incomplete.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("ml-1m/users.dat", "only one file")
    with pytest.raises(ValueError, match="missing required"):
        prepare_data.extract_movielens_archive(archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()
