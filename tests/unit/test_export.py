from candidate.pipeline import write_submission


def test_export_has_fixed_cross_platform_line_endings(tmp_path):
    output = write_submission(["H", "D"], tmp_path / "submission.csv")
    assert output.read_bytes() == b"frame,label\r\n0,H\r\n1,D\r\n"
