import re
import pandas as pd
from scripts.build_report import markdown_table


def test_markdown_metadata_does_not_create_extra_columns():
    frame = pd.DataFrame({"title": ["A | B"], "genres": ["Comedy|Drama\nRomance"]})
    lines = markdown_table(frame).splitlines()
    assert len(lines) == 3
    assert all(len(re.findall(r"(?<!\\)\|", line)) == 3 for line in lines)
    assert "Comedy\\|Drama<br>Romance" in lines[2]
