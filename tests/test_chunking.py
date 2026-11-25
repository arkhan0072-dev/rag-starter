from app.chunking import by_structure, fixed_size

DOC = """# Handbook

Intro paragraph.

## Leave

Employees get 25 days of paid leave.

Unused leave expires in March.

## Equipment

### Laptops

Laptops are replaced every three years.
"""


def test_structure_keeps_heading_path():
    chunks = by_structure(DOC)
    assert [c.heading for c in chunks] == [
        "Handbook",
        "Handbook > Leave",
        "Handbook > Equipment > Laptops",
    ]


def test_structure_packs_paragraphs_under_one_heading():
    leave = by_structure(DOC)[1]
    assert leave.content == "Employees get 25 days of paid leave.\n\nUnused leave expires in March."


def test_structure_starts_new_chunk_when_full_and_never_splits_a_paragraph():
    para = "word " * 50  # ~250 chars
    chunks = by_structure(f"# A\n\n{para}\n\n{para}\n\n{para}", max_chars=600)
    assert len(chunks) == 2
    assert all(c.content.count("word") % 50 == 0 for c in chunks)


def test_heading_level_resets_deeper_levels():
    chunks = by_structure("# A\n\n## B\n\n### C\n\ntext c\n\n## D\n\ntext d")
    assert [c.heading for c in chunks] == ["A > B > C", "A > D"]


def test_fixed_size_overlaps():
    chunks = fixed_size("abcdefghij" * 10, size=40, overlap=10)
    assert all(len(c.content) <= 40 for c in chunks)
    assert chunks[0].content[-10:] == chunks[1].content[:10]
