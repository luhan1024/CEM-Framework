import re
from pathlib import Path

def test_user_guide_fenced_python_examples():
    source = Path(__file__).parents[1] / "docs" / "user_guide.md"
    text = source.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\r?\n(.*?)```", text, flags=re.DOTALL)
    assert blocks, "user guide must contain executable Python examples"
    for block in blocks:
        exec(compile(block, str(source), "exec"), {"__name__": "__docs_example__"})
