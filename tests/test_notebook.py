"""Structural checks on the demo notebook.

The notebook is generated, and a generated cell shipped with a SyntaxError once:
it was written by string substitution, so a mangled escape produced a broken
literal that nothing caught until the notebook was run. These checks are static
and fast; executing the notebook end to end is a manual step.
"""

import ast
import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parent.parent / "examples" / "demo.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    if not NOTEBOOK.exists():
        pytest.skip(f"{NOTEBOOK.name} not present")
    return json.loads(NOTEBOOK.read_text())


def _code_cells(notebook: dict) -> list[tuple[int, str]]:
    return [
        (i, "".join(cell["source"]))
        for i, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
    ]


class TestNotebookIsValid:
    def test_every_code_cell_parses(self, notebook):
        """A cell that cannot be parsed will stop the notebook at that point."""
        for index, source in _code_cells(notebook):
            try:
                ast.parse(source)
            except SyntaxError as exc:
                pytest.fail(f"cell {index} has a syntax error: {exc}")

    def test_has_code_and_prose(self, notebook):
        cells = notebook["cells"]
        assert sum(1 for c in cells if c["cell_type"] == "code") >= 5
        assert sum(1 for c in cells if c["cell_type"] == "markdown") >= 5

    def test_outputs_are_not_committed(self, notebook):
        """Stored outputs would drift out of step with the code that made them."""
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert not cell.get("outputs"), f"cell {index} has stored output"
                assert cell.get("execution_count") is None, f"cell {index} has a run count"


class TestNotebookSetup:
    """The install cell is what a first-time reader depends on most."""

    def test_installs_the_phonological_extras(self, notebook):
        """Rhythmic and Phonemic raise ImportError without them."""
        source = "\n".join(src for _, src in _code_cells(notebook))

        assert "linguistic-diversity[phonological]" in source

    def test_checks_extras_independently_of_the_package(self, notebook):
        """Guarding the extras behind `is linguistic_diversity installed?` skips them
        for anyone who already has the package, which is the common case on a re-run.
        """
        source = "\n".join(src for _, src in _code_cells(notebook))

        for module in ("pyphen", "pronouncing", "g2p_en"):
            assert module in source, f"{module} is never checked for"

    def test_downloads_the_spacy_pipeline_and_nltk_corpora(self, notebook):
        source = "\n".join(src for _, src in _code_cells(notebook))

        assert "en_core_web_sm" in source
        assert "nltk.downloader" in source
        # The modern tagger name; the pre-rename one silently yields empty phonemes
        assert "averaged_perceptron_tagger_eng" in source
