from pathlib import Path
import socket
from unittest.mock import patch
import pytest
from aegis.core import Passage


def test_local_reader_offsets_and_no_network():
    if not Path("models/reader").exists():
        pytest.skip("Staged reader required")
    from aegis.reader import Reader

    passage = Passage("p", "doc", "Fixture", "The lighthouse is painted cobalt blue.")
    with patch.object(
        socket.socket, "connect", side_effect=AssertionError("Network forbidden")
    ):
        reader = Reader()
        result = reader.answer("What color is the lighthouse?", [passage])
    assert not result["abstained"]
    citation = result["citations"][0]
    span = passage.text[citation["char_start"] : citation["char_end"]]
    assert span == "cobalt blue"
    assert result["answer"] == span + " [S1]"
