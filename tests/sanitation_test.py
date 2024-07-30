import pytest

from routes.helpers.submodules.sanitation import sanitize_input

def test_sanitation():
    assert sanitize_input("<script>alert('Hello World')</script>") == "&lt;script&gt;alert(&#x27;Hello World&#x27;)&lt;/script&gt;"