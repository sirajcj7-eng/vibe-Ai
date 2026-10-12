import pytest

from vibe.licenses import (
    LICENSE_FIELD, SOURCE_LEVEL_LICENSE, canonical_license, doc_license_status,
)


@pytest.mark.parametrize("value,expected", [
    ("Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/", "cc-by"),
    ("Creative Commons - Attribution - https://creativecommons.org/licenses/by/3.0/", "cc-by"),
    ("Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/4.0/", "cc-by-sa"),
    ("Creative Commons Zero - Public Domain - https://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
    ("Public Domain", "public-domain"),
    ("CCBY", "cc-by"), ("CCBYSA", "cc-by-sa"), ("CC0", "cc0"),
    ("MIT", "mit"), ("mit", "mit"),
    ("Apache-2.0", "apache-2.0"), ("apache-2.0", "apache-2.0"),
    ("BSD-3-Clause", "bsd-3-clause"), ("bsd-2-clause", "bsd-2-clause"),
    ("Unlicense", "unlicense"), ("CC-BY-4.0", "cc-by"),
    ("Creative Commons - Attribution Non-Commercial No Derivatives", "restricted"),
    ("arXiv.org - Non-exclusive license to distribute - http://arxiv.org/licenses/none", "restricted"),
    ("GPL-3.0", "restricted"), ("AGPL-3.0", "restricted"),
    ("None", "missing"), (None, "missing"),
    ("Some Custom License", "unknown:some custom license"),
])
def test_canonical_license(value, expected):
    assert canonical_license(value) == expected


def test_doc_license_status_multi_valued():
    assert doc_license_status({"license": "MIT,Apache-2.0"}) == ("apache-2.0+mit", True)
    assert doc_license_status({"license": "['MIT']"}) == ("mit", True)
    assert doc_license_status({"license": "MIT,GPL-3.0"})[1] is False


def test_doc_license_status_uses_the_given_field():
    assert doc_license_status({"oa_license": "CCBY"}, "oa_license") == ("cc-by", True)
    meta = {"license": "Creative Commons Zero - Public Domain - https://creativecommons.org/publicdomain/zero/1.0/",
            "full_text_license": "Creative Commons - Attribution Non-Commercial No Derivatives"}
    assert doc_license_status(meta, "license")[1] is True  # abstract text is CC0


def test_missing_and_source_level():
    assert doc_license_status({}, "license") == ("missing", False)
    assert doc_license_status(None, "license") == ("missing", False)
    assert doc_license_status({}, "license", source_level=True) == ("source-verified", True)


def test_registry_constants():
    assert LICENSE_FIELD["pes2o"] == "oa_license"
    assert "cccc" in SOURCE_LEVEL_LICENSE