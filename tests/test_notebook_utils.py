from pandalytics.notebook_utils import create_code_toggle


def test_create_code_toggle():
    assert (
        str(create_code_toggle()) == "<IPython.core.display.HTML object>"
    ), "create_code_toggle did NOT return an <class 'IPython.core.display.HTML'> object"
