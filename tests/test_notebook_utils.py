from pandalytics.notebook_utils import create_code_toggle
import IPython


def test_create_code_toggle():
    assert isinstance(
        create_code_toggle(), IPython.core.display.HTML
    ), "create_code_toggle did NOT return a <class 'IPython.core.display.HTML'> object"
