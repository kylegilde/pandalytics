import IPython
from IPython.display import HTML


def create_code_toggle(button_name: str = "Toggle Code") -> IPython.core.display.HTML:
    """
    Hides your code and creates a Toggle button at the top and bottom of the notebook
    # source: https://stackoverflow.com/a/28073228/4463701

    Parameters
    ----------
    button_name: Name of your button

    Returns
    -------
    IPython.core.display.HTML object

    The following html works but not as well.
    %%html
    <style id=hide>div.input{display:none;}</style>
    <button type="button"
    onclick="var myStyle = document.getElementById('hide').sheet;myStyle.insertRule('div.input{display:inherit !important;}', 0);">
    Show inputs</button>
    """
    return HTML(
        """
        <script>
            code_show = true;
            function code_toggle() {
                if (code_show) {
                    $('div.input').hide();
                } else {
                    $('div.input').show();
                }
                code_show = !code_show
            }
            $(document).ready(code_toggle); 
        </script>
        """
        f"""
        <form action="javascript:code_toggle()">
            <input type="submit" value="{button_name}">
        </form>
        """
    )
