import IPython
from IPython.display import HTML, display, Javascript


def process_completed(msg: str | None = "Done"):
    """
    Plays an audio message.
    Put this function at the end of long-calculating notebook cells.
    Source: https://stackoverflow.com/a/74525871/27596895
    """
    js = f"""
      var msg = new SpeechSynthesisUtterance();
      msg.text = "{msg}";
      window.speechSynthesis.speak(msg);
    """
    
    display(Javascript(js))


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
