from functools import partial
import IPython
from IPython.display import HTML, display, Javascript


def play_message(msg: str):
    """
    Plays an audio message.
    Source: https://stackoverflow.com/a/74525871/27596895
    """
    js = f"""
      var msg = new SpeechSynthesisUtterance();
      msg.text = "{msg}";
      window.speechSynthesis.speak(msg);
    """
    
    display(Javascript(js))


def process_done(msg: str | None = "Done"):
    """
    Plays an audio message, Done by default
    Put this function at the end of long-calculating notebook cells.
    Source: https://stackoverflow.com/a/74525871/27596895
    """
    play_message(msg)
    
    
def _play_exception_sound(
    self, 
    etype,
    value,
    tb, 
    tb_offset=None, 
    full_exception: bool | None = False
):
    """
    Helper function to play exception messages
    source: https://stackoverflow.com/a/41603739/27596895
    """
    self.showtraceback((etype, value, tb), tb_offset=tb_offset)
    exception_message = etype.__name__
    if full_exception:
        # remove punctuation that JavaScript cannot handle
        exception_message += str(value).replace("'", "").replace('"', "")
    play_message(exception_message)
    return


def play_exception_sounds(full_exception: bool | None = False):
    """
    Call this function in a notebook to play exception messages
    """
    
    msg_fn = partial(_play_exception_sound, full_exception=full_exception)
    get_ipython().set_custom_exc((Exception,), msg_fn)
    return


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
