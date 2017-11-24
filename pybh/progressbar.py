import sys
import tqdm

# Adapted from tensorpack


def get_tqdm_kwargs(**kwargs):
    """
    Return default arguments to be used with tqdm.
    Args:
        kwargs: extra arguments to be used.
    Returns:
        dict:
    """
    default = dict(
        smoothing=0.5,
        dynamic_ncols=True,
        ascii=True,
        bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_noinv_fmt}]'
    )

    f = kwargs.get('file', sys.stderr)
    isatty = f.isatty()
    # Jupyter notebook should be recognized as tty.
    # Wait for https://github.com/ipython/ipykernel/issues/268
    try:
        from ipykernel import iostream
        if isinstance(f, iostream.OutStream):
            isatty = True
    except ImportError:
        pass

    if isatty:
        default['miniters'] = 1
    else:
        # If not a tty, don't refresh progress bar that often
        default['miniters'] = 10
    default.update(kwargs)
    return default


def get_progressbar(**kwargs):
    """ Similar to :func:`get_tqdm_kwargs`,
    but returns the tqdm object directly. """
    tqdm_kwargs = get_tqdm_kwargs(**kwargs)
    return tqdm.tqdm(**tqdm_kwargs)


def progress_range(*args, **kwargs):
    #tqdm_kwargs = get_tqdm_kwargs(**kwargs)
    tqdm_kwargs = kwargs
    return tqdm.trange(*args, **tqdm_kwargs)

