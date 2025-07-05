import os
import streamlit as st
import streamlit.components.v1 as components
from typing import Optional

# Determine whether we are running in dev mode (frontend dev server)
# If the env var CLIPBOARD_COMPONENT_DEV is set, assume dev server is running at localhost:3001
_DEV_SERVER = os.getenv('CLIPBOARD_COMPONENT_DEV') == '1'

# Path to build directory after `npm run build`
_build_dir = os.path.join(os.path.dirname(__file__), "frontend", "build")

if _DEV_SERVER or not os.path.exists(_build_dir):
    _component = components.declare_component(
        "clipboard_image",
        url="http://localhost:3001"  # Assumes you will run `npm start` in the frontend folder
    )
else:
    _component = components.declare_component(
        "clipboard_image",
        path=_build_dir
    )

def clipboard_image(key: Optional[str] = None):
    """Return a base-64 data-URL string of the pasted image, or None.

    Usage::
        img_data = clipboard_image(key="clip")
        if img_data:
            # do something with it
    """
    return _component(key=key, default=None) 