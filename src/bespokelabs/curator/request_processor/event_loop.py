import asyncio
from time import sleep

import nest_asyncio


def run_in_event_loop(coroutine):
    """
    Run a coroutine in the current event loop or create a new one if there isn't one.
    """
    nest_asyncio.apply()

    return asyncio.run(coroutine)

