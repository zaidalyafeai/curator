import asyncio
from time import sleep


def run_in_event_loop(coroutine):
    """
    Run a coroutine in the current event loop or create a new one if there isn't one.
    """
    try:
        loop = asyncio.get_running_loop()
        future = loop.create_task(coroutine)
        while not future.done():
            sleep(1)
        return future.result()
    except RuntimeError as e:
        # If no event loop is running, asyncio will
        # return a RuntimeError (https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.get_running_loop).
        # In that case, we can just use asyncio.run.
        return asyncio.run(coroutine)

