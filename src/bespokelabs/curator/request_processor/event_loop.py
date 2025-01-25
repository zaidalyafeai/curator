import asyncio

import nest_asyncio

from concurrent.futures import ProcessPoolExecutor


def run_in_event_loop(coroutine):
    """Run a coroutine in the current event loop or create a new one if there isn't one."""
    try:
        # This call will raise an RuntimError if there is no event loop running.
        asyncio.get_running_loop()

        # If there is an event loop running (the call
        # above doesn't raise an exception), we can
        # use nest_asyncio to patch the event loop.
        nest_asyncio.apply()

        return asyncio.run(coroutine)
    except RuntimeError:
        # Explicitly pass, since we want to fallback to asyncio.run
        pass

    # If no event loop is running, asyncio will
    # return a RuntimeError (https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.get_running_loop).
    # In that case, we can just use asyncio.run.
    return asyncio.run(coroutine)


def run_in_executor(func, *args, **kwargs):
    """Run a synchronous function in a process pool executor.
    
    This function is useful for running CPU-bound operations in parallel
    on a different CPU core.
    
    Args:
        func: The synchronous function to execute
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function execution
    """
    async def _run_in_executor():
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor() as executor:
            return await loop.run_in_executor(executor, func, *args)
    
    return run_in_event_loop(_run_in_executor())


