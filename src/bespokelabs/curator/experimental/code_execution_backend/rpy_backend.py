# Res


class RPyCodeExecutor(CodeExecutor):
    def __init__(self, config: RPyCodeExecutorConfig):
        try:
            from RestrictedPython import compile_restricted, safe_globals

        except ImportError:
            raise ImportError("RestrictedPython is not installed. Please install it using `pip install restrictedpython`.")

    def execute(self, code: str, input_str: str) -> str:
        byte_code = compile_restricted(code, "<inline>", "exec")
        loc = {}
        globs = safe_globals()
        exec(byte_code, safe_globals, loc)
        return loc["stdout"]
