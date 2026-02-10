class Logger:

    def __init__(self, verbose: bool):
        self.verbose = verbose

    def logger(self, message: str):
        if self.verbose:
            print(f"[DEBUG] {message}")
