def logger(message: str, verbose: bool):
    if verbose:
        print(f"[DEBUG] {message}")