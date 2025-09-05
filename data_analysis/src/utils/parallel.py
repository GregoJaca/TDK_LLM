# STATUS: PARTIAL
import concurrent.futures
from typing import Callable, List, Tuple
from config import CONFIG
import traceback

def map_pairs(
    func: Callable,
    pairs: List[Tuple[int, int]],
    *args
) -> List:
    """Maps a function over a list of pairs in parallel, with error handling."""
    max_workers = CONFIG["parallel"]["max_workers"]
    use_process = CONFIG["parallel"]["use_multiprocessing"]
    Executor = concurrent.futures.ProcessPoolExecutor if use_process else concurrent.futures.ThreadPoolExecutor

    results = []
    with Executor(max_workers=max_workers) as executor:
        # Submit tasks
        futures = {executor.submit(func, pair, *args): pair for pair in pairs}

        for future in concurrent.futures.as_completed(futures):
            pair = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # TODO: Save partial results and error log
                print(f"Error processing pair {pair}: {e}")
                traceback.print_exc()
                # For now, we just print the error. In a full implementation,
                # we would save partial results and log the error to a file.
                # Re-raising the exception might be desirable depending on the desired behavior.
                # raise e
    
    return results
