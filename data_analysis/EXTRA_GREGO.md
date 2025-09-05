EXTRA_GREGO

linear techniques dim reduction: NMF , FACTOR ANALYSIS, pLSA

Use random, structure-oblivious projections rather than learned ones:
Johnson-Lindenstrauss (dense Gaussian RP)


 Areas for Improvement:


   * `src/metrics/cos.py` (PARTIAL): The aggregation of shifted cosine distances is currently basic. It could
     be made more configurable to allow for different aggregation strategies (e.g., mean, median, or specific
     per-shift results).
   * `src/metrics/dtw_fast.py` (PARTIAL): The current implementation relies on external libraries (fastdtw or
     tslearn). It should include more robust error handling or a clearer fallback mechanism if these are not
     installed, rather than just raising an ImportError.
   * `src/metrics/frechet.py` (PARTIAL): The pure-Python fallback for Fréchet distance is not optimized and
     can be computationally expensive for large datasets. If the frechetdist library is not used, this
     implementation might need optimization.
   * `src/utils/parallel.py` (PARTIAL) and `src/runner/metrics_runner.py` (PARTIAL): The parallel execution of
      metrics is currently disabled due to issues with passing arguments and handling errors in the map_pairs
     function. Enabling and properly implementing parallel processing would significantly improve performance
     for large numbers of trajectory pairs.
   * `src/runner/lyapunov.py` (PARTIAL): The auto_detect_linear_window heuristic for Lyapunov estimation is
     functional but basic. More sophisticated algorithms could be explored for robust linear region detection
     across diverse data characteristics.

   * `run_sweep.py` (PARTIAL): While the sweep script runs, the integration of sweep parameters (like shift
     for cosine distance) into the metric functions is not fully implemented. The metric functions currently
     use their default CONFIG values rather than the specific sweep parameter for each run.
   * `src/cli.py` (PARTIAL): The individual reduce, metrics, and lyapunov commands in the CLI are currently
     placeholders. They need to be fully implemented to allow users to run these specific pipeline steps
     independently.






Add the option of "none" for the dimension reducing technique. 

is the shift properly implemented


Add  slice or max-sliced Wasserstein-1 
Use the POT library (Python Optimal Transport). import as ot
Inside each sliding window (length w) collect the hidden-state vectors for the two trajectories.
Draw k random unit vectors (k ≈ 100–200) with numpy.random.normal → normalise.
Project all vectors onto every random direction → two 1-D samples.
Call ot.wasserstein_1d on each 1-D pair → k scalar Wasserstein distances.
Average (slice) or take the maximum (max-slice) over these k values → scalar window distance.


Procrustes distance between entire trajectories after optimal rotation+scaling alignment.