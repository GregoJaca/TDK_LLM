EXTRA_GREGO

linear techniques dim reduction: NMF , FACTOR ANALYSIS, pLSA

Use random, structure-oblivious projections rather than learned ones:
Johnson-Lindenstrauss (dense Gaussian RP)

is the shift properly implemented


Add  slice or max-sliced Wasserstein-1 
Use the POT library (Python Optimal Transport). import as ot
Inside each sliding window (length w) collect the hidden-state vectors for the two trajectories.
Draw k random unit vectors (k ≈ 100–200) with numpy.random.normal → normalise.
Project all vectors onto every random direction → two 1-D samples.
Call ot.wasserstein_1d on each 1-D pair → k scalar Wasserstein distances.
Average (slice) or take the maximum (max-slice) over these k values → scalar window distance.


Procrustes distance between entire trajectories after optimal rotation+scaling alignment.


hausdorff currently
d_ab = directed_hausdorff(a, b)[0]
d_ba = directed_hausdorff(b, a)[0]
symmetric_d = max(d_ab, d_ba)




validate which metrics REQUIRE sliding window



currently initial conditions are not always the same distance apart from each other, because of random direction of disturbance.