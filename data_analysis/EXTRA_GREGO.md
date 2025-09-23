EXTRA_GREGO

<>
ALIGN DISTANCE CURVES BEFORE LYAPUNOV: 
The distance curves should be shifted horizontally before averaging.
There should be an option in config that allows me to manually select which pairs to exclude from the analysis.
There should be an outlier detection algorithm. I will list several outlier detection strategies:
- if the distance is always very small (1e-15 aprox) (two almost identical trajectories), then discard that pair. GG THRESHOLD
- if the divergence part starts very late, then (i think this goes away naturally with threshold (see previous))

add a bool option which, if True, plots all the distance curves used (after aligning) together in a single plot.
also another bool which plots them in separate plots individually, named with the pair used.
all plots should be saved appropriately in a results_lyap folder
another bool should determine whether the discarded pairs (both by me selecting them manually and by the outlier algo) should be stored in a .json file which lists them (separately for the manually discarded vs the outlier detected) (in the same file but separated, use a dict)


G: if using embedded with a sliding window of 8, then we can use the 1st derivative to identify the saturation region. although, dont cut at the first not increasing datapoint, also check the next 2 for example and do it based on that.
G: When averaging, because the curves (after trimming in order to align) will be of different sizes, I need to do the averaging properly and column wise, dividing by the number of datapoints properly. easy

G: XXX {hausdorff; frechet; dtw}: for (almost) all j distance_traj_i_traj_j when i is {0;13} looks identical with some differences in the saturation part, but the index where divergence starts and everything is so close.
(wasserstein also but it is much noisier. however (in embed) the divergence region is quite similar)
<>

THRESHOLD: also check if currently before RP the distance is normalized. colorbar is 0-1 so be consistent

rn the total distance depends on window wize. improper averaging.
are currently wrong:
- wasserstein
- rank_eigen
are done well:
- dtw
- cos
- hausdorf
- frechet


what the fuck does rank eigen do when window_size = 1


-------------------


linear techniques dim reduction: NMF , FACTOR ANALYSIS, pLSA

Use random, structure-oblivious projections rather than learned ones:
Johnson-Lindenstrauss (dense Gaussian RP)

is the shift properly implemented

sliding window averaging or dividing by window size, check. also at the edges, the window size shrinks so do appropriately. DONE

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