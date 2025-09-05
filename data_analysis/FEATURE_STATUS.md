# Feature Status

## Phase 0: Repo & minimal foundation

- **Repo Skeleton**: DONE
- **config.py**: DONE (initial version)
- **src/utils/logging.py**: DONE
- **src/io/loader.py**: DONE
- **src/__main__.py**: DONE
- **src/cli.py**: PARTIAL (stubs for some commands)
- **run_all.py**: PARTIAL (stubs implemented)
- **requirements.txt**: DONE

## Phase 1: I/O, preprocessing, and reduction scaffolding

- **src/io/saver.py**: DONE
- **src/preproc/normalization.py**: DONE
- **src/preproc/whitening.py**: DONE
- **src/reduce/base.py**: DONE
- **src/reduce/pca.py**: DONE (explained variance stored)
- **src/reduce/whitening_wrapper.py**: DONE
- **src/reduce/autoencoder_stub.py**: PARTIAL (stub implemented)
- **examples/small_example.pt**: DONE

## Phase 2: Metrics module

- **src/metrics/base.py**: DONE
- **src/metrics/cos.py**: PARTIAL (aggregation strategy is basic)
- **src/metrics/dtw_fast.py**: PARTIAL (requires user to install `fastdtw` or `tslearn`)
- **src/metrics/hausdorff.py**: DONE
- **src/metrics/frechet.py**: PARTIAL (fallback implementation is not optimized)
- **examples/example_data_generator.py**: PARTIAL (basic checks implemented)

## Phase 3: Pairing logic, runner, and parallel execution

- **src/runner/pair_manager.py**: DONE
- **src/utils/parallel.py**: PARTIAL (error handling and argument passing needs improvement)
- **src/runner/metrics_runner.py**: PARTIAL (parallel execution is disabled)

## Phase 4: Lyapunov (fast pairwise slope) and diagnostics

- **src/runner/lyapunov.py**: PARTIAL (auto-detection heuristic is basic)
- **src/viz/plots.py**: PARTIAL (stubs for some plots remain)

## Phase 5: Visualization utilities

- **src/viz/plots.py**: PARTIAL (some plots are placeholders)

## Phase 6: Hyperparameter sweep script

- **run_sweep.py**: PARTIAL (basic implementation, limited parameter integration)

## Phase 7: CLI, examples, and documentation

- **src/cli.py**: PARTIAL (stubs for some commands)
- **src/__main__.py**: DONE
- **scripts/example_run.bat**: DONE
- **README.md**: DONE

## Phase 7: CLI, examples, and documentation

- **src/cli.py**: PARTIAL (stubs for some commands)
- **src/__main__.py**: DONE
- **scripts/example_run.bat**: DONE
- **README.md**: DONE
