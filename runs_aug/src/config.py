import torch

class Default:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
    MAX_LENGTH = 3096
    REPETITION_PENALTY = 1.1
    
    # For launch_pentek.py
    N_INITIAL_CONDITIONS = 6
    RESULTS_DIR = "./launch_sep_area_preserving"
    SELECTED_LAYERS = list(range(0,25))

class Experiment:
    # RADII = [0.0003, 0.0004]
    # TEMPS = [0, 0.6]
    # TOP_PS = [1, 0.95]
    # TOP_KS = [1, 50]
    RADII = [0.00035]
    TEMPS = [0]
    TOP_PS = [1]
    TOP_KS = [1]

class Analysis:
    SAVE_PLOTS = True
    PAIRS_TO_PLOT =  [[i,j] for i in range(0,Default.N_INITIAL_CONDITIONS) for j in range(i+1, Default.N_INITIAL_CONDITIONS)] # [[0, 1]]
    SLIDING_WINDOW_SIZE = 16
    SLIDING_WINDOW_DISPLACEMENT = 16
    MINIMUM_VARIANCE_EXPLANATION = 0.9

    DEVIATION_METRIC = "rms" # 'mad' or 'rms'
    PLOT_HYPER_AND_AXIS = True
    RUN_LOCAL_DIMENSIONALITY = False
    RUN_RANK_EIGENVECTORS = False
    # Layer-volume analysis options
    DO_LAYER_VOLUME_ANALYSIS = True
    # methods: list containing any of 'pca_svd' (product of PCA singular values) or 'logdet_cov' (sqrt(det(cov)))
    LAYER_VOLUME_METHOD = ['pca_svd', 'logdet_cov']
    # If True, normalize each hidden-state vector to unit L2 norm before computing volumes
    NORMALIZE_HIDDEN_STATES = True # GG
    # Number of principal axes to include in the volume computation. None = use all
    LAYER_VOLUME_N_AXES = 14 # GG
    # Explained variance threshold used when LAYER_VOLUME_N_AXES is None (0-1)
    LAYER_VOLUME_EXPLAINED_VAR = 0.9 # GG
    # Output filename for saved per-trajectory per-layer volumes
    LAYER_VOLUME_OUTPUT_FILENAME = 'layer_volumes.pt'

# Model Configurations
class ModelConfig:
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    LOCAL_DIR = "." # "deepseek_r1_14b"

    # MODEL_NAME = "arcee-ai/virtuoso-lite" 
    # LOCAL_DIR = "./virtuoso-lite-10b" 

    # MODEL_NAME = "Noorhan/mistral-7b-4bit"
    # LOCAL_DIR = "./mistral-7b"  # local path to save the model

    # MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" 
    # LOCAL_DIR = "deepseek_r1_14b"   # local path to save the model




# Prompts
class Prompts:
    prompts = [
        # Space Technology (Detailed Technical Review)
        # "Provide a comprehensive technical review of current and proposed propulsion systems for interstellar travel. Compare chemical rockets, nuclear propulsion, laser sails, antimatter drives, and other theoretical concepts in terms of energy requirements, achievable speeds, technological feasibility, and projected timelines for development. Include discussion of major projects in the history of the field."

        # Psychology (Exploratory Tone)
        "Examine how childhood experiences shape personality development. Discuss various influences including family environment, education, friendships, and significant life events. Explain psychological concepts like attachment theory and nature vs. nurture in accessible terms. Provide examples of how positive and negative experiences can affect adult personality traits and behaviors."
    ]
   
    prompt_names = [
        # "interstellar_propulsion_review",       # Space Technology (Detailed Technical Review)
        "childhood_personality_development",    # Psychology (Exploratory Tone)
    ]