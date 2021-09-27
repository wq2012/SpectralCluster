"""Example configurations."""

from spectralcluster import autotune
from spectralcluster import constraint
from spectralcluster import laplacian
from spectralcluster import refinement
from spectralcluster import spectral_clusterer

AutoTune = autotune.AutoTune
ConstraintName = constraint.ConstraintName
ConstraintOptions = constraint.ConstraintOptions
RefinementName = refinement.RefinementName
RefinementOptions = refinement.RefinementOptions
ThresholdType = refinement.ThresholdType
SymmetrizeType = refinement.SymmetrizeType
LaplacianType = laplacian.LaplacianType
SpectralClusterer = spectral_clusterer.SpectralClusterer

# Configurations that are closest to the ICASSP2018 paper
# "Speaker Diarization with LSTM".
ICASSP2018_REFINEMENT_SEQUENCE = [
    RefinementName.CropDiagonal,
    RefinementName.GaussianBlur,
    RefinementName.RowWiseThreshold,
    RefinementName.Symmetrize,
    RefinementName.Diffuse,
    RefinementName.RowWiseNormalize,
]

icassp2018_refinement_options = RefinementOptions(
    gaussian_blur_sigma=1,
    p_percentile=0.95,
    thresholding_soft_multiplier=0.01,
    thresholding_type=ThresholdType.RowMax,
    refinement_sequence=ICASSP2018_REFINEMENT_SEQUENCE)

icassp2018_clusterer = SpectralClusterer(
    min_clusters=2,
    max_clusters=7,
    autotune=None,
    laplacian_type=None,
    refinement_options=icassp2018_refinement_options,
    custom_dist="cosine")

# Configurations of Turn-to-Diarize system using the
# Turn + Constraint Propagation + AutoTune method described in the paper
# "Turn-to-Diarize: Online Speaker Diarization Constrained by
# Transformer Transducer Speaker Turn Detection".
TURNTODIARIZE_REFINEMENT_SEQUENCE = [
    RefinementName.RowWiseThreshold, RefinementName.Symmetrize
]

turntodiarize_refinement_options = RefinementOptions(
    thresholding_soft_multiplier=0.01,
    thresholding_type=ThresholdType.Percentile,
    thresholding_with_binarization=True,
    thresholding_preserve_diagonal=True,
    symmetrize_type=SymmetrizeType.Average,
    refinement_sequence=TURNTODIARIZE_REFINEMENT_SEQUENCE)

turntodiarize_constraint_options = ConstraintOptions(
    constraint_name=ConstraintName.ConstraintPropagation,
    apply_before_refinement=True,
    constraint_propagation_alpha=0.4)

turntodiarize_auto_tune = AutoTune(
    p_percentile_min=0.40,
    p_percentile_max=0.95,
    init_search_step=0.05,
    search_level=1)

turntodiarize_clusterer = SpectralClusterer(
    min_clusters=2,
    max_clusters=7,
    refinement_options=turntodiarize_refinement_options,
    constraint_options=turntodiarize_constraint_options,
    autotune=turntodiarize_auto_tune,
    laplacian_type=LaplacianType.GraphCut,
    row_wise_renorm=True,
    custom_dist="cosine")
