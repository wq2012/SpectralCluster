"""Example configurations."""

from spectralcluster import refinement
from spectralcluster import spectral_clusterer

RefinementName = refinement.RefinementName
RefinementOptions = refinement.RefinementOptions
ThresholdType = refinement.ThresholdType
SymmetrizeType = refinement.SymmetrizeType
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
