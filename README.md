# GetGrainSize

Analysis of metallography images comparing Melography and Clemex predictions to Ground Truth.

## Overview

We processed human-tagged Ground Truth (GT) images into 256x256 squares and analyzed them using Heyn intercept method [E. Heyn, The Metallographist, Vol. 5, 1903, pp. 39-64.]. We then compared predictions from Melography and Clemex models, excluding overlapping sections with the GT used during training, and created 256x256 squares for consistency. Meta-statistical analysis was performed to extract mean, median, and variance for each group to determine the similarity of grain distributions in model predictions to the GT.
