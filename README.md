# COMBINE
COMBINE - COst Minimization Bed INvErsion model for ice caps and valley glaciers

This project is an adaption/extension to [OGGM](https://github.com/OGGM/oggm) and utilizes its dynamical model together with backwards functionalities (Automatic/Algorithmic Differentiation) of [PyTorch](https://pytorch.org/) to enable a cost function based inversion of bedrock topography.

COMBINE2D is based on a dynamical 2D Shallow-Ice-Approximation model, using surface outlines, surface topography, surface mass-balance time series and optionally also existing ice thickness measurements for ice caps. For further information look at the [master thesis](https://diglib.uibk.ac.at/ulbtirolhs/content/titleinfo/3086935/full.pdf) of @phigre.

COMBINE1D is based on a dynamical 1D or flowline model, using flowline surface heights and widths as well as surface mass-balance time series for valley glaciers. For further information look at the [master thesis](https://diglib.uibk.ac.at/ulbtirolhs/content/titleinfo/6139027/full.pdf) of @pat-schmitt.
