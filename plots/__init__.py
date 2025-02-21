"""
Package information and creates the logger
"""
import logging


__version__ = '0.0.1'
__author__ = 'Ethan Tregidga'
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

try:
    from plots.clustering import PlotClusters
    from plots.grids import PlotConfusion, PlotPearson
    from plots.misc import PlotDistributions, PlotImages, PlotSaliency
    from plots.pair_plots import PlotParamPairs, PlotParamPairComparison
    from plots.single_plots import PlotComparison, PlotPerformance, PlotPlots, PlotDistribution
except ModuleNotFoundError:
    pass
except ImportError:
    pass
