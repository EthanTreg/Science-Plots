"""
Package information and creates the logger
"""
import logging


__version__ = '1.0.12'
__author__ = 'Ethan Tregidga'
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

try:
    from sciplots.clustering import PlotClusters
    from sciplots.misc import PlotImages, PlotSaliency
    from sciplots.grids import PlotConfusion, PlotPearson
    from sciplots.distributions import PlotDistribution, PlotDistributions
    from sciplots.pair_plots import PlotParamPairs, PlotParamPairComparison
    from sciplots.single_plots import PlotComparison, PlotPerformance, PlotPlots
except ModuleNotFoundError:
    pass
except ImportError:
    pass
