"""
Package information and creates the logger
"""
import logging


__version__ = '0.0.2'
__author__ = 'Ethan Tregidga'
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

try:
    from sciplots.clustering import PlotClusters
    from sciplots.grids import PlotConfusion, PlotPearson
    from sciplots.misc import PlotDistributions, PlotImages, PlotSaliency
    from sciplots.pair_plots import PlotParamPairs, PlotParamPairComparison
    from sciplots.single_plots import PlotComparison, PlotPerformance, PlotPlots, PlotDistribution
except ModuleNotFoundError:
    pass
except ImportError:
    pass
