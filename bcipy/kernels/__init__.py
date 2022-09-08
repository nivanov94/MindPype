#from .cdf import CDFKernel
#from .concatenation import ConcatenationKernel
#from .covariance import CovarianceKernel
#from .csp import CommonSpatialPatternKernel
#from .enqueue import EnqueueKernel
#from .extract import ExtractKernel
#from .feature_normalization import FeatureNormalizationKernel
#from .filter_ import FilterKernel
#from .filtfilt import FiltFiltKernel
#from .max_ import MaxKernel
#from .mean import MeanKernel
#from .min_ import MinKernel
#from .reduced_sum import ReducedSumKernel
#from .resample import ResampleKernel
#from .riemann_mdm_classifier_kernel import RiemannMDMClassifierKernel
#from .riemann_distance import RiemannDistanceKernel
#from .riemann_mean import RiemannMeanKernel
#from .riemann_potato import RiemannPotatoKernel
#from .set_data import SetKernel
#from .stack import StackKernel
#from .std import StdKernel
#from .svm import SVMClassifierKernel
#from .tensor_stack import TensorStackKernel
#from .threshold import ThresholdKernel
#from .transpose import TransposeKernel
#from .var import VarKernel
#from .zscore import ZScoreKernel

from .arithmetic import (AbsoluteKernel, LogKernel, AdditionKernel, 
                         DivisionKernel, MultiplicationKernel, SubtractionKernel)
from .logical import (NotKernel, AndKernel, OrKernel, XorKernel, 
                      EqualKernel, LessKernel, GreaterKernel)
