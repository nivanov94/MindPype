"""
BCIP Model Parser

parse a BCIP model specified in a JSON file and create BCIP sessions
"""

import json

# import BCIP
from ..classes.session import Session
from ..classes.array import Array
from ..classes.block import Block
from ..classes.circle_buffer import CircleBuffer
from ..classes.filter import Filter
from ..classes.node import Node
from ..classes.scalar import Scalar
from ..classes.source import BcipMatFile, LSLStream

# import BCIP kernels
import kernels

