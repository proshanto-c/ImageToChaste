import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import cv2
import sys
import os
import random
import json
from PIL import Image

from sam.sam2.build_sam import build_sam2
from sam.sam2.sam2_image_predictor import SAM2ImagePredictor
from sam.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from functions.mask_displays import show_anns
from functions.find_centroids import find_centroid
from functions.load_params import load_generator_from_run
from functions.mask_dilation import dilate_masks, get_shared_edge_map, get_shared_edges_and_adjacency, extract_shared_edges_from_disjoint_masks, extract_shared_edge_segments, extract_shared_edge_segments_from_disjoint_masks, extract_expanded_edge_segments_from_disjoint_masks, clean_and_connect_segments
from functions.mask_gen import mask_gen
from functions.set_device import get_device