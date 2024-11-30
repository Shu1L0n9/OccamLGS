# Occam's LGS: A Simple Approach for Language Gaussian Splatting

[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)]()
[![Project Page](https://img.shields.io/badge/Project-Page-blue)]()

This is the official implementation of "Occam's LGS: A simple approach for Language Gaussian Splatting".

## Overview

Occam's LGS is a simple, training-free approach for Language-guided 3D Gaussian Splatting that achieves state-of-the-art results with a 100x speed improvement. Our method:

- 🎯 Lifts 2D language features to 3D Gaussian Splats without complex modules or training
- 🚀 Provides 100x faster optimization compared to existing methods  
- 🧩 Works with any feature dimension without compression
- 🎨 Enables easy scene manipulation and object insertion

## Key Features

- Training-free global optimization approach
- Direct reasoning in language feature space
- Support for arbitrary language feature dimensionality  
- Fast processing time (~15s runtime)
- Compatible with SAM+CLIP features
- Includes tools for object insertion and scene manipulation