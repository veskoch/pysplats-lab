# A Simple Spacetime Gaussian Renderer

> An educational, CPU-based renderer for understanding 4D Spacetime Gaussian Splatting in pure Python

This is a small repository which I put together learning how to implement a renderer of [Gaussian Splats](https://arxiv.org/pdf/2312.16812) – on the CPU, in Python, for simplicity.

My primary goal was to deconstruct the the math from the original paper and understand how Spacetime Gaussians extend the canonical Gaussians and how to render them.



## Getting Started


- [Data_Flow.md](Data_Flow.md) – The primary technical documentation for the project
- [Notebook.ipynb](Notebook.ipynb) – Hands-on entry-point to the repo with executable examples of how to use your rendering library
- [A Gentle Start to 4D Gaussians: Writing a CPU Renderer in Python](https://dev.vesko.ch/gentle-start-to-4d-gaussians/) – A blog post where I shared notes on some of my "gotcha" moments & lessons learned


#### Overview

The codebase is structured to explicitly mirror the stages of a modern GPU rendering pipeline.

```
Splat Data -> (Vertex Stage) -> 2D Ellipse Params -> (Rasterization Stage) -> Pixel Fragments -> (Fragment Stage) -> Final Pixel Color
```

The `src` directory is organized in the following way:
- `splatting.camera` - Camera model for managing view and projection transformations
- `splatting.io` - I/O utilities for loading Gaussian Splatting data from .ply files
- `splatting.primitives` - core data structures for representing scenes and individual splats
- `splatting.render` - rendering functions for visualizing splats, including vectorized and rasterized approaches
- `splatting.toy_data` - manually defined splats for purposes for unit testing the correctness of the code
- `splatting.utils` - helper utilities for data manipulation, like converting scenes to objects




#### What this repo is not

*  This is definitely not a production renderer. It is a learning exercise to understand the *how* and *why* behind Gaussian Splatting.
*   It is not optimized for performance.. The entire pipeline runs on the CPU and is implemented in Python, which is orders of magnitude slower than a proper C++/CUDA implementation. The focus is on algorithmic transpar


## Acknowledgements

Articles & code bases which helped me in the process.


Data:
- [Raw files for the 3D video data by facebookresearch/Neural_3D_Video](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0)


Articles:
- https://www.thomasantony.com/posts/gaussian-splatting-renderer/
- [How to Render a Single Gaussian Splat?](https://shi-yan.github.io/how_to_render_a_single_gaussian_splat/)
- [Kerbl et al., 2023 Reviewed](https://medium.com/@AriaLeeNotAriel/numbynum-3d-gaussian-splatting-for-real-time-radiance-field-rendering-kerbl-et-al-60c0b25e5544)

- [xoft.tistory.com | 4D Gaussian Splatting](https://xoft.tistory.com/54)
- [xoft.tistory.com | Concept Summary 3D Gaussian and 2D projection](https://xoft.tistory.com/49)

Other implementations

-  [GitHub GaussianSplattingViewer/util_gau.py](https://github.com/limacv/GaussianSplattingViewer/blob/main/util_gau.py) - 3D Gaussians implementions which I used as a reference, modifying the `GaussianData` class, `naive_gaussian()` and `load_ply_4d()` methods

- [GitHub | 3D Gaussian splatting for Three.js](https://github.com/mkkellogg/GaussianSplats3D)
- [GitHub | Houdini Gaussian Splatting Viewport Renderers](https://github.com/rubendhz/houdini-gsplat-renderer)
- [GitHub | splaTV](https://github.com/antimatter15/splaTV)
- [GitHub | GaussianSplattingViewer](https://github.com/limacv/GaussianSplattingViewer)
- [gsplat.js now supports 4D Gaussian Splatting](https://github.com/huggingface/gsplat.js/tree/main/examples/4d)
- [Converting .PLY to .SPLATV](https://github.com/antimatter15/splaTV/issues/1)

Discussions
- [.splat universal format discussion #47](https://github.com/mkkellogg/GaussianSplats3D/issues/47)
- https://aras-p.info/blog/2023/09/13/Making-Gaussian-Splats-smaller/
- https://aras-p.info/blog/2023/09/27/Making-Gaussian-Splats-more-smaller/
