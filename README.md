# gd_tools.py
Computational design tools for generative design in Python

Useage information can be found in this tutorial: https://medium.com/generative-design/routing-with-graphs-5fb564b02a74

# process_images.py
Script for generating multi-image composites from design screenshots output during optimization.

Available parameters can be found in the `if __name__ == "__main__":` code block at the end of the script:
- mode - (0,1,2) overlay mode for images: (0) transparency, (1) multiply*, (2) darken*
- mix - (float) blend factor for multiply and darken blending modes, where 1.0 is most blended (darkest image) and 0.0 is least blended (no effect)
- gen_size - (int) number of designs per generation (number of designs to merge into single image)
- gen_stride - (int) number of generations to skip
- des_stride - (int) number of designs to skip
- make_index - (True/False) make a single grid layout with all composite images
- aspect - (float) target (w/h) aspect ratio for index image

*for best results with the multiply and darken blending modes, images should have a white background

![bridge](bridge.png)
