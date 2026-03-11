# project_merbok

Batch processing code for CoastSeg in northern Alaska. Utilizes deep learning based filtering models, segmentation models, and additional post-processing steps for mapping shorelines and reducing noise.

Setup:

1) Install wsl/miniforge

2) Create conda environment

```conda env create -f project_merbok_full.yaml```

3) Setup shoreline sections (GCRRSSS) with reference shorelines and reference polygons.

Directory strucure

root
    G#
        C#
            RR##
                SSS###
                

4) Configure settings with config_gui.py

```python config_gui.py```

5) Run various functions with merbok_workflow.py

```python merbok_workflow.py --config config.yaml```

