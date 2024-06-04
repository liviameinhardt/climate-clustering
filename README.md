# Climate Clustering Model

Recognizing the pressing need for adaptable models in the face of climate change effects, this project aims to examine these impacts using local climate data. By clustering climate patterns and analysing trends, the goal is to gather evidence of shifts in observed climate patterns over time. To conduct this analysis, a self-supervised technique called Swapping Assignments between multiple Views (SwAV) is employed.

# Methodology

Build from [Ayush Thakur](https://twitter.com/ayushthakur0) and [Sayak Paul](https://twitter.com/RisingSayak)
TensorFlow 2 implementation of [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882) by Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, Armand Joulin.

A few changes to this code were made:
- Uses Resnet-18 instead of Resnet-50
- Set options.experimental_deterministic = True   
- Add a seed to dataset.shuffle on get_multires_dataset function

# Results 

Initial results can be seen in [Self-supervised weather data clustering for inspecting local climate change](https://hdl.handle.net/10438/35365)


