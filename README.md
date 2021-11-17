# Content-Based Image Retrieval with Deep Learning

<p align="center">
  <img src="/report/img/complete-system-architecture.jpg" alt="Image Retrieval System Architecture"/>
</p>

## Table of Contents
* [Technologies](#technologies)
* [What You Need](#what-you-need)
* [License](#license)

## Technologies
**Programming Language**: Python <br>
**Search Engine**: Elasticsearch <br>
**Machine Learning**: OpenCV, Scikit-Image, Scikit-Learn <br>
**Deep Learning**: Pytorch, Ray <br>
**Frontend**: HTML, Jinja2, CSS <br>
**Application Framework**: Flask <br>
**Other Libraries**: NumPy, Matplotlib

## What You Need
* [Anaconda](https://www.anaconda.com/)
* [Elasticsearch client](https://www.elastic.co/)
* Virtual environments from .yml files.

  * Create the environment from the **cbir-ml.yml** and **cbir-dl.yml** file:
      ```
      conda env create -f cbir-ml.yml 
      ```
      ```
      conda env create -f cbir-dl.yml 
      ```
  * Activate the new environment (only cbir-dl is needed for the application): 
      ```
      conda activate cbir-dl
      ```
  * Verify that the new environment was installed correctly:
      ```
      conda env list
      ```
      
## License
Distributed under the MIT License. See [LICENSE.md](LICENSE.md) for more information
