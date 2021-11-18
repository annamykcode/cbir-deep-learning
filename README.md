# Content-Based Image Retrieval with Deep Learning

<p align="center">
  <img src="/report/img/complete-system-architecture.jpg" alt="Image Retrieval System Architecture"/>
</p>

## Table of Contents
* [Technologies](#technologies)
* [What You Need](#what-you-need)
* [Run the Application](#run-the-application)
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

  * Create the environments from the **cbir-ml.yml** and **cbir-dl.yml** file:
      ```
      conda env create -f cbir-ml.yml 
      ```
      ```
      conda env create -f cbir-dl.yml 
      ```
* CIFAR-10 dataset.

  * Activate **cbir-ml** environment:
    ```
    conda activate cbir-ml
    ```
  * Run ``` notebooks/Search Engine Files (Miscellaneous).ipynb ``` jupyter notebook (no need to run the 3d section).
  * The CIFAR-10 data can be found under ``` static/cifar10/ ```.

## Run the Application
To run the application:
* start Elasticsearch client (on Windows) by running ``` elasticsearch-x.xx.x/bin/elasticsearch.bat ```.
* activate **cbir-dl** environment:
  ```
  conda activate cbir-dl
  ```
* run the following command in the terminal window (in the complete) directory:
  ```
  python app.py
  ```
Then, on the browser, visit ``` http://localhost:5000/ ``` to open the web page.
      
## License
Distributed under the MIT License. See [LICENSE.md](LICENSE.md) for more information
