# Content-Based Image Retrieval with Deep Learning

<p align="center">
  <img src="/report/img/complete-system-architecture.jpg" alt="Image Retrieval System Architecture"/>
</p>

## Table of Contents
* [About the Project](#about-the-project)
* [Dataset](#dataset)
* [Technologies](#technologies)
* [What You Need](#what-you-need)
* [Run the Application](#run-the-application)
* [Demo](#demo)
* [License](#license)

## About the Project
Although we communicate in a variety of ways with each other, our favorite way to do so is via the written word. However, when you think, do you think in words or images? Pictures sometimes are easier to recognize and process than words. What  is more, they can be a way  of communicating something that’s impossible to verbalize, like thoughts, feelings, memories. So, how can we improve information retrieval and accessibility via images?
 
<p>
  <div>
    This project's purpose is to find a way to make image retrieval as accurate as possible by leveraging computer vision methods. The idea that is proposed is simple yet
    complex to implement:
  </div>
  <div align="center">
    <b>We should use machine learning and deep learning models in the background to process images.</b>
  </div>
  <div>
    By utilizing computer vision models we are able to extract image features, which will be indexed in the search engine. By doing so, image retrieval will be done by
    <b>comparing query-images and index-images feature vectors using cosine similarity</b>.
  </div>
</p>

There are two computer vision methods we've looked into:
* **Bag of Visual Words**: The general idea is to represent an image as a set of features. Features consists of keypoints and descriptors. We use the keypoints and descriptors to construct visual vocabularies and then we quantize the image features. By doing so, we have successfully represented images as a frequency histogram of features that are in the images. With the use of visual vocabularies, later, we can perform many tasks, such as classification, retrieval and more.
* **Visual Embeddings**: Refers to the collection of features of the last fully connected layer (prior to a loss layer) appended to a CNN. The visual embeddings are learned by jointly training the feature extractor with the embedding layer and the classifier **on the classification task**. 

The two Information Retrieval Systems we have explored, are evaluated using the **trec_eval** evaluation tool and its metrics. Our focus is mainly on the behaviour of **mean average precision** on the top 100 retrieved images.

For the full presentation of the problem, our approach, the results, and the system's architecture, you can download and look into this [report](report/report.pptx) (powerpoint format).

## Dataset
To build the search engine, **CIFAR-10 dataset** has been used. This is an image-based dataset by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton and it is publicly available from the [University of Toronto](https://web.cs.toronto.edu/). You can download it from [here](https://www.cs.toronto.edu/~kriz/cifar.html).

The data consist of of images, about 50,000 training images and 10,000 test images. Each image is a **32x32 color image**. The dataset contains **10 classes** which are mutually exclusive (e.g. there is no overlap).
* Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

To build the search engine, we consider the **training images as index images**, meaning that these will be indexed in our search engine. In the same manner, we consider the **test images as query images**. The **query relevance** is defined as follows: **each query image (test image) is related with a set of indexed images (training images) where the relevance relationship depends on the class label**. 
* For example, a query image that is a car is associated with indexed images that belong to the car class.

## Technologies
**Programming Language**: Python <br>
**Search Engine**: Elasticsearch <br>
**Machine Learning**: OpenCV, Scikit-Image, Scikit-Learn <br>
**Deep Learning**: Pytorch, Ray <br>
**Frontend**: HTML, Jinja2, CSS <br>
**Application Framework**: Flask <br>
**Other Libraries**: NumPy, Matplotlib

## What You Need
* Anaconda
* Elasticsearch client
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

## Demo
1. Run application.
<p align="center">
  <img src="/report/img/demo-1.png" alt="Demo pt.1"/>
</p>

2. Upload your image query and search.
<p align="center">
  <img src="/report/img/demo-2.png" alt="Demo pt.2"/>
</p>

2. Scroll down to see the top 10 relevant images, with respect to your query.
<p align="center">
  <img src="/report/img/demo-3.png" alt="Demo pt.3"/>
</p>
      
## License
Distributed under the MIT License. See [LICENSE.md](LICENSE.md) for more information
