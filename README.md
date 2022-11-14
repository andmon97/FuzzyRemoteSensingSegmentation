# Segmentation of Remotely Sensed Images with an Adaptive Neuro Fuzzy Inference System
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Pq1CU0vskGq2A5hyANWM7gGJZ_KHqxvM?usp=sharing)

The semantic segmentation of remotely sensed images is a difficult task because the images do not represent well-defined objects. To tackle this task, fuzzy logic represents a valid alternative to convolutional neural networks—especially in the presence of very limited data, as it allows to classify these objects with a degree of uncertainty. Unfortunately, the rules for doing this have to be defined by hand. To overcome this limitation, in this work we propose to use an adaptive neuro-fuzzy inference system (ANFIS), which automatically infers the fuzzy rules that classify the pixels of the remotely sensed images, thus realizing their semantic segmentation. The resulting fuzzy model guarantees a good level of accuracy
This model is also explanatory, since the classification rules produced are similar to the way of thinking of human beings.

This approach is used for segmenting remotely sensed images into six different classes: Building (Red), Road (Yellow), Pavement (Darker Yellow), Vegetation (Green), Bare Soil (Grey) and Water (Blue).

![](images/exampleSegmentation.png)

This work is part of the Computer Vision exam at University of Bari "Aldo Moro".

****
### Repository content

- Folder 'ANFIS-imgSatellitari' contains the ANFIS code (to train and test the model). There is also the folder 'model'
  which contains the ANFIS trained models (0 was used for the exam and the others are the model for the 6-class segmentation with 2, 3, 4 Fuzzy Sets per variable)
- Folder 'preprocessing' contains the original dataset, the pixel dataset and the scripts to generate it.
- The notebook "Anfis_training" is the notebook with used for the training of the 6-class segmentation model.

****
### Best Model
For the experiment the "reducedTopClass" were used, which was built by choosing the top 3 images with the greater number of pixel for each class (so for 6 classes there are 18 total images used to compose the pixel dataset).

The best model used for the experiment are those that have the words "topClass" in the models folder.

****
### Paper

Moreover, the work led to a [paper](https://www.researchgate.net/publication/358021524_Segmentation_of_remotely_sensed_images_with_a_neuro-fuzzy_inference_system) that was presented at the 13th International Workshop on Fuzzy Logic and Applications (WILF2021).

#### BibTeX Citation
```
@inproceedings{castellano2021segmentation,
  title={Segmentation of remotely sensed images with a neuro-fuzzy inference system.},
  author={Castellano, Giovanna and Castiello, Ciro and Montemurro, Andrea and Vessio, Gennaro and Zaza, Gianluca},
  booktitle={WILF},
  year={2021}
}
```
****

### References

[1]R. Wang, J. A. Gamon, Remote sensing of terrestrial plant biodiversity, Remote Sensing ofEnvironment 231 (2019) 111218.

[2]M. Weiss, F. Jacob, G. Duveiller, Remote sensing for agricultural applications: A meta-review, Remote Sensing of Environment 236 (2020) 111402.

[3]X. Yuan, J. Shi, L. Gu, A review of deep learning methods for semantic segmentation ofremote sensing imagery, Expert Systems with Applications (2020) 114417.

[4]O. Ronneberger, P. Fischer, T. Brox, U-net: Convolutional networks for biomedical imagesegmentation, in: International Conference on Medical image computing and computer-assisted intervention, Springer, 2015, pp. 234–241.

[5]V. Badrinarayanan, A. Kendall, R. Cipolla, Segnet: A deep convolutional encoder-decoderarchitecture for image segmentation, IEEE Trans. on PAMI 39 (2017) 2481–2495.

[6]R. Dong, X. Pan, F. Li, Denseu-net-based semantic segmentation of small objects in urbanremote sensing images, IEEE Access 7 (2019) 65347–65356.

[7]Y. Yi, Z. Zhang, W. Zhang, C. Zhang, W. Li, T. Zhao, Semantic segmentation of urbanbuildings from vhr remote sensing imagery using a deep convolutional neural network,Remote sensing 11 (2019) 1774.

[8]J. M. A. Moral, C. Castiello, L. Magdalena, C. Mencar, Explainable Fuzzy Systems: Pavingthe way from Interpretable Fuzzy Systems to Explainable AI Systems, Springer, 2021.

[9]G. Casalino, G. Castellano, C. Castiello, V. Pasquadibisceglie, G. Zaza, A fuzzy rule-baseddecision support system for cardiovascular risk assessment, in: International Workshopon Fuzzy Logic and Applications, Springer, 2018, pp. 97–108.

[10]H. Leon-Garza, H. Hagras, A. Peña-Rios, A. Conway, G. Owusu, A big bang-big crunchtype-2 fuzzy logic system for explainable semantic segmentation of trees in satellite imagesusing hsv color space, in: IEEE Int. Conf. on Fuzzy Systems (FUZZ-IEEE), 2020, pp. 1–7.

[11]C. Wang, A. Xu, X. Li, Supervised classication high-resolution remote-sensing imagebased on interval type-2 fuzzy membership function, Remote Sensing 10 (2018) 710.

[12]K. Shihabudheen, G. N. Pillai, Recent advances in neuro-fuzzy system: A survey,Knowledge-Based Systems 152 (2018) 136–162.

[13]S. K. Meher, N. S. Kothari, Interpretable rule-based fuzzy elm and domain adaptation forremote sensing image classication, IEEE Trans. on Geoscience and R. Sensing (2020).

[14]Z. Tianyu, J. Xu, Hyperspectral remote sensing image segmentation based on the fuzzydeep convolutional neural network, in: 13th International Congress on Image and SignalProcessing, BioMedical Eng. and Informatics (CISP-BMEI 2020), IEEE, 2020, pp. 181–186.

[15]J.-S. Jang, Ans: adaptive-network-based fuzzy inference system, IEEE Transactions onSystems, Man, and Cybernetics 23 (1993) 665–685.

[16]T. Takagi, M. Sugeno, Fuzzy identication of systems and its applications to modeling andcontrol, IEEE Transactions on Systems, Man, and Cybernetics (1985) 116–132.

[17]J.-S. Jang, C.-T. Sun, Functional equivalence between radial basis function networks andfuzzy inference systems, IEEE Transactions on Neural Networks 4 (1993) 156–159.

[18] M. Brown, C. J. Harris, Neurofuzzy adaptive modelling and control, Prentice Hall, 1994.

[19]Z. Shao, W. Zhou, X. Deng, M. Zhang, Q. Cheng, Multilabel Remote Sensing Image RetrievalBased on Fully Convolutional Network, IEEE Journal of Selected Topics in Applied EarthObservations and Remote Sensing 13 (2020) 318–328.

[20]E. Shelhamer, J. Long, T. Darrell, Fully convolutional networks for semantic segmentation,IEEE Transactions on Pattern Analysis and Machine Intelligence 39 (2017) 640–651.
