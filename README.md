# R-MAC Layer for TensorFlow 2

Implementation of R-MAC (Regional Maximum Activations of Convolutions) for TensorFlow 2

&copy; 2020 IMATAG

wwww.imatag.com

Author: Vedran Vukotic



### Details:
* works in TensorFlow with or without the high-level Keras API
* easy to replace in place of the last layers of a pretrained Keras Applications network
* models using the R-MAC layer can be exported to TensorFlow Lite and used transparently

### Usage:
```python
from rmac import RMAC

...
# function definition:
# RMAC(shape, levels=3, power=None, overlap=0.4, norm_fm=False, sum_fm=True, verbose=False)

# create RMAC Layer
rmac = RMAC(model.output_shape)

# add RMAC Layer to existing sequential model
model.add(Lambda(rmac.rmac, name="rmac"))
```
#### Optional Parameters:
* _levels_ - number of levels / scales at which to to generate pooling regions (default = 3)
* _power_ - power exponent to apply (not used by default)
* _overlap_ - overlap percentage between regions (default = 40%)
* _norm_fm_ - normalize feature maps (default = False)
* _sum_fm_ - sum feature maps (default = False)
* _verbose_ - verbose output - shows details about the regions used (default = False)

### Files:
* _rmac.py_ - main module with R-MAC implementation
* _demo_tensorflow.py_ - example usage with a custom model defined via the Keras API
* _demo_keras_app.py_ - example usage with a pretrained model from Keras Applications
* _demo_keras_app_tflite.py_ - example of a TF-Lite export / import of a custom model containing a custom R-MAC layer


### Citing:
If you liked and used the code, please consider citing the work where it was used (and implemented for):
```
@article{vukotic2020classification,
  title={Are Classification Deep Neural Networks Good for Blind Image Watermarking?},
  author={Vukoti{\'c}, Vedran and Chappelier, Vivien and Furon, Teddy},
  journal={Entropy},
  volume={22},
  number={2},
  pages={198},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

as well as the original paper of the R-MAC creator:
```
@article{tolias2016particular,
   author    = {Tolias, Giorgos and Sicre, Ronan and J{\'e}gou, Herv{\'e}},
   title     = {Particular object retrieval with integral max-pooling of CNN activations},
   booktitle = {Proceedings of the International Conference on Learning Representations},
   year      = {2016},
}
```
