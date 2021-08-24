# How to train a model in Tensorflow Obj Detection API

Take a look at [Official TensorFlow Object Detection API Tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) to set up the environment and to have more insights.

Look also to the git repo [README](https://github.com/tensorflow/models/tree/master/research/object_detection) file to have more examples and tutorials. 


### OpenImagesV4 Dataset
Open Images is a dataset of ~9M images annotated with image-level labels, object bounding boxes, object segmentation masks, visual relationships, and localized narratives. It contains a total of 16M bounding boxes for 600 object classes on 1.9M images, making it the largest existing dataset with object location annotations. 

### Download a subset of OpenImagesV4 Dataset

There is a [GitHub Repo](https://github.com/EscVM/OIDv4_ToolKit) that allows to download all the images of this this dataset containing a specific class (and only that class!).The annotations of the images will include just labels and boxes for that class too (for example I downloaded just images of license plates).

Example of a command for executing OIDv4_ToolKit:
```python3 main.py downloader --classes Vehicle_registration_plate --type_csv all```
> **--classes** : you specify what classes you want to download (write the corresponding label). if the class name has a space in it , like "polar bear", you must write it with **underscores** "polar_bear". To download multiple classes, you can create a classes.txt file (and give this to the --classes opt) in which each line corresponds to a class name.
> 
>**--type_cvs** : you can select "train", "test", "validation", "all". Selecting "all" you will download 3 folders with images divided into train, valid and test sets (so you are downloading all the images available for your class)

### TFRecord generation

There is a [GitHub Repo](https://github.com/zamblauskas/oidv4-toolkit-tfrecord-generator/blob/master/README.md) that gives an easy script for generating the TFRecords of the OIDv4 subset downloaded.

`python generate-tfrecord.py --classes_file=classes.txt --class_descriptions_file=class-descriptions-boxable.csv --annotations_file=train-annotations-bbox.csv --images_dir=train_images_folder/ --output_file=train.tfrecord`

**IMPORTANT:** Here the classes.txt file doesn't want underscores instead of white spaces!!! (unlike the dataset downloader OIDv4ToolKit!)
For example, you have to write again "polar bear" instead of "polar_bear".



### Prepare Workspace

``` 
tensorflow_api
 └─ models
     ├── official
     └── research
            └── object_detection
                  |     ...
                  └── greenwaves
                        ...
```

```
greenwaves
  ├── pretrained_models (store starting checkpoint for finetuning)
  ├── training (where training checkpoints are saved)
  |     └── config_file.config
  ├── trained-inference-graphs (exported frozen graph)
  ├── model_main.py (main script for training and evaluation)
  └── export_inference_graph.py
```

- **pre-trained-model**: This folder will contain the pre-trained model of our choice, which shall be used as a starting checkpoint for our training job.

- **training**: This folder will contain the training pipeline configuration file *.config, as well as a *.pbtxt label map file and all files generated during the training of our model by "model_main.py" script.

- **trained-inference-graphs**: this folder will contain the frozen graph exported by "export_inference_graph.py" script.

### Annotating your own images

You can use the [LabelImg](https://github.com/tzutalin/labelImg) package.
 
- Use [xml_to_cvs.py](???) script to convert xml files generated my LabelImg to a unique csv file with all annotations.

- Use [from_cvs_to_tfrecord.py](???) (inside labelImg_utilities folder) script to convert the csv file to a tfrecord

### Label Map 

TensorFlow requires a label map, which namely maps each of the used labels to an integer values. This label map is used both by the training and detection processes.

The Label Maps for standard datasets can be found in the tensorflowAPI repository at `models/research/object_detection/data`

The classes included in the label map should be exactly the ones that you are training on. If you set to train on just 1 class, then leave only that class in the label_map.pbtxt file with `id: 1`.

### Configuration

How to setup the training/config_file.config file. 

You can find all the default config files in `models/research/object_detection/samples/configs` folder. Make sure to set correctly all the paths (search for "PATH_TO_BE_CONFIGURED" to find the fields). 

More details on the essential fields to set can be found [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configuring-a-training-pipeline) and at the "Useful stuff" paragraph.


### Metrics

All the metrics available are declared in `models/research/object_detection/eval_util.py`.

By default, [COCO metrics](http://cocodataset.org/#detection-eval) are used. Follows the definition:

![image](images/Coco-metrics.PNG)

You can look at the [Tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) for reference accuracy of the mAP metric over COCO and OpenImages dataset

### Training

Use the `model_main.py` script to train your model. It will save checkpoints and tensorflow events that will keep trace of the training process.


Example training command:
```python model_main.py --pipeline_config_path=training/ssd_mobilenet_v2_oid_v4.config --model_dir=training/ --alsologtostderr ```

> **--model_dir** : where checkpoints ad tensorboard logs will be saved

### TensorBoard

To visualize the training history (mAP scores, loss, learning rate, predictions) use [TensorBoard](https://www.tensorflow.org/tensorboard). Look [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#tensorboard-sec) to install it.

First, lunch a session on the training folder with this command:
```tensorboard --logdir='training'```

Then open firefox and copy the http address that pops up. You will see:

- **DetectionBoxes_Precision**: the mAP score is the main reference score, with IoU averaged on [0.5 ; 0.95]
- **DetectionBoxes_Recall** : Average Recall scores
- **Loss**: validation loss, divided in "classification" and "localization" losses
- **learning_rate**: how the learning rate decays with steps
- **learning_rate_1**: same as before
- **loss**: same as "Loss", is the validation loss
- **loss_1**: is the training loss!
- **loss_1**: same as "loss_1", is the training loss

You can switch to the IMAGES folder inside TensorBoard to see how predictions evolve with training steps (it's fun!)


### Export frozen graph

Before testing a model you need to export a frozen graph with the `export_inference_graph.py`:

Example command:
`python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v2_oid_v4.config --trained_checkpoint_prefix training/model.ckpt-##### --output_directory trained-inference-graphs/output_inference_graph_320x240.pb`

where:
> **--pipeline_config_path**: config file 
> **--trained_checkpoint_prefix**: model saved checkpoint (replace #### with the corresponding step number) 
> **--output_directory**: where to save the frozen model

### Testing

Example testing command:

```python model_main.py --pipeline_config_path=training/ssd_mobilenet_v2_oid_v4.config --checkpoint_dir=trained-inference-graphs/output_inference_graph_320x240.pb --run_once```

As output you will get the metrics selected, COCO metrics by default.

### Useful stuff 

#### 1. Set image size for trainig and testing

In `training/config_file.config` set these (self explanatory) parameters: 

```     
model {
  ssd {
    image_resizer {
      fixed_shape_resizer {
        height: 240
        width: 320
      }
    }
}}
```


#### 2. Freeze backbone at training time

In `training/config_file.config` set this parameter: 

```     
train_config: {
  ...
  freeze_variables: "FeatureExtractor"
  ...
}
```

#### 3. how to quantize

In `training/config_file.config` set these parameters: 

```     
graph_rewriter {
  quantization {
    delay: 60000       # after how many steps quantization kiks in
    weight_bits: 8      # bits for weights
    activation_bits: 8  # bits for act funct
  }
}
```

#### 4. Set learing rate exponential decay

In `training/config_file.config` set these parameters: 

```     
train_config: {
  optimizer {
    rms_prop_optimizer: {
      learning_rate: {
        exponential_decay_learning_rate {
          initial_learning_rate: 0.0008
          decay_steps: 20000
          decay_factor: 0.95
        }
      }
      momentum_optimizer_value: 0.9
      decay: 0.9
      epsilon: 1.0
    }
  }
```

> **decay_steps**: how many steps before decreasing the learning rate
> **decay_factor: 0.95**: the final learning rate will be the 95% of the original

![image](images/exponential_decay_example.png)

#### 5. Print trainable variables

Add [here](https://github.com/tensorflow/models/blob/master/research/object_detection/model_lib.py#L401) this line:

``` print('Tranable variables: \n', *trainable_variables, sep = "\n") ```

#### 6. Count the number of network's parameters (Tensorflow)


Two ways:
1. The original Tensorflow repo provides a [script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/inspect_checkpoint.py ) called `inspect_checkpoint` to analize `.ckpt` files.
You can use this to count the number of the model's parameter
Note: this counts also all the training parameters that wont be part of the inference model. So it is a pessimistic estimation.

2. convert thefrozen graph `graph.pb` to tf lite with the [Tensorflow command](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/cmdline_examples.md) `tflite_convert` with "dummy 8-bit quantization". Look at the size of the `.tflite` file, that is the number of parameters

Example command: 

```tflite_convert --graph_def_file=graph.pb --output_file=graph.tflite --inference_type=QUANTIZED_UINT8 --default_ranges_min=0 --default_ranges_max=6 --input_arrays=input --output_arrays=d_predictions --mean_values=128 --std_dev_values=127``` 

Note: find the names of `--input_arrays` and `--output_arrays` opening the  `graph.pb` with [netron web](https://lutzroeder.github.io/netron/)

![image](images/netron.PNG)




<!-- 
SCALETTA:
- scaricare dataset
- generare tfrecord
  
- come bloccare backbone
- come stampare tutte le variabili trainabili
- settare size
- come quantizzare
- settare exp decay
  
- settare tutti i path ai file (finetune checkpoint, tfrecords, label maps), il numero diclassi
- generare il file label map
- analize ckpt file con tensorflow repo

- STATO DELL'arte EXCEL file che ho fatto un mese fa -->
