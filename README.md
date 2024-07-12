# Bubbles
Bubbles is a Apriltag detection system developed by FRC Team 10015

## Usage
Install these packages manditory to use main.py and other scripts:
```bash
$ pip install flask
$ pip install opencv-python
$ pip install pupil_apriltags
$ pip install numpy
$ pip install glob
$ pip install roboflow
```
Before using main.py, please calibrate in calibrate.py

### testing.py

Is a safe alternative to test the detection before using the main script

```🛈``` | calibrate.py is currently under development, if you know your focal length and Apriltag length import it manually in settings.txt

### Flask

After running main.py, the display output would be in a ip running in your localhost.

```🛈``` | Use ```http://``` instead of ```https://``` when typing the link.

```🛈``` | This is still under development! Either this kind of works or doesn't.

## Calibration

```🛈``` | Under development

## Custom Training

Custom training is in train.py using the model [YOLOv8](https://github.com/ultralytics/ultralytics) from Ultralytics. we are also using [Roboflow](https://roboflow.com) to get the dataset to train the model. Edit lines 18-20 and 97 to your preferences before using.

```🛈``` | Under development

### YOLOv8

```🛈``` | Under development
