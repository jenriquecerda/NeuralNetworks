## Neural Networks implementation

This projects seeks to explore OpenCV functionality.

### Running scripts

Create a virtual environment:

```sh
$ python3 -m venv .venv
```

Activate virtual environment:

```sh
$ source .venv/bin/activate
```

...or to deactivate once in (virtual env):

```sh
$ deactivate
```

Once your virtual environment is activated, install required packages:

```sh
$ pip install -r requirements.txt
```

### Camera Selection

If using default camera, `index = 0`.
See [OpenCV Documentation.](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a5d5f5dacb77bbebdcbfb341e3d4355c1)

```python
capture = cv2.VideoCapture(index)
```
