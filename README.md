# Table detector exercise  
  Thank you for considering my application ! Here is my submission for the
  exercise described in BACKEND.md.  
  I hope you'll like it :) 
  
## Installation   
### Docker (recommended)  
To use this repo you simply need to fire up the docker container:  
  
```shell  
docker compose up --build  
```
This will build the image defined in the dockerfile and as a bonus  
run all pytest tests.  
_Note: the `--build` flag is only needed if you need to regenerate the image.  
Otherwise you can simply use `docker compose up`_
### Locally  
You can also use this project locally (using a venv is recommended)  
  
```shell  
python3 -m venv dataleon-dev  
source dataleon-dev/bin/activate  
  
pip install -r requirements.txt  
```  
  
## Usage  
Once you've setup you environment using the method of your choice,  
it's time to start developing!    
The class is quite straightforward to use, let's get into it step by step:  
### Initialization  
Here we define all the attributes of the class we'll need later:  
```python  
 def __init__(self):  
 self.processor = DetrImageProcessor.from_pretrained(  
 "TahaDouaji/detr-doc-table-detection") 
self.model = DetrForObjectDetection.from_pretrained(  
 "TahaDouaji/detr-doc-table-detection")  
```  
If you want to use a better performing model or you own fine-tuned version,  
this is the place to do it.    
  
### Preprocessing  
Although pretty basic in this example, you can do a lot within this function.  
Depending on the quality of your images, you might add operations to bump  
the contrast and other metrics of use pillow's `verify` method to filter out  
any corrupted images.  
  
### Prediction  
This is where the magic happens. We load the preprocessed image into our model   
and predict whether it contains an image. You can set a threshold to filter out  
predictions with a low confidence score (by default anything under 90% gets  
thrown out)  
  
### Prediction Formatting  
Since `predict` returns an object containing tensors, we clean up the data a bit  
so it is easier to use withing python. Perhaps it is possible to define another  
processor for the model but this was not within the scope of this exercise

### Tests
This class comes with a suite of tests to verify everything works fine.
You can find them in test_app.py.  
These tests verify the following scenarios:  
- All white image  
- Empty file  
- Invalid Format
- Successful extraction
- Failure during pre-processing
- Performance against various image formats of the same document

You can run all those tests manually by typing `pytest`. They will also be run
everytime you run the docker container.
