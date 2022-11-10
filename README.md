# Midterm project for ML Zoomcamp

## Create the models

To generate the models execute the train.py script. The train.py
was exported from the notebook.ipynb.
```bash
$ ptyhon train.py
```

The result are two trained models exported for BentoML.
The two models are :
* *star_type_model:<hash>*  for a XGBoost model
* *star_type_model_skl:<hash>*  for a random forest model

The *bentofile.yaml* contains the information how to build the service.
You can commented in/out the one model you want to use for performance testing.

To build the bentoml  model you have to use:
```bash
$ bentoml build
```
Note:  For convenience I always load int eh predict.py and predict_skl.py the
latest generated model.

To start the service for testing you can enter:
```bash
$ bentoml serve predict.py:svc --reload
# or depending which model you want to test
$ bentoml serve predict_skl.py:svc --reload
```

To test the model you can open the browser and use the url: http://localhost:3000

For performance testing use those commands:
```bash
$ bentoml serve predict.py:svc --production
# or depending which model you want to test
$ bentoml serve predict_skl.py:svc --production
```

For testing you can use locust testing tool.
I have created a locustfile.py
To start the tests you have to install locust and start it with:
```bash
$ locust -H http://localhost:3000
```

After testing the two models I decided to use the XGBoost model.

To build the container I used the following command:

```bash
$ bentoml containerize startype:ey724qtbg2too4yo
```

Running docker to deploy the service locally.
```bash
$ docker run -it --rm -p 3000:3000 startype:ey724qtbg2too4yo serve --production
```
Here I got into trouble when starting the container I got the error message:
Error: [bentoml-cli] `serve` failed: Failed loading Bento from directory /home/bentoml/bento: Failed to import module "predict": No module named 'sklearn'

I put sklearn as an dependecy into the bentofile.yaml I tried a lot of different things but was not able to resolve the issue.

