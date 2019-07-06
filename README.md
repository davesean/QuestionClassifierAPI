# QuestionClassifierAPI

A basic Flask API that takes in two values and returns either 0 or 1.

# Local

To start the API locally cd into the api folder and run:
```
cd api
python api.py
```
To send requests use either a browser or curl:
```
curl -X GET "http://127.0.0.1:5000/api/v1/QC/?ql=90&np=1000"
```
The first value ql is the number of characters.
The second value np stands for the number of purchases.

# Docker
After you set up your Docker environment cd into the api folder if you are not in it anymore and run:
```
cd api
docker build -t qc .
docker run -i -t -p 5000:5000 qc
```
The first docker command downloads all the required libraries and copies the files over into the container.
The second command run it, while also forwarding the port, such that one can access it.
Again using curl, one can send requests.
To send a request with curl, we change the IP as follows:
```
curl -X GET "http://0.0.0.0:5000/api/v1/QC/?ql=90&np=1000"
```
# Docker-machine
If you use a virtual machine, then the IP isn't the default local IP, but the IP of the virtual environment.
To find the IP:
```
docker-machine ls
```
This command returns a table with a column for the URL with the IP.
```
NAME   ACTIVE   DRIVER       STATE     URL                         SWARM   DOCKER     ERRORS
dev    -        virtualbox   Running   tcp://192.168.90.101:2370           v18.09.7
```
To send a request with curl, we change the IP as follows:
```
curl -X GET "http://192.168.90.101:5000/api/v1/QC/?ql=90&np=1000"
```
# Training a model
If you have data, you can train this model based on a fully connected neural network.
First you need to install all the required packages necessary and
then with the data in a .csv file in the base folder, run:
```
pip install -r requirements.txt
python train_model.py
```
After training, the model's checkpoint is directly saved into the api folder, where you can have requests answered either using it locally or with a docker container.
To have the model decide in the API, the curl request is extended by one parameter:
```
curl -X GET "http://127.0.0.1:5000/api/v1/QC/?ql=90&np=1000&model=1"
```
