# Training on a real robot

Thanks Antonin, Roma, Wayve.ai, and crew for sharing these great ideas and code. This note is for those who might endeavor to use these techiniques to train a real robot to drive.

I've forked Antonin's work and added two clients:

1. a VAEDonkeyRemoteController in  donkey_remote.py derived from DonkeyRemoteContoller in my fork of donkeycar here:
https://github.com/tawnkramer/donkey

this client is designed to control the robot remotely while the data collection and backprop occur on the pc connected via networking.

2. A client class LocalControlEnv in local_client.py which is designed to run on the robot, where data collection and backprop occur on-board.

## Tradeoffs

There's a few tradeoffs. When control occurs off the robot ( scenerio 1 ) the training update is quite fast. But the control loop can have latency which prevents an ideal stop signal time, and therefore inaccuracy in the reward signal that can lead to longer training sessions that don't converge.

In scenario 2, the control occurs on the robot. So a joystick is used via bluetooth to start and stop training. The timing of this input has much less latency and can improve the timing of the stop signal and therefore increase accuracy of rewards and reduce training sessions. However, the robot running on a PI3 is many times slower than a pc and typically takes 20-25 seconds to process a training epoch.

## Setup

### Robot

1. Install my fork of donkeycar on your robot. Directions here:
https://github.com/tawnkramer/donkey/blob/master/docs/guide/install_software.md#get-the-raspberry-pi-working

2. setup the car to run using special remote template

```
donkey createcar --path ~/d2_rl --template manage_remote
```

3. create a myconfig.py and edit.

```
cd ~/d2_rl
nano myconfig.py
```

Add settings:
```
DONKEY_UNIQUE_NAME = "<your_name>"
MQTT_BROKER = "localhost"
```

And any other PWM settings you need to get you robot to run.

4. Install mosquitto mqtt broker service

```
sudu apt-get install -y mosquitto mosquitto-clients
```

Optional (scenario 2):

5. If you plan to train locally on the robot, then setup stable_baselines. This can take a lot of work. In my experiece tensorflow 1.12 didn't contain many of the libraries that were needed to pull in from tensorflow.contrib and therefore I had to heavily modify 
`/home/tkramer/env/lib/python3.5/site-packages/tensorflow/contrib/__init__.py`
to comment out packages which caused errors:
```
#from tensorflow.contrib import cloud
#from tensorflow.contrib import distribute
#from tensorflow.contrib import distributions
#from tensorflow.contrib import estimator
#from tensorflow.contrib import factorization
#from tensorflow.contrib import kernel_methods
#from tensorflow.contrib import learn
#from tensorflow.contrib import predictor
#from tensorflow.contrib import tensor_forest
#from tensorflow.contrib import timeseries
#from tensorflow.contrib import tpu
```
6. install my fork of learning to drive
```
git clone https://github.com/tawnkramer/learning-to-drive-in-5-minutes
```

### PC Setup

1. Follow Antonin's setup
2. Install my fork of donkeycar
```
git clone https://github.com/tawnkramer/donkey
pip install -e donkey[pc]
```
___________________________

## Train VAE

1.  First gather a bunch of images in whatever way is convenient. Put them on the host pc in a folder ./logs/images

2. Train vae
```
learning-to-drive-in-5-minutes/scripts/train_vae.sh
```

_____________________________

## Training Remote

### Robot

1. 
```
cd ~/d2_rl
python manage.py
```

### PC

1. modify ./scripts/train_rl.sh so that the donkey-name argument matches what you put in your myconfig for DONKEY_UNIQUE_NAME

2. run ``` learning-to-drive-in-5-minutes/scripts/train_rl.sh ```

3. hit `m` key to start the training session. The robot will start moving. Hit `m` key again when it has started to move outside the boundary. It will then do a short training session. When completed, repeat this cycle as until training session is over.

_____________________________

## Training on Robot

### Robot

1. plugin ps3 controller or get to work over bluetooth
https://github.com/tawnkramer/donkey/blob/master/docs/parts/controllers.md#physical-joystick-controller


2. ssh into robot. In one ssh session:
```
cd ~/d2_rl
python manage.py
```

3. copy vae to the robot
```
scp ./logs/vae-32.pkl <user>@<robot_ip>:~/learning-to-drive-in-5-minutes/logs
```

4. In another ssh session, modify learning-to-drive-in-5-minutes/scripts/local_train_rl.sh so that the donkey-name argument matches what you put in your myconfig for DONKEY_UNIQUE_NAME

5. run ``` learning-to-drive-in-5-minutes/scripts/local_train_rl.sh ```

6. Hit `circle` to start and stop training session. Expect training sessions to last 20-25 seconds before it's ready to drive again.


