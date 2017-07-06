# Object Recognition Skill

A object recognition service for Mycroft AI using [Google's TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection) and [OpenCV](http://opencv.org/).

This skill is a proof of concept to use tensorflow and openCV to provide object recognition using "whatever image source"

Hopefully this is just a start and with optimization and further development this skill will provide a concept to create more skills around object recognition

WIP

# TODO

bus messages to trigger
helper class to use anywhere

# logs

        2017-07-06 09:01:46,519 - ObjectRecogSkill - INFO - Loading test image
        2017-07-06 09:01:46,521 - ObjectRecogSkill - INFO - Detecting objects
        2017-07-06 09:01:48,727 - ObjectRecogSkill - INFO - Processing labels
        2017-07-06 09:01:48,728 - ObjectRecogSkill - INFO - Processing scores
        2017-07-06 09:01:48,729 - ObjectRecogSkill - INFO - Processing bounding boxes
        2017-07-06 09:01:48,729 - ObjectRecogSkill - INFO - Counting objects and removing low scores
        2017-07-06 09:01:48,729 - ObjectRecogSkill - INFO - detected : {u'bench': 1, u'potted plant': 2, u'book': 1, u'vase': 1}
        2017-07-06 09:01:48,818 - Skills - DEBUG - {"type": "speak", "data": {"expect_response": false, "utterance": "1 bench \n2 potted plant \n1 book \n1 vase \n", "metadata": {}}, "context": {"target": null, "mute": true, "photo": "https://scontent.fagc1-2.fna.fbcdn.net/v/t1.0-1/p32x32/19275245_236756243492432_4770223864515611100_n.jpg?oh=ccb8132bb115a6a48636291cdae79ed4&oe=59D1743E", "destinatary": "fbchat_100014741746063", "source": "server_skills", "user": "Jarbas Ai", "more_speech": false}}


## Requirements
- [Mycroft](https://docs.mycroft.ai/installing.and.running/installation)
- [TensorFlow 1.2](https://www.tensorflow.org/)
- [OpenCV](http://opencv.org/)

## Installation

Clone the repository into your skills directory. Then install the
dependencies inside your mycroft virtual environment:

If on picroft just skip the workon part and the directory will be /opt/mycroft/skills

```
cd /opt/mycroft/skills (or wherever your working skills directory is located)
git clone https://github.com/eClarity/skill-realtime-object-recognition
workon mycroft
cd skill-realtime-object-recognition
pip install -r requirements.txt
```

## Usage:
* `View Objects`
