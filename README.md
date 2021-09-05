# Tensorflow learning rate patcher

### Description
This is a simple project which implements one feature: it allows you to 
change the learning rate schedule without stopping the training process by
editing the specified file. The file is just a json containing the output
of the `tensorflow.keras.optimizers.schedules.deserialize` method, for example:
```json
{
  "class_name": "PolynomialDecay",
  "config":{
    "cycle": false,
    "decay_steps": 1000,
    "end_learning_rate": 0.01,
    "initial_learning_rate": 0.1,
    "power": 0.5
  }
}
```
Custom learning rate schedules are also supported. In order to use them pass
`custom_objects` to `LiveLrSchedule` constructor as you'd do for
`tensorflow.keras.optimizers.schedules.deserialize`.
