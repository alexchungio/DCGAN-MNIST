# DCGAN-MNIST
Try to use Deep Convolutional Generative Adversrial Network(DCGAN) to generate images of hand written digits


## train logs
### dropout=0
```shell script
dropout_rate = 0.0
generator_learning_rate = 2e-4
discriminator_learning_rate = 2e-4
epoch = 40
```

![acc](docs/generator_loss_dropout_00.png) | ![acc](docs/discriminator_loss_dropoout_00.png) |
|:-------------------------:|:-------------------------:|
Displayed generator Loss on Tensorboard | Displayed discriminator Loss on Tensorboard | 

![acc](docs/dcgan_mnist_dropout_00.gif) |
|:-------------------------:|
every epoch generator result of training | 

### dropout=0.5
```shell script
dropout_rate = 0.5
generator_learning_rate = 1e-4
discriminator_learning_rate = 4e-4
epoch = 50
```

![acc](docs/generator_loss_dropout_05.png) | ![acc](docs/discriminator_loss_dropoout_05.png) |
|:-------------------------:|:-------------------------:|
Displayed generator Loss on Tensorboard | Displayed discriminator Loss on Tensorboard | 

![acc](docs/dcgan_mnist_dropout_05.gif) |
|:-------------------------:|
every epoch generator result of training | 