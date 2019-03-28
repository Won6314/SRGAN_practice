# SRGAN_practice
This is the SRGAN implementation in pytorch with DIV 2K dataset



# RUN
you can run my code with div 2k data.
I saved it with .raw data and loaded with my own utils
[https://www.dropbox.com/sh/n22ytq7ovbyhzzg/AADaOGVp7V72sQksUlrs4-FZa?dl=0]

here's div2k raw data, you can download it and run with this data.
only thing you should do is change 'path_root', and move raw data to that directory.



# Training
overall 36000 iterations.
trained 34000 iterations with vgg loss, and trained last 2000 iterations without vgg loss to reduce color difference
trained with about 200 Generator/Discriminator iteration ratio

This is what I got

![alt_text](https://github.com/Won6314/SRGAN_practice/blob/master/images/bear_LR.png)
![alt_text](https://github.com/Won6314/SRGAN_practice/blob/master/images/bear_SR.png)
![alt_text](https://github.com/Won6314/SRGAN_practice/blob/master/images/rhino_LR.png)
![alt_text](https://github.com/Won6314/SRGAN_practice/blob/master/images/rhino_SR.png)
![alt_text](https://github.com/Won6314/SRGAN_practice/blob/master/images/turtle_LR.png)
![alt_text](https://github.com/Won6314/SRGAN_practice/blob/master/images/tutle_SR.png)
