# oeawai_challenge
Instrument Classfication Challenge for the OEAW AI Summer School:
https://www.kaggle.com/c/oeawai/

Our score in private board was 0.77231 (F1). It was achieved by an geometric average of the outputs of two models:

SpectralResNet18 (private score alone: 0.75906)
learns the MFC
trained for 9 epochs with Adam Optimizer, lr 0.0001, batch size 8 on GeForce GTX1080

MSResNet (private score alone: 0.71199)
learns the raw audio 
trained for 10 epochs with Adam Optimizer, lr 0.0001, batch size 8 on GeForce GTX1080
