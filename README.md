# TCC

This folder contains the files and database for my undergraduation final project (February 2016) entitled "Sistema de verificação de locutor na Raspberry Pi empregando modelos de misturas de gaussianas" (in English: An automatic speaker verification (ASV) system embedded on a Raspberry Pi board, employing Gaussian mixture models (GMM)). The project consisted of building an ASV system using GMM and mel-frequency cepstrum coefficients (MFCC) as model parameters, all embedded in a Raspberry Pi board connected to a microphone, an LED-display and a breadboard with auxiliary circuitry. The system is standalone and starts working as soon as it is powered.

Main files:
- ASV.py: main functions for automatic speaker verification, parameter extraction and GMM.
- RPIO.py: functions for dealing with the Raspberry Pi IO.
- atboot.py: Python script to be executed at the boot of the RPi board and automatically start the ASV routine.
