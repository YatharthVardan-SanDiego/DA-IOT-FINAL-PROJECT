# DA-IOT-FINAL-PROJECT

**Course:** AAI-530 — Data Analytics and the Internet of Things
**Group:** AAI-530 Group 12
**Institution:** University of San Diego
**Dataset:** RT-IoT2022 (UCI ML Repository ID 942)

Intelligent IoT Intrusion Detection System (RT-IoT2022)

This repository contains an Intelligent Intrusion Detection System (IDS) designed for Industrial IoT (IIoT) environments using the RT-IoT2022 dataset. The project focuses on identifying and classifying malicious network activities—such as DDoS, Brute Force, and ARP Poisoning—that target common smart devices like IP cameras and Wipro bulbs. By analyzing network flow characteristics, this system provides a robust defensive framework for modern smart manufacturing and office infrastructures.

The technical implementation features two distinct machine learning architectures built from scratch using TensorFlow and Keras. The first is a Deep Neural Network (DNN) designed for high-accuracy multi-class attack classification (13 categories), while the second is an LSTM-based time-series model used to forecast network flow durations and identify temporal anomalies. All preprocessing scripts, model training histories, and prediction outputs are organized within the directory structure for full reproducibility.

Complementing the machine learning models is an interactive Tableau dashboard that visualizes the security posture of the network in real-time. This dashboard integrates our model predictions with exploratory data analysis to provide actionable insights into the "Network Heartbeat" and attack persistence. For a live view of the system’s performance, please refer to the linked Tableau Public workbook and the comprehensive project report included in this repository.

# Install dependencies
pip install ucimlrepo tensorflow scikit-learn pandas numpy

# Run the Attack Classifier
python models/model1_attack_classifier.py

# Run the Time-Series Predictor
python models/model2_flow_duration_lstm.py


## Team — AAI-530 Group 12

- Lokesh Upputri
- Yatharth Vardan
- Senthil Arasu T
---
## Citation

If referencing the dataset:

> Sharmila Kinnal, B., Khanum, F., Manzoor, U., Akhter, N., & Bhavani, R. (2023). *RT-IoT2022* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5P338
