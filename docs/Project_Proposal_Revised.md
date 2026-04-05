# Robust Wearable Next-Day Fatigue Inference Under Realistic Mobile Sensing Constraints

## Abstract
Wearable devices such as smartwatches have made large-scale continuous sleep monitoring possible outside clinical environments. However, deploying reliable Next-Day Fatigue inference directly on mobile hardware remains challenging due to real-world sensing constraints such as battery-saving duty cycling, OS scheduling gaps, and motion artifacts.

This project proposes a robust wearable fatigue inference pipeline that operates directly on devices like the Apple Watch. Using public physiological datasets and direct HealthKit extraction, which contains wrist accelerometer signals and photoplethysmography (PPG) heart-rate signals, we train a lightweight classification system to detect wake, light, deep, and REM sleep stages, and seamlessly use those stages to predict baseline user fatigue. To simulate realistic mobile conditions, we introduce controlled degradation including heart-rate dropout, sampling-rate reduction, and motion artifact noise. We then apply lightweight robustness strategies such as signal imputation and multi-resolution feature extraction to improve stability.

To satisfy requirements for interpretability and user interaction, the system features a dynamic Streamlit dashboard and a native iOS visualization interface. Our goal is to empirically demonstrate how robust on-device fatigue inference can maintain high accuracy while operating under realistic mobile sensing constraints, and to visually present these degradation thresholds.

## 1 Introduction and Motivation
Wearable devices have rapidly expanded the possibilities of continuous health monitoring. Smartwatches equipped with motion sensors and optical heart-rate monitors allow large-scale sleep and recovery monitoring. These systems can provide valuable insights into circadian rhythms and long-day fatigue outcomes.

However, translating predictive algorithms from controlled research environments to real-world mobile devices presents significant challenges. Mobile sensing pipelines must operate under practical constraints such as battery-saving modes that reduce sensor sampling rates, operating system scheduling that introduces intermittent data gaps, and user motion that introduces noise and artifacts.

Most existing predictive models assume continuous, high-quality sensor streams. In practice, wearable data is often incomplete or noisy, which can significantly degrade model performance. As a result, designing sleep-derived fatigue systems that remain reliable under realistic sensing conditions is an important systems problem.

This project addresses this challenge by building a robust wearable fatigue inference pipeline. By introducing controlled sensor degradation and evaluating lightweight robustness strategies, we hypothesize that by utilizing feature-level dropout and signal interpolation, our lightweight model can experience up to 30% metric dropout while maintaining a Macro-F1 score within 5% of a baseline, high-fidelity server model.

## 2 Related Work
Several studies have explored sleep-stage prediction using wearable sensor data. Walch et al. [1] demonstrated that Apple Watch accelerometer and heart-rate signals can be used to predict physiological states with reasonable accuracy. 

However, most prior work assumes continuous high-quality sensor streams. Shen et al. [3] highlighted significant robustness gaps in mobile inertial tracking systems caused by real-world sensing variability such as packet loss, sampling gaps, and orientation drift. On-device machine learning optimization has also been widely studied. Techniques such as DeepIoT [5] compress neural networks to run efficiently on resource-constrained sensing devices. 

Despite these advances, limited work has systematically evaluated Next-Day fatigue inference pipelines under realistic mobile sensing degradation. Our project aims to fill this gap by stress-testing wearable models under controlled sensor degradation scenarios and visualizing the outcomes through interactive UIs.

## 3 Proposed Method
Our proposed system processes raw physiological signals (heart rate, respiration, motion) to detect wake, light, deep, and REM sleep stages, and perform subsequent fatigue classification. The overall pipeline includes signal preprocessing, degradation simulation, robust feature extraction, and an interactive UI.

### 3.1 Preprocessing and Degradation Simulation
Raw signals are first filtered and segmented into night-window features suitable for sleep-recovery analysis. To simulate real-world mobile sensing conditions, we introduce controlled degradation including:
*   **Heart-rate dropout:** Simulating packet loss or sensor failures
*   **Sampling-rate reduction:** Representing battery duty-cycling constraints
*   **Noise Injection:** Representing real-world user movement artifacts

### 3.2 Robustness Strategies
To improve model stability under degraded sensing conditions, we introduce several lightweight robustness techniques:
*   Last-value carry-forward and linear imputation for missing metric segments
*   Multi-resolution feature extraction to handle reduced sampling rates
*   Decision-threshold calibration for edge environments

### 3.3 Sleep Stage & Fatigue Classifier
Once robust features are extracted, a lightweight classification pipeline (e.g., utilizing Random Forests) is deployed to consistently detect wake, light, deep, and REM sleep progression throughout the night, which then actively drives the Next-Day Fatigue prediction. The model is compressed to run directly on wearable hardware, allowing deployment on edge devices and iOS via CoreML/HealthKit integration.

### 3.4 Interactive UI and Visualization
To translate our backend robustness framework into an interpretable system, we are developing two interfaces:
1.  **A native iOS application:** Continuously fetches and processes live HealthKit sensing data.
2.  **An interactive web dashboard (Streamlit):** Allows users to dynamically test the model by visually simulating data dropouts in real-time and observing the resulting shifts in fatigue probability.

## 4 Evaluation Plan
To evaluate the effectiveness of our approach, we compare the robust pipeline against baseline inference models under varying levels of sensor degradation.

### 4.1 Experimental Setup
Experiments will be conducted using structured Apple Watch data arrays. Model inference performance will be tested locally on a dynamic dashboard and tested natively on iOS frameworks.

### 4.2 Evaluation Metrics & Visual Outcomes
The evaluation will focus on precise analytical and visual outcomes:
*   **Tracking Accuracy:** Macro-F1 score and Cohen’s kappa for fatigue classification.
*   **Resource Footprint:** Model size, latency per inference, and throughput.
*   **Robustness Degradation Curves:** We will present visual graphs charting accuracy decay against precise percentages of simulated data dropouts.
*   **UI-Driven Case Studies:** During the presentation, our interactive UI will be used to run live, visual comparisons showing the model's prediction stability when high-quality versus highly-degraded signals are routed into the system.

### 4.3 Baselines
The proposed robust model will be compared against baseline edge-based models without robustness mechanisms, highlighting the trade-offs between model robustness, computational efficiency, and deployment feasibility on mobile hardware.

## References
[1] O. J. Walch et al., “Sleep stage prediction with raw acceleration and photoplethysmography heart rate data,” Sleep, 42(12), 2019.
[2] O. J. Walch et al., “Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography,” PhysioNet, 2019.
[3] S. Shen, M. Gowda, and R. Roy Choudhury, “Closing the gaps in inertial motion tracking,” Proceedings of MobiCom, 2018.
[4] S. Yao et al., “DeepIoT: Compressing deep neural network structures.”
