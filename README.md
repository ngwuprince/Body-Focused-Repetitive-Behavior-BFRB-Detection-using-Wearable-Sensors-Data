# Body-Focused Repetitive Behavior (BFRB) Detection using Wearable Sensors

## Project Overview

This project explores the application of machine learning, specifically deep learning, to detect Body-Focused Repetitive Behaviors (BFRBs) using data collected from a custom-built wrist-worn device called Helios. BFRBs, such as hair pulling (trichotillomania) and skin picking (excoriation disorder), are often subtle and highly individualized, making their accurate detection a significant challenge. This work aims to leverage rich, multi-modal sensor data to distinguish BFRB-like gestures from everyday actions, paving the way for smarter mental health monitoring and early intervention tools.

## Problem Statement

Accurately identifying BFRBs is crucial for effective diagnosis and treatment. However, the inherent subtlety of these behaviors and their resemblance to common, innocuous movements (e.g., scratching, adjusting glasses) pose a considerable hurdle for automated detection. Traditional methods often fall short in capturing the nuanced patterns indicative of BFRBs. This project addresses this challenge by utilizing advanced sensor technology to capture detailed physiological and motion data, enabling a more precise differentiation between BFRB and non-BFRB gestures.

## Project Goal

The primary goal is to develop a robust predictive model capable of identifying BFRBs from time-series sensor data. This involves:

*   **Distinguishing BFRB-like gestures from non-BFRB everyday gestures**: Utilizing data from inertial (IMU), temperature (thermopile), and proximity (Time-of-Flight) sensors.
*   **Classifying specific BFRB types**: Differentiating between various clinically distinct BFRB gestures.
*   **Evaluating sensor efficacy**: Assessing the incremental value of thermopile and Time-of-Flight sensors beyond traditional IMU data to inform future wearable device development for mental health applications.

## Dataset Description

The dataset originates from a Kaggle competition and comprises two main components:

### Sensor Data (`train.csv` / `test.csv`)

This time-series data captures sensor readings for various gestures. Key columns include:

*   `sequence_id`: Unique identifier for each gesture sequence.
*   `sequence_counter`: Row index within a sequence, representing synchronized sensor readings at a specific timestamp.
*   `acc_[x/y/z]`: Linear acceleration data from the IMU (m/s²).
*   `rot_[w/x/y/z]`: Quaternion components representing 3D orientation from the IMU.
*   `thm_[1-5]`: Temperature readings from five thermopile sensors (°C).
*   `tof_[1-5]_v[0-63]`: 8x8 pixel Time-of-Flight sensor readings (distance), where -1 indicates no reflection detected.

**Data Structure**: Each `sequence_id` represents a single gesture trial, typically structured as `Transition → Pause → Gesture`.

**Train-only Columns**:

*   `gesture`: The actual BFRB or non-BFRB gesture label.
*   `behavior`: The phase of the gesture (e.g., `Transition`, `Pause`, `Gesture`).
*   `sequence_type`: Categorization as `Target` (BFRB-like) or `Non-Target` (non-BFRB-like).
*   `orientation`: Subject's physical posture during data collection.
*   `subject`: Anonymized participant ID.

### Demographic Data (`train_demographics.csv` / `test_demographics.csv`)

This dataset provides participant-level metadata that can be used to enhance model performance:

*   `age`, `sex`, `adult_child`, `handedness`
*   `height_cm`, `shoulder_to_wrist_cm`, `elbow_to_wrist_cm`

**Note on Missing Data**: Some sensor values may be missing due to hardware communication issues. Specifically, Time-of-Flight sensor missingness is encoded as -1. The hidden test set for the competition is expected to have IMU values mostly filled, while other sensors (thermopile, ToF) may be missing for approximately 50% of sequences, necessitating robust handling of missing data.

## Evaluation Metrics

The model's performance is evaluated using an **averaged F1-score** across two classification tasks, as defined in the original notebook:

1.  **Binary Classification F1-Score**: Measures the ability to distinguish between BFRB-like (target) and non-BFRB-like (non-target) gestures.
    $$\text{F1}_{\text{binary}} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$ 
    Where: 
    *   **Precision** = $\frac{\text{TP}}{\text{TP} + \text{FP}}$ 
    *   **Recall** = $\frac{\text{TP}}{\text{TP} + \text{FN}}$

2.  **Multi-Class F1-Score (Macro)**: Averages the F1-score equally across all gesture classes, with all non-BFRB gestures collapsed into a single `non_target` class.
    $$\text{F1}_{\text{macro}} = \frac{1}{C} \sum_{i=1}^{C} \text{F1}_i$$ 
    Where: 
    *   $C$ is the number of gesture classes (target gestures + 1 non-target class).
    *   $\text{F1}_i$ is the F1-score for class $i$.

**Final Score Calculation**: The final score is the average of the Binary F1-Score and the Macro F1-Score.

$$\text{Final Score} = \frac{1}{2} \left( \text{F1}_{\text{binary}} + \text{F1}_{\text{macro}} \right)$$

## Exploratory Data Analysis (EDA) and Feature Engineering

The provided notebook primarily focuses on a comprehensive EDA and initial feature engineering steps. Key aspects include:

*   **Data Loading and Initial Checks**: Verifying dataset integrity, identifying duplicates, and examining data types.
*   **Missing Value Analysis**: Detailed examination of missing data patterns, particularly for thermopile and ToF sensors, and handling the -1 encoding for missing ToF readings.
*   **Feature Engineering**: Deriving new, informative features from raw sensor data, such as:
    *   **Accelerometer Magnitude**: Calculated as $\sqrt{\text{acc_x}^2 + \text{acc_y}^2 + \text{acc_z}^2}$, providing a scalar measure of motion intensity.
    *   **Rotation Angle**: Derived from the quaternion `rot_w` component using $2 \cdot \arccos(\text{rot_w})$, quantifying wrist twist.
    *   **Thermopile and ToF Summaries**: Aggregating mean and standard deviation for each thermopile sensor and mean pixel values for ToF sensors across sequences.
*   **Visualizations**: Extensive use of histograms, box plots, and time-series plots to visualize data distributions, identify outliers, and understand the temporal dynamics of sensor readings for different gestures.

## Insights from EDA

*   **Multi-Modal Data Richness**: The combination of IMU, thermopile, and ToF sensors provides a rich, complementary data source for activity recognition. IMU captures motion, thermopile indicates skin contact and location, and ToF provides spatial proximity information.
*   **Missing Data Challenges**: Significant missingness in thermopile and ToF data (especially -1 values in ToF) highlights the need for robust imputation or model architectures that can handle sparse inputs.
*   **Feature Importance**: The engineered features (e.g., accelerometer magnitude, rotation angle) are crucial for capturing the dynamic aspects of gestures and are expected to be highly predictive.
*   **Behavioral Nuances**: Visualizations reveal distinct patterns for BFRB-like gestures compared to non-BFRB actions, suggesting that a well-trained model can differentiate these subtle behaviors.

## Deep Learning Model (Future Work)

While the provided notebook focuses on EDA and feature engineering, the ultimate goal is to develop a deep learning model capable of processing the time-series sensor data. Potential architectures could include:

*   **Recurrent Neural Networks (RNNs)**: Such as LSTMs or GRUs, well-suited for sequential data to capture temporal dependencies in sensor readings.
*   **Convolutional Neural Networks (CNNs)**: For extracting local patterns and features from the time-series data, potentially in combination with RNNs (e.g., ConvLSTM).
*   **Transformer Networks**: Given their success in sequence modeling, transformers could be explored for their ability to capture long-range dependencies and complex interactions across different sensor modalities.
*   **Multi-modal Fusion**: Strategies for effectively combining features from IMU, thermopile, and ToF sensors (e.g., early, late, or hybrid fusion) will be critical for optimal performance.


### Additional Insights from EDA (Non-Target Key Insights)

Further analysis of the dataset revealed several key insights regarding the characteristics of the sensor data and demographic information:

*   **Sequence Length (`sequence_counter`)**: Gestures exhibit wide variations in duration, necessitating models capable of handling variable-length inputs rather than assuming fixed windows. This highlights the importance of time-series modeling approaches.
*   **Anthropometric Data (`height_cm`, `shoulder_to_wrist_cm`, `elbow_to_wrist_cm`)**: Participant height and arm segment lengths (shoulder-to-wrist, elbow-to-wrist) show considerable ranges. These anthropometric variations can influence wrist-motion amplitudes and spatial volumes occupied by gestures, directly impacting IMU and ToF sensor readings. For instance, forearm length directly limits how far the wrist can travel toward the face in gestures.
*   **Accelerometer Magnitude (`acc_mag_mean`, `acc_mag_std`, `acc_mag_max`)**: The mean accelerometer magnitude consistently hovers around 10 m/s², indicating a baseline motion level. The standard deviation reveals the spread from near-static to very jerky sequences, while maximum values capture extreme motion peaks, such as forceful tugs characteristic of some BFRBs.
*   **Rotation Angle (`rot_angle_mean`, `rot_angle_max`)**: The mean rotation angle is approximately 2.4 radians, representing typical wrist tilt during gestures. Maximum values approaching π radians demonstrate full wrist twists in certain cases, providing crucial information for distinguishing complex movements.
*   **Thermopile Sensor Readings (`thm_*_mean`, `thm_*_std`)**: Mean thermopile readings of 27–28 °C are consistent with gestures performed near the skin. Near-zero values indicate the wrist is far from warmth (e.g., during transition or pause phases). Occasional spikes to 31–32 °C in the test set suggest closer or warmer contact points. High standard deviations (up to ~5 °C) in thermopile data mark strong shifts between 


contact vs. air,” which is characteristic of contact gestures like hair pulls or cheek pinches.
*   **Time-of-Flight (ToF) Sensor Readings (`tof_*_mean`, `tof_*_std`)**: ToF values span a full range (0–250), with near-zero indicating direct contact and higher values reflecting “air” gestures. High standard deviations (up to ~140) in the training set suggest rapid in-and-out motions, typical of BFRB tugs, while lower standard deviations in the test set might imply smoother or steadier distances.
*   **Handedness**: Approximately 88% of participants in the training set are right-handed, suggesting minimal variation expected from this factor, though its influence on sensor readings (e.g., which thermopile faces the target first) is noted.

### Target Key Insights (`gesture`)

Analysis of the `gesture` target variable reveals important characteristics for model development:

*   **Multimodal, Nearly Uniform Spread Across Many Classes**: The training set contains 18 distinct gestures, with no single class dominating more than ~14% of samples. High-frequency activities like `Text on phone` and `Neck – scratch` each account for about 10% of the data, while many mid-frequency gestures (e.g., `Eyebrow – pull hair`, `Forehead – scratch`) hover around 7–8%. Lower-frequency gestures (e.g., `Write name on leg`, `Pinch knee/leg skin`) are in the 1.5–2% range, indicating a long tail of rarer events. This implies that a robust model must handle a fine-grained, multi-class ratio without overfitting to common gestures, potentially requiring class-balanced sampling or weighted loss to ensure rare gestures are learned effectively.
*   **BFRB vs. Non-BFRB Labeling**: Half of the gesture classes are BFRB-like (e.g., `Forehead – pull hairline`, `Cheek – pinch skin`), and the other half are non-BFRB everyday actions (e.g., `Drink from bottle/cup`, `Wave hello`). The distribution between BFRB and non-BFRB is fairly balanced when collapsed to a binary classification, suggesting that the initial binary classification stage (target vs. non-target) should be robust without extreme skew. This also implies a two-stage approach: first, binary classification (BFRB vs. non-BFRB), followed by multi-class refinement to differentiate subtle behaviors.

### Feature in Consideration: Handedness

Handedness significantly influences how sensor modalities respond to gestures. For instance, when comparing left-handed versus right-handed subjects performing the same gesture (e.g., “Write name on leg”):

*   **IMU (acc_x/y/z)**: Handedness changes which acceleration axis peaks first or higher. A right-handed “write” might show a larger positive `acc_x` (lateral) as the wrist pivots differently than a left hand. Models can learn to swap or re-align axis information based on handedness to avoid misinterpretation.
*   **Rotation (rot_w/x/y/z)**: A tilt bias may be observed; right-handed users might rotate the wrist inward (e.g., `rot_y` ramp earlier), while left-handed users might rotate outward first. A slower `rot_angle` rise on one side signals how the forearm twists, informing temporal alignment to prevent handedness-based misclassification.
*   **Thermopiles (thm_1–5)**: Handedness affects which thermopile channel spikes first as skin heat enters the 50° FoV. For right-handed individuals, a specific thermopile index (e.g., `thm_2` on the inner wrist) may warm earlier than in left-handed sequences. This allows the algorithm to normalize which thermopile to prioritize based on reported handedness.
*   **ToF Mean Distance (tof_1–5)**: Handedness influences which ToF sensor’s mean distance drops earliest. A right-handed user’s index “front” ToF (e.g., `tof_3_mean`) might register proximity sooner than the left, and vice-versa. Training on these systematic ToF timing offsets can prevent overfitting to one handedness.

### Feature in Consideration: Adult vs. Child

Comparing gestures performed by adults versus children (e.g., “Neck – pinch skin”) reveals distinct patterns:

*   **Acceleration Magnitude**: Children often exhibit more abrupt movements, leading to spikier and larger amplitude `acc_mag` peaks. The model should account for this, understanding that high-magnitude IMU bursts do not always indicate a BFRB but could reflect a child’s natural vigor.
*   **Rotation Angle**: Adults might maintain a steadier rotation during pauses, whereas children’s `rot_angle` might oscillate more. This informs the model to treat minor oscillations differently based on the subject’s age.
*   **Thermopile**: Children’s thermopile values may rise more sharply (indicating closer contact) but also cool off faster. Integrating this helps the model differentiate between genuine contact and faster rebounds typical of children.
*   **ToF**: If a child’s ToF mean distances drop sooner (due to hands being closer), the model can learn to adjust distance thresholds based on age, preventing misclassification of a child’s large motion as a different gesture.



## Deep Learning Model Architecture and Training

The architecture is a **Two-Branch HAR Model** designed to process multi-modal sensor data effectively. It incorporates several advanced neural network components:

### Model Architecture Overview

The model processes IMU (Inertial Measurement Unit) data and combined Time-of-Flight (ToF) and Thermopile data through separate branches before merging them for final classification.

1.  **IMU Branch**: This branch handles `acc_[x/y/z]` and `rot_[w/x/y/z]` features. It consists of:
    *   **Residual SE Blocks**: Two `ResidualSEBlock` layers are used. Each block includes 1D Convolutional layers (`nn.Conv1d`), Batch Normalization (`nn.BatchNorm1d`), ReLU activation, and a Squeeze-and-Excitation (SE) Block. The SE Block dynamically recalibrates channel-wise feature responses, enhancing the model's ability to focus on important sensor signals. Residual connections help in training deeper networks, and Max Pooling (`nn.MaxPool1d`) reduces sequence length.

2.  **TOF/Thermal Branch**: This branch processes `thm_[1-5]` and `tof_[1-5]_v[0-63]` features. It uses:
    *   **Convolutional Layers**: Two pairs of `nn.Conv1d` layers followed by Batch Normalization, ReLU activation, and Max Pooling. These layers are designed to extract relevant patterns from the temperature and proximity sensor data.
    *   **Dropout**: Applied after each pooling layer to prevent overfitting.

3.  **Feature Merging**: The outputs from the IMU and TOF/Thermal branches are concatenated along the channel dimension, creating a merged feature representation that combines information from all sensor modalities.

4.  **Bi-directional LSTM (BiLSTM)**: A `nn.LSTM` layer processes the merged features. Being bidirectional, it captures temporal dependencies in both forward and backward directions, which is crucial for understanding the sequential nature of gestures.

5.  **Attention Mechanism**: An `Attention` layer is applied to the BiLSTM output. This mechanism allows the model to selectively focus on the most relevant parts of the sequence, assigning higher weights to time steps that are more indicative of a BFRB event.

6.  **Dense Head**: The output from the attention layer is fed into a series of fully connected (dense) layers (`nn.Linear`) with Batch Normalization, ReLU activation, and Dropout. This final part of the network maps the learned features to the output classes.

7.  **Output Layer**: A final `nn.Linear` layer produces the raw scores (logits) for each of the `num_classes` (different gesture types).

### Training Setup

*   **Optimizer**: Adam optimizer with a learning rate of 1e-3 and a weight decay of 1e-4 for L2 regularization.
*   **Learning Rate Scheduler**: `CosineAnnealingWarmRestarts` is used to dynamically adjust the learning rate during training, mimicking a cosine decay with warm restarts to help the model escape local minima.
*   **Loss Function**: A custom `soft_cross_entropy` function is implemented to handle soft labels, which are particularly useful when using techniques like Mixup.
*   **Mixup Augmentation**: Applied during training to generate synthetic training examples by linearly interpolating between pairs of samples and their labels. This technique helps improve model generalization and robustness.
*   **Early Stopping**: Implemented to monitor validation loss and stop training if there is no improvement for a specified number of epochs (`patience = 10`), preventing overfitting.
*   **Class Weights**: Balanced class weights are computed and implicitly handled by the mixup soft labels, addressing potential class imbalance in the dataset.

### Data Preprocessing for Model Input

Before feeding into the model, the sensor data undergoes several preprocessing steps:

*   **Label Encoding**: Gesture labels are encoded into numerical representations.
*   **Feature Selection**: Metadata columns are excluded, and features are separated into IMU and TOF/Thermal groups.
*   **Missing Value Imputation**: Missing values are handled by forward-fill, backward-fill, and then filling any remaining NaNs with zeros.
*   **Standardization**: `StandardScaler` is fitted on all training data and applied to normalize feature values.
*   **Sequence Padding/Truncation**: Sequences are grouped by `sequence_id`, and then padded or truncated to a uniform length (90th percentile of sequence lengths) using `tensorflow.keras.preprocessing.sequence.pad_sequences`.
*   **One-Hot Encoding**: Labels are one-hot encoded for compatibility with the `soft_cross_entropy` loss and Mixup.
*   **Train/Validation Split**: Data is split into training and validation sets with stratification to maintain class distribution.
*   **PyTorch Dataset and DataLoader**: Custom `SequenceDataset` and `DataLoader` are used to efficiently manage data loading and batching for PyTorch training.

### Feature in Consideration: Sex (Female vs. Male)

Differences in sensor patterns between male and female subjects for a given gesture (e.g., “Forehead – scratch”) can arise from variations in average muscle strength, hand size, or gesture style. The model should account for these:

*   **IMU – Acceleration**: If males consistently exhibit higher acceleration peaks, the model can normalize IMU magnitude or learn sex-specific calibration.
*   **Rotation Angle**: Different wrist rotation profiles between sexes suggest the classifier should not over-rely on a particular rotation pattern without considering sex.
*   **Thermopile**: Female subjects often have slightly different baseline skin temperatures. The model should learn relative changes rather than absolute values for robustness.
*   **ToF**: Male subjects’ hand shapes may produce different ToF cluster sizes. Incorporating sex helps the model adjust distance thresholds and spatial patterns, reducing misclassifications.

### Feature in Consideration: Shoulder-to-Wrist Length

Shoulder-to-wrist length influences how subjects perform gestures and how sensors respond. For example, for a gesture like “Eyelash – pull hair”:

*   **IMU**: Longer-armed subjects may have smoother `acc_mag` curves due to slower angular acceleration. The classifier can learn length-normalized features to avoid penalizing smoother acceleration patterns.
*   **Rotation Angle**: Long-armed subjects might exhibit shallower `rot_angle` peaks. The model’s temporal filters should adjust for this to avoid confusing “slower rotations” with a different gesture type.
*   **Thermopile**: A longer arm may cause thermopile sensors to detect heat slightly later. Training the model on both arm-length groups prevents mislabeling a delay as “no contact.”
*   **ToF**: The initial ToF distance drop for a long-armed subject may occur at a higher distance. By learning length-informed thresholds, the model reduces false negatives when contact occurs farther away.


## Getting Started

To replicate the EDA and feature engineering steps:

1.  Clone this repository.
2.  Ensure you have the necessary Python libraries installed (e.g., `pandas`, `numpy`, `matplotlib`, `seaborn`).
3.  Download the dataset from the Kaggle competition ([BFRB Project on Kaggle](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data)).
4.  Run the `sensor-pulse-viz-eda-for-bfrb-detect.ipynb` notebook.

## Author
Ogbonna Ngwu. 
