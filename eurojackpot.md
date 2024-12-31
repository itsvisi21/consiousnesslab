### Mathematical Model for Lottery Draw Predictions

The following mathematical model combines randomness, balance, proximity, and novelty into a cohesive framework for generating lottery predictions.

---

### **Model Components**

#### **1. Randomness Factor (\( R \))**
- Measures the variability of the main numbers.
- \( R = \text{Standard Deviation of Main Numbers} \)

#### **2. Balance Factor (\( B \))**
- Represents the average value of the main numbers.
- \( B = \text{Mean of Main Numbers} \)

#### **3. Proximity Factor (\( P \))**
- Measures the overlap between the predicted draw and the last known draw.
- \( P = |\text{Main Numbers}_{\text{predicted}} \cap \text{Main Numbers}_{\text{last}}| + |\text{Euro Numbers}_{\text{predicted}} \cap \text{Euro Numbers}_{\text{last}}| \)

#### **4. Novelty Factor (\( N \))**
- Ensures the prediction is distinct from historical data.
- \( N = \begin{cases} 
1, & \text{if } (\text{Main Numbers, Euro Numbers}) \notin \text{Historical Data} \\
0, & \text{otherwise}
\end{cases} \)

#### **5. Feature Weights (\( w_R, w_B, w_P, w_N \))**
- Weights are assigned to each factor to balance their contribution:
  - \( w_R \): Weight for randomness.
  - \( w_B \): Weight for balance.
  - \( w_P \): Weight for proximity.
  - \( w_N \): Weight for novelty.

---

### **Objective Function**
The goal is to maximize the weighted sum of the factors:
\[
\text{Score} = w_R \cdot f_R(R) + w_B \cdot f_B(B) + w_P \cdot f_P(P) + w_N \cdot N
\]

Where:
- \( f_R(R) \): A function that normalizes randomness to a desired range.
- \( f_B(B) \): A function that normalizes balance to a desired range.
- \( f_P(P) \): A function that prioritizes proximity.

---

### **Model Steps**

1. **Normalize Factors**:
   Normalize \( R \), \( B \), and \( P \) based on historical ranges:
   \[
   f_R(R) = \frac{R - R_{\text{min}}}{R_{\text{max}} - R_{\text{min}}}
   \]
   \[
   f_B(B) = \frac{B - B_{\text{min}}}{B_{\text{max}} - B_{\text{min}}}
   \]
   \[
   f_P(P) = \frac{P}{P_{\text{max}}}
   \]

2. **Score Predictions**:
   Compute the score for each prediction using the objective function.

3. **Rank Predictions**:
   Rank predictions based on their scores and select the one with the highest score.

---

### **Application to Predictions**
For each prediction:
- Compute \( R \), \( B \), \( P \), and \( N \).
- Apply the objective function.
- Rank predictions based on the score.

Would you like to implement this model to evaluate the given predictions?