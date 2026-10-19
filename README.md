# 🛺 Coimbatore RideFlow AI: End-to-End Intelligent Ride Optimization System
### Transforming Urban Mobility Operations through Machine Learning and Deep Learning

This real-time enterprise intelligence dashboard is fully operational and hosted on Streamlit Cloud. To interact with the predictive layers, modify transit bottlenecks, or evaluate dynamic dispatch workflows, use the gateway link below:

🚀 **[Launch Live Production RideFlow AI Dashboard Here](https://streamlit.app)**

---

🏢 Business Context & Problem Statement
In fast-growing Tier-2 urban economic hubs like Coimbatore, localized transit networks suffer from high operational inefficiencies. Informal auto-rickshaw networks experience volatile, non-standardized ride pricing, frequent driver cancellations due to fare friction, and unpredictable travel delays. 

* **The Challenge:** Standard ride-hailing pricing architectures are built on uniform city-wide assumptions. They fail to address hyper-local constraints, such as localized monsoon patterns, academic peak shifts (e.g., tech corridor surges in Saravanampatti), or informal behavioral friction.
* **The Solution:** This project delivers an End-to-End Intelligent Fleet Optimization System. By deploying concurrent machine learning engines alongside a spatiotemporal deep learning sequence node, the framework balances vehicle fares, flags dispatch failure risks, and projects traffic-aware wait times—requiring zero hardware infrastructure dependencies.

---

🏗️ Project Lifecycle & Methodology

### Phase 1: Data Strategy & Domain-Knowledge Simulation
Because actual telemetry and booking logs are highly confidential corporate assets, a structural simulation pipeline was architected to generate a production-ready dataset containing **50,000 transaction records**. 
* **Real-World Waves:** Integrated non-linear peak hours matching industrial shifts and regional college timetables.
* **Environmental Constraints:** Factored in seasonal monsoon disruptions and evening tariff metrics across core operational zones (Saravanampatti, Gandhipuram, Peelamedu, RS Puram, Ukkadam).
* **Deterministic Friction:** Programmed deliberate data anomalies—injecting structural null indicators and extreme physical coordinate errors—to rigorously test system resilience.

### Phase 2: Data Preprocessing & Anomaly Resolution
* **Localized Spatial Imputation:** Developed an enterprise cleaning pipeline where missing target metrics were resolved using descriptive spatial median grouping matrices relative to targeted zone routes.
* **Anomalous Coordinates Fix:** Out-of-bounds geographic indicators (such as structural error codes) were programmatically audited and mapped back to verified metrics using a static regional transit matrix.

### Phase 3: Exploratory Data Analysis (EDA) Insights
* Isolated distinct bimodal booking spikes aligned with the morning student/office rush (07:00–09:00) and evening market peaks (17:00–21:00).
* Discovered logarithmic fare expansions during monsoon anomalies driven by a heavy contraction in active driver supply pools.
* Verified that driver-initiated extra-cash demands are the primary operational vector causing ride cancellation spikes.

### Phase 4: Feature Engineering Pipeline
* Deconstructed temporal timestamps into discrete mathematical variables (Hour of Day, Day of Week).
* Transformed categorical attributes into **One-Hot Encoded sparse matrices** to maintain shape consistency between real-time inference arrays and model inputs, completely avoiding runtime dimensional mismatch errors.

### Phase 5: Machine Learning Core Development
* **Dynamic Tariff Surge Modeler (Linear Regression):** Built an optimization engine trained on supply-demand variance indices and weather shock flags to calculate dynamic price surge multipliers on the fly.
* **Operational Dispatch Classifier (Random Forest):** Deployed an ensemble bagging tree architecture to catch behavioral variables, evaluate transit risk factors, and output explicit ride cancellation probabilities.

### Phase 6: Deep Learning Spatiotemporal Node (LSTM)
* **Recurrent Sequence Network:** Rather than relying on simple static routing formulas, the system integrates a recurrent Deep Learning **LSTM (Long Short-Term Memory)** network. This node processes temporal markers, zone coordinates, and environmental arrays sequentially. By analyzing non-linear traffic delays over time, it outputs highly reliable travel time projections (ETA minutes).

### Phase 7: UI Integration & Cloud Infrastructure
* Designed a professional **Operations Control Room Dashboard** using Streamlit, featuring real-time input sliders and dynamic diagnostic alert blocks.
* Deployed the interface on **Streamlit Community Cloud**, utilizing array-flattening and memory caching techniques to ensure fast, low-latency globally accessible model execution.

---

🛠️ Technical Stack

| Category | Tools & Technologies |
| :--- | :--- |
| **Core Architecture** | End-to-End Predictive Analytics, Time-Series Feature Pipelines |
| **Machine Learning** | Scikit-Learn (Linear Regression, Random Forest Ensemble, Joblib Deserialization) |
| **Deep Learning** | TensorFlow, Keras (LSTM Recurrent Neural Networks, Sequential Layers) |
| **Data Pipelines** | Pandas, NumPy, Spatial Median Imputation, Coordinate Lookup Matrices |
| **Web UI Framework** | Streamlit Component Interface |
| **Cloud Deployment** | Streamlit Community Cloud Infrastructure |

---

📊 Key Features & System Evaluation Metrics

The production system has been stress-tested across three specific edge cases to confirm mathematical and operational consistency:

1. **Standard Operations Baseline:** Under clear weather and stable driver supply, the machine learning engine restricts surge multipliers near `1.0x` and keeps cancellation risks low, prompting a green **`🟢 OPTIMAL ROUTE`** clearance flag.
2. **Environmental Monsoon Shock:** When heavy rain triggers low vehicle availability, the pipeline automatically processes a flat `+₹50` monsoon premium coupled with a regression multiplier peak. Simultaneously, the deep learning LSTM node lengthens the travel window sequentially based on traffic density layers.
3. **Behavioral Anomaly Safeguard:** When a driver requests extra cash, the Random Forest Classifier flags this operational friction immediately. The system shifts into a red alert state (**`🚨 HIGH DISPATCH RISK`**) when cancellation probabilities exceed 50%, enabling operators to intervene manually before a dispatch fails.

---

📦 Local Station Setup
To run this intelligent fleet optimization workspace locally:
1. Clone this repository onto your terminal.
2. Install the library configurations listed in the `requirements.txt` file.
3. Boot the native deployment layers using the command: `streamlit run ride.py`
