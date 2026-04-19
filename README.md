# Weed Detection System

An advanced weed detection and removal system using computer vision and autonomous navigation for agricultural robotics.

## 📋 Project Overview

This project implements a complete weed detection and management pipeline combining:
- **Deep Learning Detection**: YOLO-based weed classification using DeepWeeds and PlantSeedlings datasets
- **Autonomous Navigation**: A* path planning and robot navigation simulation
- **Precision Spraying**: Automated selective herbicide application based on detection results
- **Heat Map Analysis**: Visual representation of weed distribution across fields
- **Web Visualization**: Interactive 3D farm simulation interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/25hency/Weed.git
cd "Weed Detection"
```

2. **Create a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Project

1. **Prepare the dataset**
```bash
python src/detection/prepare_dataset.py
```

2. **Train the YOLO model**
```bash
python src/detection/train_yolo.py
```

3. **Run the detection pipeline**
```bash
python main.py
```

4. **View the simulation (Optional)**
```bash
# Serve the web visualization
python -m http.server 8080
# Open http://localhost:8080 in your browser
```

## 📁 Project Structure

```
Weed Detection/
├── main.py                 # Main entry point
├── config/                 # Configuration files
│   ├── dataset_summary.json
│   └── weed_dataset.yaml
├── dataset/               # Datasets (excluded from git)
│   ├── DeepWeeds/
│   ├── PlantSeedlings/
│   └── yolo_combined/
├── src/                   # Source code
│   ├── core/             # Core messaging and node architecture
│   ├── detection/        # Weed detection (YOLO training & inference)
│   ├── heatmap/          # Heat map generation
│   ├── navigation/       # A* path planning
│   ├── sensors/          # Sensor simulation
│   ├── simulation/       # Farm simulation and robot control
│   └── spraying/         # Spray application logic
├── models/               # Trained model weights (excluded from git)
├── outputs/              # Results and analysis (excluded from git)
├── evaluation/           # Evaluation scripts
├── visual_simulation/    # Web-based farm visualization
└── README.md            # This file
```

## 🔧 Main Components

### Detection Module (`src/detection/`)
- **train_yolo.py**: Trains YOLO model on weed detection
- **detector_node.py**: Real-time detection inference
- **prepare_dataset.py**: Dataset preparation and augmentation

### Simulation (`src/simulation/`)
- **farm_world.py**: Simulates farm environment and weed distribution
- **robot.py**: Robot agent with navigation and spraying capabilities

### Navigation (`src/navigation/`)
- **astar_planner.py**: A* pathfinding for optimal robot routes

### Spraying (`src/spraying/`)
- **sprayer_node.py**: Precision spray control and application

### Visualization (`visual_simulation/`)
- Interactive 3D farm simulation
- Real-time weed detection visualization
- Robot movement tracking

## 📊 Configuration

Edit `config/weed_dataset.yaml` to customize:
- Dataset paths
- Model parameters
- Training hyperparameters
- Simulation settings

## 🎯 Usage Examples

### Basic Detection
```python
from src.detection.detector_node import DetectorNode

detector = DetectorNode()
results = detector.detect(image_path="path/to/image.jpg")
```

### Farm Simulation
```python
from src.simulation.farm_world import FarmWorld
from src.simulation.robot import Robot

world = FarmWorld(width=100, height=100)
robot = Robot(x=50, y=50, world=world)
robot.navigate_and_spray()
```

## 📈 Outputs

The system generates:
- `outputs/ablation_results.json` - Model ablation study results
- `outputs/simulation_results.json` - Simulation metrics
- `outputs/visual_data.json` - Visualization data
- `outputs/figures/` - Generated plots and images

## 🔬 Datasets

This project uses publicly available datasets:
- **DeepWeeds**: Deep-learning benchmark for weed detection
- **PlantSeedlings**: Plant seedling classification dataset

Datasets are excluded from git but can be downloaded from:
- DeepWeeds: https://github.com/AlexOlsen/DeepWeeds
- PlantSeedlings: https://www.kaggle.com/c/plant-seedlings-classification

## 🛠️ Development

### Adding New Features
1. Create a new node in `src/`
2. Inherit from `NodeBase` in `src/core/node_base.py`
3. Implement message passing using `message_bus.py`
4. Register in the main pipeline

### Running Tests
```bash
pytest evaluation/
```

## 📝 Configuration Files

- `config/weed_dataset.yaml` - Dataset and training configuration
- `config/dataset_summary.json` - Dataset statistics

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Authors

- **Hency** - Project Lead

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Last Updated**: April 2026
