# AI Pose Tracking & Exercise Analysis

Python application for physical exercise analysis using real‑time pose estimation. The system detects human poses with YOLO, evaluates exercises (squats, jumps, punches, etc.) using biomechanical rules or LSTM models, counts repetitions, and provides visual feedback.

## 🎯 Goal

Provide a modular and extensible platform for:
- **Pose detection** in video/webcam using YOLO‑pose.
- **Exercise evaluation** through joint‑angle thresholds or sequence models (LSTM).
- **Repetition counting** and set tracking.
- **Data collection** for model training and later analysis.
- **UI presentation** with a real‑time dashboard.

Design prioritizes **Clean Code, SOLID, true modularity, and low coupling** between layers.

## ✨ Key Features

- **Multiple exercises**: squats, jumping jacks, skipping, punches, kicks, push‑ups, crunches.
- **Two evaluation modes**:
  - **Angle‑based evaluation**: biomechanical rules based on joint thresholds.
  - **LSTM‑based evaluation**: neural‑network models trained on pose sequences.
- **Layered architecture**:
  - Explicit contracts (`Protocol`, ABCs).
  - Dependency inversion.
  - Clear separation between domain, infrastructure, and UI.
- **Parallel pipeline**:
  - Processing on a dedicated thread (does not block UI).
  - Person tracking and registry across frames.
- **Data collection**:
  - CSV/JSONL export of poses, states, and repetitions.
  - Datasets ready for LSTM model training.
- **PySide6 UI**:
  - Dashboard with preview, counters, and feedback.
  - Hot‑reload configuration via YAML.

## 🏗️ Architecture

The code follows a strictly layered architecture:

```
src/core/           ← Contracts, types, cross‑cutting logic
src/exercises/      ← Exercise‑specific behavior
src/pipeline/       ← Technical frame processing and orchestration
src/presentation/   ← State adaptation to UI (presenters, viewmodels)
src/data_collection/← Data persistence and publishing
src/models/         ← External model access (YOLO)
src/tracking/       ← Repetition counting and state tracking
src/visualization/  ← Frame annotation rendering
```

**Pipeline flow** (`main.py` → `pipeline_thread.py`):
1. Frame capture (cv2.VideoCapture)
2. Pose detection (YOLO)
3. Person tracking (PersonRegistry)
4. Exercise processing (FrameProcessor)
5. Data collection (DataCollectionBus)
6. Rendering (Renderer)
7. UI presentation (DashboardPresenter)

## 🚀 How to run

### Requirements
- Python 3.11+
- PyTorch with CUDA support (recommended) or CPU‑only
- Dependencies listed in `requirements.txt`

### Quick installation
```bash
# 1. Clone repository
git clone <repo-url>
cd pose_2

# 2. Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install PyTorch with CUDA (if you have an NVIDIA GPU)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# 5. Download YOLO weights (if not already in models/weights/)
# Weights are downloaded automatically on first run.
```

### Run the application
```bash
python main.py
```
- Configuration is read from `config/app.yml`, `config/exercises.yml`, `config/model.yml`.
- Default camera is `source: 0` (webcam). Changeable in `app.yml`.
- To exit: `Ctrl+C` in terminal or close the window.

### Useful commands
```bash
# Run tests
pytest

# Train LSTM model for squats
python training/scripts/train_lstm.py

# Process session data into dataset
python training/scripts/data_processing.py

# Verify CUDA availability
python check_cuda.py
```

## ⚙️ Configuration

The application is configured via YAML files in `config/`:

- **`app.yml`** – General settings (window, FPS, data collection).
- **`exercises.yml`** – Per‑exercise parameters (thresholds, messages, evaluator type).
- **`model.yml`** – YOLO weights and detection parameters.

Example exercise configuration (squat):
```yaml
squat:
  enabled: true
  evaluator_type: lstm   # angle | lstm
  down_prob_threshold: 0.7
  up_prob_threshold: 0.3
  window_size: 30
  model_path: "models/lstm/squat_model.h5"
```

## 📁 Project structure

```
pose_2/
├── config/                 # YAML configuration
├── data/                  # Stored sessions (generated)
├── models/                # Model weights (YOLO and LSTM)
│   ├── weights/
│   └── lstm/
├── training/              # Training scripts and data
├── src/                   # Source code (layered architecture)
│   ├── core/             # Contracts, types, base evaluators
│   ├── exercises/        # Exercise‑specific logic
│   ├── pipeline/         # Frame processing, threading
│   ├── presentation/     # Presenters, adapters, viewmodels
│   ├── data_collection/  # Persistence (CSV, JSONL)
│   ├── models/           # Model wrappers (YOLO)
│   ├── tracking/         # Repetition counting and tracking
│   └── visualization/    # Dashboard rendering
├── tests/                # Unit tests
├── main.py               # Entry point and composition root
├── CLAUDE.md             # Extensive developer guide
└── README.md             # This file
```

## 🧪 Development

### Adding a new exercise
1. Create class in `src/exercises/` (e.g., `my_exercise.py`) implementing `IExerciseDetector`.
2. Add configuration in `config/exercises.yml`.
3. Register in `src/exercises/registry.py`.
4. Include in `src/pipeline/frame_processor.py` (`DETECTOR_CLASSES` list).
5. Optional: train an LSTM model if using sequence‑based evaluator.

### Code conventions
- Type hints in public APIs.
- Short functions, small classes, single responsibility.
- Depend on interfaces (`Protocol`, ABC) not concrete implementations.
- Errors should communicate cause and context.
- Tests for behavior changes.

### Testing
```bash
# Run all tests
pytest

# Run a specific test
pytest tests/test_csv_collector.py::TestCsvCollector::test_creates_file_with_header
```

## 📚 Resources

- **CLAUDE.md** – Detailed architecture, commands, and development guide.
- **`.claude/rules/`** – Layer‑specific rules (architecture, style, interfaces, exercises, etc.).
- **Ultralytics YOLO documentation**: https://docs.ultralytics.com/
- **PySide6**: https://doc.qt.io/qt-6/pyside6.html

## 📄 License

[Include project license here]

---

*Project oriented toward biomechanics research, fitness tech, and real‑time computer‑vision applications.*