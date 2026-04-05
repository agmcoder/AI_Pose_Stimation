# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
```bash
python main.py
```
- Uses configuration from `config/app.yml`, `config/exercises.yml`, `config/model.yml`
- Source camera can be changed in `app.yml` (`source: 0` for webcam)
- Requires CUDA-compatible PyTorch for GPU acceleration (see `requirements.txt`)

### Running Tests
```bash
pytest
```
- Tests are in the `tests/` directory
- To run a specific test: `pytest tests/test_csv_collector.py -v`
- No separate linting configuration; follow existing Python style conventions

### Training LSTM Models
```bash
python training/scripts/train_lstm.py
```
- Expects CSV dataset at `training/data/dataset_squats.csv`
- Output model saved to `models/lstm/` (path configurable in `config/exercises.yml`)
- Uses TensorFlow/Keras for LSTM training

### Checking CUDA Availability
```bash
python check_cuda.py
```
- Verifies CUDA, PyTorch, and TensorFlow GPU support

### Data Processing
```bash
python training/scripts/data_processing.py
```
- Processes raw session data into training datasets

## Architecture

### Layered Design
The codebase follows a strict layered architecture defined in `.claude/rules/00-architecture.md`:

1. **`src/core`** – Core types, contracts, and reusable domain logic
2. **`src/exercises`** – Exercise-specific behavior (squat, jumping jacks, kick, etc.)
3. **`src/pipeline`** – Technical frame processing and execution flow orchestration
4. **`src/presentation`** – Domain state adaptation to UI (presenters, view models)
5. **`src/data_collection`** – Data persistence and publishing (CSV, JSONL)
6. **`src/models`** – External model access (YOLO pose detection)
7. **`src/tracking`** – Exercise repetition counting and state tracking
8. **`src/visualization`** – Frame annotation and rendering

### Key Design Principles
- **Dependency inversion**: High-level modules depend on abstractions, not concrete details
- **Single Responsibility**: Each class/file has one clear responsibility
- **Explicit contracts**: Use `Protocol` and ABCs for interfaces when multiple implementations exist
- **Threading separation**: UI runs on main Qt thread, pipeline runs on dedicated background thread
- **Data flow**: Technical input → transformation → evaluation → tracking → presentation/persistence

### Pipeline Flow (`main.py` → `pipeline_thread.py`)
1. **Capture**: `cv2.VideoCapture` reads frames
2. **Detection**: `YoloPoseDetector` extracts pose keypoints
3. **Tracking**: `PersonRegistry` merges detections across frames
4. **Processing**: `FrameProcessor` evaluates exercises per person
5. **Collection**: `DataCollectionBus` persists frame/rep records
6. **Rendering**: `Renderer` annotates frames with visual feedback
7. **Presentation**: `DashboardPresenter` prepares view models for UI

### Exercise System
- Each exercise lives in `src/exercises/` (e.g., `squat.py`, `jumping_jacks.py`)
- Exercises can use angle-based evaluation or LSTM models (configurable in `exercises.yml`)
- `ExerciseCounter` tracks repetitions across all active exercises
- Exercise registry (`registry.py`) provides factory-like access to exercise detectors

### Data Collection
- **Frame records**: Raw keypoints + exercise state per frame per person (`FrameRecord`)
- **Rep records**: Completed repetition metrics (`RepRecord`)
- **Backends**: CSV and JSONL output configurable in `app.yml`
- **Location**: `data/sessions/` by default

## Configuration

### Configuration Files (`config/`)
- `app.yml` – Application settings (window, FPS, data collection)
- `exercises.yml` – Exercise-specific thresholds and feedback messages
- `model.yml` – YOLO model weights and detection parameters

### Exercise Configuration
Each exercise in `exercises.yml` defines:
- `enabled`: Whether the exercise is active
- `evaluator_type`: `angle` or `lstm` (LSTM requires trained model)
- Thresholds for joint angles and movement detection
- Feedback messages for user guidance

### Model Configuration
- **YOLO weights**: `models/yolo/yolo26x-pose.pt` (can use n/s/m/l/x variants)
- **Device**: `auto` (prefers CUDA if available)
- **LSTM models**: Stored in `models/lstm/`, referenced in exercise config

## Development Guidelines

### Adding New Exercises
1. Create exercise class in `src/exercises/` following existing patterns
2. Implement evaluation logic (angle-based or LSTM-based)
3. Add configuration to `config/exercises.yml`
4. Register in `src/exercises/registry.py`
5. Update `src/pipeline/frame_processor.py` to include new exercise

### Testing
- Write unit tests for domain logic, evaluators, presenters, and data collectors
- Use fakes/stubs over complex mocks
- Test observable contracts, not implementation details
- Each behavior change should come with new or adjusted tests

### Python Style
- Use type hints for public APIs and new classes/functions
- Prefer short functions and small classes
- Use `dataclass` for DTOs/value objects when it improves clarity
- Errors should communicate cause and context

### Data & Artifacts
- Session data, datasets, and model weights are project artifacts
- Don't edit generated files manually; modify source code/scripts instead
- Reproducible logic should live in code or configuration

## Dependencies

### Core Dependencies (`requirements.txt`)
- **Pose detection**: `ultralytics` (YOLO)
- **UI**: `PySide6`
- **ML**: `tensorflow`, `scikit-learn`
- **Utilities**: `numpy<2`, `opencv-python`, `pandas`, `loguru`

### PyTorch with CUDA
Install separately (not in requirements.txt due to platform-specific URLs):
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

### Environment Setup
```bash
pip install -r requirements.txt
# Then install PyTorch with CUDA if needed
```

## Common Development Tasks

### Running a Single Test
```bash
pytest tests/test_csv_collector.py::TestCsvCollector::test_creates_file_with_header
```

### Checking Configuration Changes
```bash
python -c "import yaml; print(yaml.safe_load(open('config/exercises.yml'))['exercises']['squat'])"
```

### Testing Exercise Detection
```bash
python gym_test.py  # Simple test script for exercise evaluation
```

### Monitoring GPU Usage
```bash
nvidia-smi  # If CUDA is available
```