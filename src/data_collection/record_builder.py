from src.core.types import Person, ExerciseSnapshot, FrameRecord


def build_frame_records(
    persons: list[Person],
    frame_number: int,
    timestamp: float,
    current_fps: float = 0.0,
) -> list[FrameRecord]:
    """
    Transforma la lista de Person del frame actual en FrameRecords.

    Función pura: no muta Person ni accede a estado externo.
    Mantiene la lógica de transformación fuera de main.py (SRP).
    """
    records: list[FrameRecord] = []

    for person in persons:
        if person.keypoints is None:
            continue

        # Construir snapshots de cada ejercicio activo
        snapshots = {
            ex_name: ExerciseSnapshot(
                name=ex_name,
                phase=state.phase,
                angles=dict(state.angles),
                feedback=state.feedback,
                valid_down=state.valid_down,
            )
            for ex_name, state in person.exercises.items()
        }

        records.append(FrameRecord(
            timestamp=timestamp,
            frame_number=frame_number,
            track_id=person.track_id,
            fps=current_fps,
            bbox=person.bbox,
            keypoints_xy=person.keypoints.coords.copy(),
            keypoints_conf=person.keypoints.scores.copy(),
            exercises=snapshots,
        ))

    return records
