from src.core.types import Person, ExerciseState


class PersonRegistry:
    """
    Single Responsibility: única fuente de verdad del estado persistente
    de cada persona a lo largo de toda la sesión.

    Fusiona cada frame nuevo con el estado acumulado por track_id.
    Resuelve la raíz del problema: YoloPoseDetector crea Person nuevos
    cada frame, perdiendo el historial.
    """

    def __init__(self):
        self._store: dict[int, Person] = {}

    def merge(self, persons: list[Person]) -> list[Person]:
        """
        Recibe la lista de Person del frame actual (recién creados, sin historial).
        Devuelve la lista con el estado acumulado restaurado.
        """
        enriched = []
        for person in persons:
            tid = person.track_id
            if tid in self._store:
                # Restaurar exercises del historial antes de procesar
                person.exercises = dict(self._store[tid].exercises)
            self._store[tid] = person
            enriched.append(person)
        return enriched

    def persist(self, persons: list[Person]) -> None:
        """
        Guarda el estado actualizado tras el procesamiento de ejercicios.
        Llamar DESPUÉS de FrameProcessor.process().
        """
        for person in persons:
            self._store[person.track_id] = person

    def all_persons(self) -> list[Person]:
        """Devuelve todas las personas vistas en la sesión, visibles o no."""
        return list(self._store.values())

    def reset(self) -> None:
        self._store.clear()
