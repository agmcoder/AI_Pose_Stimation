# Arquitectura

## Capas y dependencias

- `src/core` define contratos, tipos y lógica transversal reutilizable.
- `src/exercises` implementa comportamiento específico por ejercicio.
- `src/pipeline` orquesta procesamiento técnico de frames y flujo de ejecución.
- `src/presentation` adapta estado de dominio a UI.
- `src/data_collection` persiste o publica datos.
- `src/models` encapsula acceso a modelos externos.
- `src/tracking` y `src/visualization` consumen contratos claros y no deben invadir otras capas.

## Reglas

- Las capas de alto nivel no deben depender de detalles concretos de bajo nivel.
- Las implementaciones concretas deben depender de contratos definidos en una capa estable.
- La composición de dependencias debe resolverse en `main.py`, factorías o puntos de arranque equivalentes.
- Si una pieza mezcla orquestación, cálculo y persistencia, dividirla.
- El flujo recomendado es: entrada técnica -> transformación -> evaluación -> tracking -> presentación/persistencia.

## Diseño

- Preferir estrategias, evaluadores, adaptadores y factorías frente a condicionales crecientes.
- Extraer responsabilidades compartidas a módulos pequeños y explícitos.
- Cuando un cambio afecta a varios ejercicios, primero buscar generalización en contratos o componentes comunes.