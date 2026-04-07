# Proyecto

Aplicación Python para análisis de ejercicios a partir de pose estimation, con especial foco en evaluación modular, conteo de repeticiones, recogida de datos y presentación en UI.

## Objetivo de desarrollo

La prioridad del proyecto es:
- SOLID.
- Clean Code.
- Modularidad real.
- Uso explícito de interfaces y contratos.
- Bajo acoplamiento entre capas.
- Alta capacidad de testeo.

## Mapa del repositorio

- `main.py`: punto de entrada y composition root.
- `config/`: configuración YAML de aplicación, ejercicios y modelo.
- `models/`: artefactos de modelos (YOLO, LSTM).
- `training/`: scripts y utilidades para entrenamiento de modelos.
- `src/core/`: contratos, tipos, evaluadores base, buffers y normalización.
- `src/data_collection/`: recogida, construcción y persistencia de datos de sesión/dataset.
- `src/exercises/`: lógica específica de ejercicios y registro/coordinación.
- `src/models/`: wrappers y acceso a modelos.
- `src/pipeline/`: procesamiento de frames, selección de dispositivo, threading y registro de personas.
- `src/presentation/`: presenter, adapters, viewmodels y widgets.
- `src/tracking/`: conteo y tracking de repeticiones/eventos.
- `src/visualization/`: renderizado de dashboard y agregación de estadísticas.
- `tests/`: pruebas unitarias.
- `data/`, `models/`, `training/`: datos, artefactos y herramientas de entrenamiento; no modificarlos salvo petición explícita.

## Reglas globales

- Programa contra interfaces, no contra implementaciones concretas.
- Toda dependencia entre módulos debe entrar por constructor, factoría o composition root.
- Evita acoplar dominio con UI, IO, threading, frameworks o detalles de librerías externas.
- Si una lógica puede vivir como servicio puro o función pura, prefierela frente a clases con estado innecesario.
- Una clase debe tener una sola responsabilidad clara y medible.
- Un módulo no debe conocer detalles internos de otra capa si puede depender de un contrato.
- No introducir “god classes”, utilidades genéricas difusas ni singletons globales.
- Antes de crear una abstracción nueva, comprobar si ya existe un contrato reutilizable en `src/core`.

## Cómo trabajar en cambios

1. Lee primero los archivos relacionados y detecta la capa afectada.
2. Propón cambios mínimos y coherentes con la arquitectura existente.
3. Reutiliza contratos existentes antes de crear otros nuevos.
4. Si aparece una nueva responsabilidad, extráela a un servicio, estrategia, evaluador o adaptador.
5. Mantén las funciones pequeñas y las clases enfocadas.
6. Actualiza o añade tests cuando cambie comportamiento.
7. No toques datos de sesiones, pesos de modelos ni artefactos generados salvo tarea explícita.

## Convenciones de implementación

- Usa type hints en firmas públicas.
- Prefiere nombres semánticos y orientados a dominio.
- Evita comentarios redundantes; el código debe explicar la intención.
- Documenta decisiones no obvias en docstrings cortas o en markdown de arquitectura.
- Maneja errores en el borde del sistema; el núcleo debe ser lo más determinista posible.
- Evita retornar diccionarios anónimos cuando un tipo, dataclass o contrato sea más claro.
- Mantén separadas transformación de datos, reglas de negocio y renderizado.

## Convenciones de revisión

Al revisar o generar código:
- comprobar cohesión y acoplamiento,
- validar dependencia hacia abstracciones,
- evitar duplicación,
- evitar ramas complejas innecesarias,
- verificar compatibilidad con tests,
- no degradar la legibilidad por microoptimizaciones.

## Exclusiones prácticas

No proponer cambios sobre:
- `__pycache__/`
- archivos binarios de modelo,
- datos históricos de sesiones,
- datasets generados,
salvo que la tarea lo pida de forma explícita.