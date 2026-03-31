---
paths:
  - "data/**/*"
  - "models/**/*"
  - "training/data/*.csv"
  - "*.csv"
---

# Datos y artefactos

## Protección de artefactos

- Tratar datos de sesión, datasets generados y pesos de modelos como artefactos sensibles del proyecto.
- No editar manualmente estos archivos salvo tarea explícita.
- No proponer refactors que mezclen código fuente con artefactos o datos runtime.

## Código relacionado

- Si una tarea afecta entrenamiento, procesamiento de dataset o carga de modelos, modificar scripts y código fuente antes que tocar artefactos persistidos.
- Toda lógica reproducible debe vivir en código o configuración, no en edición manual de archivos generados.