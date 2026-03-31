---
paths:
  - "src/exercises/**/*.py"
---

# Reglas para ejercicios

## Diseño

- Cada archivo de ejercicio debe contener sólo comportamiento propio del ejercicio.
- La lógica compartida entre ejercicios debe extraerse a mixins, utilidades pequeñas o componentes comunes.
- Evita copiar lógica de evaluación entre ejercicios; generaliza donde aporte claridad.

## Evaluación

- La evaluación debe apoyarse en contratos claros y componibles.
- Si existe evaluación por ángulos y evaluación por modelo, mantenerlas desacopladas y sustituibles.
- Los umbrales, estados y transiciones deben ser explícitos y fáciles de testear.

## Registro

- Cualquier alta de ejercicio nuevo debe integrarse de forma estable con el registro/factory existente.
- No introducir condicionales extensos en cascada si una estrategia, registry o factory resuelve mejor el problema.