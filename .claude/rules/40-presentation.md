---
paths:
  - "src/presentation/**/*.py"
  - "src/visualization/**/*.py"
---

# Reglas de presentación

## Separación

- `presenter`, `viewmodels`, `adapters` y `widgets` deben mantener responsabilidades separadas.
- Los widgets no deben contener reglas de negocio.
- El presenter transforma estado y eventos a un formato consumible por UI; no debe asumir detalles visuales innecesarios.

## Acoplamiento

- La UI depende de contratos y viewmodels, no del detalle interno del dominio.
- Evita que widgets accedan directamente a servicios de pipeline, modelos o persistencia.
- Si una vista necesita datos derivados, prepararlos en presenter o viewmodel.

## Cambios

- Todo cambio visual debe preservar la separación entre renderizado y lógica.
- Mantener fácil sustitución de adaptadores y tests de presenter.