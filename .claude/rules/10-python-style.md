---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
  - "main.py"
  - "gym_test.py"
---

# Python style

## Estilo general

- Usa type hints en toda API pública y en nuevas clases/funciones.
- Prefiere funciones cortas y clases pequeñas.
- Mantén bajo el nivel de anidación; usar early return cuando mejore claridad.
- Los nombres deben describir intención, no implementación accidental.

## Modelado

- Usa `dataclass` para DTOs o value objects simples cuando mejore claridad.
- Evita parámetros booleanos que cambien drásticamente el comportamiento.
- Evita estructuras `dict` ambiguas como retorno cuando un tipo explícito sea más claro.

## Errores y efectos

- Los errores deben comunicar causa y contexto.
- Los efectos secundarios deben concentrarse en bordes del sistema.
- La lógica de negocio debe ser lo más pura y determinista posible.

## Mantenibilidad

- No introducir helpers genéricos sin un caso de uso claro.
- No optimizar prematuramente a costa de la legibilidad.
- Si una función necesita muchos comentarios para entenderse, dividirla.