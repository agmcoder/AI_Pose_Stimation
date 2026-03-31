---
paths:
  - "config/**/*.yml"
  - "environment_*.yml"
---

# Configuración

## Principios

- La configuración debe ser explícita, coherente y predecible.
- No introducir claves ambiguas ni duplicadas.
- Evitar defaults silenciosos que oculten errores de configuración.

## Cambios

- Toda nueva clave debe tener nombre estable y propósito claro.
- Si una clave afecta comportamiento de dominio, reflejarlo en el código mediante un punto de acceso bien definido.
- No dispersar parsing de configuración por múltiples módulos si puede centralizarse.