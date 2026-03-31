---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
  - "main.py"
---

# Testing

## Regla base

- Cada cambio de comportamiento debe venir acompañado de test nuevo o ajuste de tests.
- Priorizar tests unitarios sobre lógica de dominio, evaluadores, presenters, buses y collectors.
- Usar fakes o stubs simples antes que mocks complejos.

## Cobertura útil

- Cubrir caso feliz, bordes y errores razonables.
- Testear contratos observables, no detalles internos frágiles.
- Evitar tests acoplados a implementación incidental.

## Diseño testable

- Si algo cuesta testear, revisar dependencias, tamaño de clase y mezcla de responsabilidades.
- Preferir inyección de dependencias y funciones puras para simplificar tests.