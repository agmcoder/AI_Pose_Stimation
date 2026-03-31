---
paths:
  - "src/**/*.py"
  - "main.py"
  - "gym_test.py"
---

# Interfaces y contratos

## Norma principal

- Diseña nuevas capacidades detrás de interfaces cuando:
  - haya más de una implementación posible,
  - el componente toque IO o librerías externas,
  - el componente deba ser fácilmente testeable,
  - la lógica pueda cambiar por estrategia o configuración.

## Preferencias

- Preferir `Protocol` o ABC cuando el contrato sea parte del diseño.
- Las dependencias de constructores deben tiparse con interfaces o tipos abstractos cuando tenga sentido.
- Las factorías devuelven contratos, no implementaciones concretas, salvo en composición root.
- Los módulos consumidores no deben importar clases concretas de infraestructura si pueden depender de una abstracción.

## No hacer

- No pasar objetos gigantes con múltiples responsabilidades “porque ya existen”.
- No usar clases concretas como dependencia por defecto si el comportamiento puede variar.
- No filtrar detalles de librerías externas al dominio si pueden encapsularse.