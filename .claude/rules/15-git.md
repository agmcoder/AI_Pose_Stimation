# Git

## Convenciones de commits

### Formato
```
<tipo>(<scope>): <descripción>
```

- **tipo**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`
- **scope** (opcional): módulo o ejercicio afectado (`squat`, `pipeline`, `ui`, etc.)
- **descripción**: imperativo, minúscula, sin punto final

### Tipos y cuándo usarlos
- `feat`: nueva funcionalidad para el usuario
- `fix`: corrección de un bug
- `docs`: cambios en documentación
- `style`: formato, linting, sin cambio funcional
- `refactor`: reestructuración sin cambio externo
- `perf`: mejora de rendimiento
- `test`: añadir o corregir tests
- `chore`: tareas de mantenimiento (deps, config, scripts)

### Ejemplos válidos
```
feat(squat): añadir evaluación por ángulos de rodilla
fix(pipeline): corregir race condition en registro de personas
docs: actualizar CLAUDE.md con flujo de entrenamiento
refactor(exercises): extraer lógica común a BaseEvaluator
test(csv_collector): verificar creación de archivo con cabecera
```

### Ejemplos a evitar
```
- "added something" (usa `feat`)
- "fixed bug" (describe qué bug)
- "update" (ambiguo, especifica tipo)
- mensajes largos sin formato
```

## Estrategia de branching

### Ramas principales
- `main`: código en producción (estable, solo releases)
- `develop`: integración continua (rama de desarrollo)

### Ramas de apoyo
- `feature/<nombre>`: nueva funcionalidad (se mergea a `develop`)
- `hotfix/<nombre>`: corrección crítica para `main` (se mergea a `main` y `develop`)
- `release/<version>`: preparación de release (se mergea a `main` y `develop`)

### Convención de nombres
- **feature**: `feature/data-processing-exercise-model-training`
- **hotfix**: `hotfix/fix-memory-leak-in-detector`
- **release**: `release/v1.2.0`

### Flujo recomendado
1. Para nueva funcionalidad:
   ```
   git checkout develop
   git pull origin develop
   git checkout -b feature/nombre-descriptivo
   # commits según convenciones
   git push origin feature/nombre-descriptivo
   # crear PR a develop
   ```

2. Para corrección crítica en producción:
   ```
   git checkout main
   git pull origin main
   git checkout -b hotfix/descripcion-corta
   # commits de fix
   git push origin hotfix/descripcion-corta
   # crear PR a main y develop
   ```

## Buenas prácticas
- Commits pequeños y atómicos (un cambio lógico por commit)
- Mensajes claros que expliquen el "por qué" no solo el "qué"
- Pull Requests con descripción que incluya contexto y testing
- Rebase interactivo antes de merge para mantener historial limpio
- Nunca force-push a ramas compartidas (`develop`, `main`)