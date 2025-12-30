# Smart Crowd Router

**Navegación peatonal inteligente para Barcelona, usando Reinforcement Learning**

> Proyecto de: Filippo Pacini, Jacopo Crispolti, Liseth Berdeja, Sieun You

---

## Visión General

Smart Crowd Router es un sistema de navegación peatonal que calcula rutas óptimas por Barcelona, teniendo en cuenta la **densidad de multitud en tiempo real**. A diferencia de los sistemas de rutas tradicionales que solo minimizan la distancia, este proyecto usa un **agente de Q-Learning** para equilibrar dos objetivos:

1. **Minimizar la distancia** al destino
2. **Evitar áreas concurridas** basándose en datos históricos y condiciones actuales

El sistema siempre compara la ruta sugerida con el camino más corto, permitiendo a los usuarios evaluar el compromiso entre distancia y comodidad.

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────┐
│                     DATA SOURCES                    │
├─────────────────┬─────────────────┬─────────────────┤
│  OpenData BCN   │  Google Maps    │ OpenStreetMap   │
│  (city POIs)    │  (popularTimes) │ (street graph)  │
└────────┬────────┴────────┬────────┴────────┬────────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           ▼
               ┌───────────────────────┐
               │    run map_builder    │
               │    (map.json.gz)      │
               └───────────┬───────────┘
                           ▼
               ┌─────────────────────┬────────────────────────────┐
               │  main.py            │ tools.py                   │
               │  (user interaction) │ (weather, pathfinding, ..) │
               └─────────────────────┴────────────────────────────┘
                          
```

---

## Estructura de Archivos

| File | Descripción |
|------|-------------|
| `main.py` | Script principal: carga datos, maneja la interacción del usuario, ejecuta la búsqueda de rutas |
| `tools.py` | Utilidades: clima, búsqueda de lugares, cálculo de distancia, algoritmo Q-Learning |
| `map_builder.py` | Construye el gráfico de la ciudad combinando POIs, datos de multitud y red de calles |
| `map.json.gz` | Gráfico de la ciudad pre-calculado (generado por map_builder.py) |

---

## Cómo Funciona

### Construcción del Mapa (Configuración única)

`map_builder.py` crea el gráfico de la ciudad:

1. **Descarga POIs** de OpenData BCN (~900 puntos de interés)
2. **Enriquece con Google PopularTimes**: para cada POI, encuentra datos históricos de multitud por hora
3. **Construye la red de calles** desde OpenStreetMap usando la librería python OSMnx
4. **Vincula POIs a calles**: cada punto de interés se asocia con el segmento de calle más cercano
5. **Agrega datos**: los niveles de multitud de los POI se propagan a sus calles anfitrionas
6. **Exporta** el gráfico como un archivo JSON comprimido

**Estructura de un Nodo:**
```json
{
  "id": 12345,
  "type": 0, // 0 = calle, 1 = lugar
  "name": "Carrer de Balmes",
  "coords": [41.3951, 2.1534],
  "len": 125.4, // longitud en metros
  "conns": [12344, 12346, 50023], // nodos conectados
  "popular_times": [[...], [...], ...]  // matriz de 7x24
}
```

### Ajustes de Multitud en Tiempo Real

Al inicio, `main.py`:

- Obtiene las **condiciones climáticas actuales** de la API Open-Meteo (Simplificado a Soleado/Nublado/Lluvioso)
- Determina **día y hora actuales**
- Calcula el nivel de multitud de cada calle usando:

```
crowd = popular_times[day][hour] × weather_modifier
```

| Weather | Modifier |
|---------|----------|
| Sunny | ×1.3 |
| Cloudy | ×0.9 |
| Rainy | ×0.3 |

### Algoritmo Q-Learning

El agente aprende el camino óptimo a través de prueba y error:

**Componentes de Aprendizaje por Refuerzo:**

| Elemento | Descripción |
|----------|-------------|
| **Estado** | Nodo actual en el gráfico (calle o POI) |
| **Acción** | Moverse a un nodo conectado |
| **Recompensa** | Función que equilibra distancia y exposición a la multitud |

**Función de Recompensa:**
```python
reward = -1                           # Penalización base por paso
reward += 10000 if goal_reached       # Bono de llegada
reward += 1 if getting_closer         # Bono de dirección
reward -= node_crowd_level            # Penalización por multitud
```

**Hiperparámetros:**
- Learning rate (α): 0.5 con decay 0.9992
- Discount factor (γ): 1.0
- Exploration rate (ε): 1.0 con decay 0.999
- Episodios máximos: 5000
- Early stopping: convergencia si ΔQ < 0.01 por 5 episodios consecutivos

### Resultados y Comparaciones

El camino óptimo se compara con el camino más corto posible encontrado (calculado usando un algoritmo A*); El usuario puede ver métricas (longitud del camino, exposición promedio a la multitud) y una vista rápida de los pasos hacia la meta para ambas opciones.

### Modo de Navegación Inteligente

El Modo de Navegación Inteligente permite a los usuarios interactuar directamente con la política de Q-learning aprendida. En cada paso, el sistema sugiere opciones de ruta clasificadas de mejor a peor, mientras sigue dando al usuario total libertad para anular la recomendación y explorar caminos alternativos.

---

## Uso

**Ejemplo de un Output:**
```
[AI Project - Smart Crowd Router] By Filippo Pacini, Jacopo Crispolti, Liseth Berdeja, Sieun You

Fetching weather info...
> Monday 14:30, Sunny

Enter your location: Sagrada Familia
> Best match: Basílica de la Sagrada Família [Place]

Enter destination (<3km distance): Parc Guell
> Best match: Park Güell [Place]

[Q-Learning] from 'Basílica de la Sagrada Família' to 'Park Güell'
· Episode 10: 245 steps, Δ = 1.2340
· Episode 50: 89 steps, Δ = 0.3421
-> Values converged at episode 127 (max Δ = 0.01, patience = 5)

[Suggested path] 2340 meters | 12.45 average crowd exposure
Sagrada Família -> Carrer de Provença -> Carrer de Sardenya -> ...

[Shortest path] 1890 meters | 28.73 average crowd exposure
Sagrada Família -> Avinguda de Gaudí -> Travessera de Gràcia -> ...
```

---

## Contribución del Proyecto
Persona  |  Área  | % 
---------|--------|----
Filippo  | Algoritmos de pathfinding (Q-learning, función de recompensa, lógica de entrenamiento, A*) | 25% 
Jacopo   | Diseño y construcción del mapa (OpenData BCN, OSM, Google PopularTimes, lógica de agregación) | 30% 
Liseth   | Integración del sistema e interacción (main.py, smart navigation, integración del meteo, UX) | 25% 
Sieun    | Investigación de datos y cohesión (OpenData BCN datasets, meteo APIs, refactorización de código, documentación) | 20%