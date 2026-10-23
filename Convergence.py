import os
import evox
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import algorithm
import problem
from Scripts.Data_processing import data


def run_aco_experiment(distances, params, n_iterations=20, seed=42):
  """Ejecuta el flujo de ACO con un conjunto específico de parámetros y retorna el historial del mejor fitness."""
  key = jax.random.PRNGKey(seed)

  aco_algorithm = algorithm.TensorACO(
      distances=distances,
      node_count=distances.shape[0],
      n_ants=params.get("n_ants", 20),
      n_best=params.get("n_best", 5),
      decay=params.get("decay", 0.9),
      alpha=params.get("alpha", 1),
      beta=params.get("beta", 1),
  )

  tsp_problem = problem.TSP(distances)
  monitor = evox.monitors.StdSOMonitor()

  workflow = evox.workflows.StdWorkflow(
      algorithm=aco_algorithm, problem=tsp_problem, monitor=monitor
  )

  state = workflow.init(key)
  historic_fit = []

  for _ in range(n_iterations):
    state = workflow.step(state)
    monitor.flush()
    best_fitness = monitor.get_best_fitness()
    historic_fit.append(best_fitness.item())

  return historic_fit


if __name__ == "__main__":
  # 1. Procesamiento de datos y carga de distancias
  export_path = "Data"
  cat_var = ["artist_name", "track_id", "track_name", "popularity", "mode"]
  data("Data/SpotifyAudioFeaturesApril2019.csv", cat_var, export_path)

  base_path = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(base_path, "Data", "Distances.npy")
  distances = jnp.load(file_path)

  # 2. Definición de las corridas con parámetros a comparar
  experiments = [
      {
          "label": "Base (ants=20, decay=0.9, α=1, β=1)",
          "params": {"n_ants": 20, "n_best": 5, "decay": 0.9, "alpha": 1, "beta": 1},
      },
      {
          "label": "Más Hormigas (ants=50, decay=0.9, α=1, β=1)",
          "params": {
              "n_ants": 50,
              "n_best": 10,
              "decay": 0.9,
              "alpha": 1,
              "beta": 1,
          },
      },
      {
          "label": "Alta Evaporación (ants=20, decay=0.5, α=1, β=1)",
          "params": {"n_ants": 20, "n_best": 5, "decay": 0.5, "alpha": 1, "beta": 1},
      },
      {
          "label": "Mayor Heurística (ants=20, decay=0.9, α=1, β=3)",
          "params": {"n_ants": 20, "n_best": 5, "decay": 0.9, "alpha": 1, "beta": 3},
      },
  ]

  n_iterations = 15  # Se amplió el número de iteraciones para apreciar la convergencia

  plt.figure(figsize=(10, 6))

  # 3. Ejecución de cada configuración y graficado
  for exp in experiments:
    print(f"Ejecutando: {exp['label']}...")
    fitness_history = run_aco_experiment(
        distances=distances,
        params=exp["params"],
        n_iterations=n_iterations,
        seed=42,
    )

    plt.plot(
        range(1, n_iterations + 1),
        fitness_history,
        marker="o",
        linewidth=2,
        label=exp["label"],
    )

  # 4. Formato de la gráfica
  plt.title(
      "Comparativa de Convergencia: ACO en TSP", fontsize=13, fontweight="bold"
  )
  plt.xlabel("Iteración", fontsize=11)
  plt.ylabel("Fitness (Distancia)", fontsize=11)
  plt.xticks(range(1, n_iterations + 1))
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.legend(loc="upper right", frameon=True)
  plt.tight_layout()

  plt.show()