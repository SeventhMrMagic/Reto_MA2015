import jax
import jax.numpy as jnp
import algorithm
import problem
import os
import matplotlib.pyplot as plt
import evox
import numpy as np
from Scripts.Data_processing import data
from Scripts.Data_sync import id

if __name__ == '__main__':
    # Data processing
    export_path = "Data"
    
    cat_var = ['artist_name', 'track_id', 'track_name', 'popularity', 'mode']
    data("Data/SpotifyAudioFeaturesApril2019.csv", cat_var, export_path)

    base_path = os.path.dirname(os.path.abspath(__file__))  
    file_path = os.path.join(base_path, "Data", "Distances.npy") 

    algorithm = algorithm.TensorACO(
        distances=jnp.load(file_path),
        node_count=jnp.load(file_path).shape[0],
        n_ants=20,
        n_best=5,
        decay=0.9,
        alpha=1,
        beta=1
    )

    problem = problem.TSP(
        jnp.load(file_path)
    )
    monitor = evox.monitors.StdSOMonitor()

    key = jax.random.PRNGKey(42)

    workflow = evox.workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitor=monitor
    )

    state = workflow.init(key)
    historic_fit = []
    for i in range(10):
        state = workflow.step(state)
        monitor.flush()
        best_path = monitor.get_best_solution()
        best_fitness = monitor.get_best_fitness()
        print("Iteration:", i, "Best Fitness:", best_fitness)
        print(f"Visited Node Order: {best_path}")
        historic_fit.append(best_fitness.item())

    print(id(best_path,"Data/SpotifyAudioFeaturesApril2019.csv"))
