import numpy as np
import random
from deap import base, creator, tools
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import optuna
import pandas as pd

# Load Iris dataset
df = pd.read_csv(r'predict_students_dropout_and_academic_success.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

le = LabelEncoder()
y = le.fit_transform(y)
# X = data.data
# y = data.target

# Split train/validation for fitness evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=42)

num_features = X.shape[1]

# DEAP fitness and individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # maximize accuracy
creator.create("Individual", list, fitness=creator.FitnessMax)

def create_toolbox():
    toolbox = base.Toolbox()
    # Individual: binary list to select features
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def eval_individual(individual):
        selected = [i for i, bit in enumerate(individual) if bit == 1]
        if len(selected) == 0:
            return 0.,
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train.values[:, selected], y_train)
        preds = clf.predict(X_val.values[:, selected])
        return accuracy_score(y_val, preds),
    
    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

def run_deap(pop_size, n_gen):
    toolbox = create_toolbox()
    pop = toolbox.population(n=pop_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    for gen in range(n_gen):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate new fitnesses
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit
        
        pop[:] = offspring
    
    best_ind = tools.selBest(pop, 1)[0]
    best_acc = best_ind.fitness.values[0]
    selected_features = [i for i, bit in enumerate(best_ind) if bit == 1]
    return best_acc, selected_features

def objective(trial):
    # Optuna trials for DEAP parameters
    pop_size = trial.suggest_int("pop_size", 10, 100)
    n_gen = trial.suggest_int("n_gen", 5, 20)
    
    best_acc, selected = run_deap(pop_size, n_gen)
    print(f"Trial pop_size={pop_size}, n_gen={n_gen} -> accuracy={best_acc:.4f} features={selected}")
    return best_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Best params:", study.best_params)
    print("Best accuracy:", study.best_value)

    # Run final DEAP with best params on train+val
    X_final_train = np.vstack([X_train, X_val])
    y_final_train = np.hstack([y_train, y_val])
    
    def run_final_deap(pop_size, n_gen):
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def eval_individual(individual):
            selected = [i for i, bit in enumerate(individual) if bit == 1]
            if len(selected) == 0:
                return 0.,
            clf = LogisticRegression(max_iter=5000)
            # clf.fit(X_final_train[:, selected], y_final_train)
            clf.fit(X_train.values[:, selected], y_train)
            # preds = clf.predict(X[:, selected])  # Full dataset predictions
            preds = clf.predict(X_val.values[:, selected])
            return accuracy_score(y, preds),
        
        toolbox.register("evaluate", eval_individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=pop_size)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for gen in range(n_gen):
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid)
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring

        best_ind = tools.selBest(pop, 1)[0]
        best_acc = best_ind.fitness.values[0]
        selected_features = [i for i, bit in enumerate(best_ind) if bit == 1]
        return best_acc, selected_features

    best_acc, best_features = run_final_deap(study.best_params["pop_size"], study.best_params["n_gen"])
    print("Final accuracy on full dataset:", best_acc)
    print("Selected features:", best_features)
