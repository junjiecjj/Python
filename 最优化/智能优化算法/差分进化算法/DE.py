

"""
https://mp.weixin.qq.com/s?__biz=Mzg3ODY4ODE5MQ==&mid=2247485947&idx=1&sn=cf6cdba8095d7b7e12732510b8193d5e&chksm=ce112c1aae61a4122976eeab7d3cced0e687b587c5205b3e69e2fd4768da34b63bf953ee4aa4&mpshare=1&scene=1&srcid=0901eWum27LI4D23x8mmVwcZ&sharer_shareinfo=894e6486198cb4d2304d20d21e7d60c5&sharer_shareinfo_first=894e6486198cb4d2304d20d21e7d60c5&exportkey=n_ChQIAhIQtiDK7TKF8bE3mLTSsAY7bhKfAgIE97dBBAEAAAAAAEgnCbdQbpAAAAAOpnltbLcz9gKNyK89dVj0KVoqinx4KbwJuBZ5E79qTr2u6PAW0PybxWCri6oW7%2FZ57o3tPC25X2SyXhkq1v5p%2F3Yrs79UcrH8UaNEKGBAJpAclM2UrzUl4sqF%2BcCIuICFiYHhZTUbAAyFkxAFxLKAf5457UCgy31ekpEwVwCyRAVSlpGdXVY19TpAF6EoAcDEFxZI2L9qS4bMQLHSWm0lVYxuIWmtFn5WUoZjevvejDQAX5MwKeX7GnhR74TOImByUJvGNtvipz8MKzgP5RR1L5p5QCAGrKofjPTocU1ovDnIhwa%2FvFmWN4V1sB%2BA8MlbrKk9cPRmVdF3tPx4HzPPIcTwMKBQKB5W&acctmode=0&pass_ticket=kbt%2FvZCAc3Zwa8h6p%2F6qbZjTsExp8L7RgIJZu7oVqBbifeCJf%2FMq%2Ff9m4ZBE%2FiXh&wx_header=0#rd

"""


from typing import List, Dict, Tuple, Union
import random


class Individual:
    def __init__(
            self,
            parameters: Dict[str, float],
            score: Union[float, None] = None
    ) -> None:
        super().__init__()
        self.parameter_values = parameters
        self.score = score


def scoring(ind: Individual) -> float:
    return ind.parameter_values["x"] ** 2 + ind.parameter_values["y"] ** 2


class Population:
    MAXIMIZE = 1
    MINIMIZE = -1

    def __init__(
            self,
            pop_size,
            direction: int,
            bounds: Dict[str, Tuple[float, float]],
            parameters: List[str],

    ) -> None:
        self.pop_size = pop_size
        self.direction = direction
        self.bounds = bounds
        self.parameters = parameters
        self.individuals: List[Individual] = []

    def _create_individual(self) -> Individual:
        dict_parameters = {}
        for parameter in self.parameters:
            param_value = random.uniform(self.bounds[parameter][0], self.bounds[parameter][1])
            dict_parameters[parameter] = param_value
        individual = Individual(parameters=dict_parameters)
        individual.score = scoring(individual)
        return individual

    def initiate(self):
        for i in range(self.pop_size):
            individual = self._create_individual()
            self.individuals.append(individual)

    def get_best_individual(self) -> Individual:
        self.best_individual = max(self.individuals, key=lambda x: x.score * self.direction)
        return self.best_individual

    def get_worst_individual(self) -> Individual:
        self.best_individual = min(self.individuals, key=lambda x: x.score * self.direction)
        return self.best_individual

    def get_mean_score(self) -> float:
        self.mean_score = sum([ind.score for ind in self.individuals]) / len(self.individuals)
        return self.mean_score

    def get_std_score(self) -> float:
        self.std_score = sum([(ind.score - self.mean_score) ** 2 for ind in self.individuals]) / len(self.individuals)
        return self.std_score


class Optimization:
    def __init__(self, f: float, cp: float, population: Population):
        self.f = f
        self.cp = cp
        self.population = population

    @staticmethod
    def _mutant_choose(population: Population) -> Tuple[Individual, Individual, Individual]:
        # chooses 3 random individuals
        individuals = population.individuals
        random_individuals = random.sample(individuals, 3)
        return tuple(random_individuals)

    @staticmethod
    def _handle_boundaries(parameter_value: float, bounds: Tuple[float, float]) -> float:
        # handles the bounds in case calculated parameter goes beyond bounds
        if parameter_value > bounds[1]:
            parameter_value = bounds[1]
        elif parameter_value < bounds[0]:
            parameter_value = bounds[0]
        return parameter_value

    @staticmethod
    def _calculate_parameter(
            semi_parents: Tuple[Individual, Individual, Individual],
            parameter_name: str,
            bounds: Tuple[float, float],
            f: float
    ) -> float:
        # calculates the parameter for trial individual based on semi parents
        trial_parameter = semi_parents[0].parameter_values[parameter_name] + \
                          f * (semi_parents[1].parameter_values[parameter_name] -
                               semi_parents[2].parameter_values[parameter_name])
        trial_parameter = Optimization._handle_boundaries(trial_parameter, bounds)
        return trial_parameter

    def _mutation(self, population: Population) -> Individual:
        # create the trial individual
        # choose semi parents
        semi_parents = Optimization._mutant_choose(population)
        trial_parameters = {}
        # for each parameter name
        for parameter in population.parameters:
            # calculate parameter for trail individual
            trial_parameter = self._calculate_parameter(
                semi_parents, parameter,
                population.bounds[parameter],
                self.f
            )
            trial_parameters[parameter] = trial_parameter
        # create trial individual
        trial_individual = Individual(parameters=trial_parameters)
        return trial_individual

    def _crossover(self, parent: Individual, trial: Individual, parameters):
        child_parameters = {}
        # choose the parameters
        for parameter in parameters:
            prob = random.random()
            if prob < self.cp:
                child_parameters[parameter] = parent.parameter_values[parameter]
            else:
                child_parameters[parameter] = trial.parameter_values[parameter]
        # create child individual
        child = Individual(parameters=child_parameters)
        return child

    @staticmethod
    def _selection(child: Individual, parent: Individual, direction: int) -> bool:
        child.score = scoring(child)
        # preferring child to parent in case of equal scores
        if direction == Population.MAXIMIZE:
            return child.score >= parent.score
        else:
            return child.score <= parent.score

    def main(self, generations: int):
        population = self.population

        for gen in range(generations):
            new_individuals = []

            for i in range(population.pop_size):
                # Mutation
                trial_individual = self._mutation(population)

                # Crossover
                parent = population.individuals[i]
                child = self._crossover(parent, trial_individual, population.parameters)

                # Selection
                if self._selection(child, parent, self.population.direction):
                    new_individuals.append(child)
                else:
                    new_individuals.append(parent)

            # Update the population with the new individuals
            population.individuals = new_individuals

            # Update statistics or perform any other necessary tasks at the end of each generation
            best_individual = population.get_best_individual()
            worst_individual = population.get_worst_individual()
            mean_score = population.get_mean_score()
            std_score = population.get_std_score()

            # Print or store relevant information about the generation
            print(
                f"Generation {gen + 1}: Best Score - {best_individual.score}, Worst Score - {worst_individual.score}, Mean Score - {mean_score}, Std Score - {std_score}")

        # After completing all generations, you can return or perform any final actions
        final_best_individual = population.get_best_individual()
        print(
            f"Optimization complete. Best individual: {final_best_individual.parameter_values}, Score: {final_best_individual.score}")


population = Population(
    pop_size=1000,
    direction=Population.MINIMIZE,
    bounds={"x": (-100, 100), "y": (-100, 100)},
    parameters=["x", "y"]
)
population.initiate()

optimization = Optimization(
    f=0.5,
    cp=0.7,
    population=population
)

optimization.main(20)



























