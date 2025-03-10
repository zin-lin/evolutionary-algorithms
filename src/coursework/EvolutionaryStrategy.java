package coursework;

import java.util.ArrayList;

import model.Fitness;
import model.Individual;
import model.NeuralNetwork;

/**
 * Implements a basic Evolutionary Algorithm to train a Neural Network
 * @author zin-lin-htun
 * 
 */
public class EvolutionaryStrategy extends NeuralNetwork {
	

	/**
	 * The Main Evolutionary Loop
	 */
	@Override
	public void run() {		
		//Initialise a population of Individuals with random weights
		population = initialise();

		//Record a copy of the best Individual in the population
		best = getBest();
		System.out.println("Best From Initialisation " + best);

		/**
		 * main EA processing loop
		 */		
		
		while (evaluations < Parameters.maxEvaluations) {

			/**
			 * this is a skeleton EA - you need to add the methods.
			 * You can also change the EA if you want 
			 * You must set the best Individual at the end of a run
			 * 
			 */

			// Select 2 Individuals from the current population. Currently returns random Individual
			Individual parent1 = select(); 
			Individual parent2 = select();

			// Generate a child by crossover. Not Implemented			
			ArrayList<Individual> children = reproduce(parent1, parent2);			
			
			//mutate the offspring
			mutate(children);
			
			// Evaluate the children
			evaluateIndividuals(children);			

			// Replace children in population
			replace(children);

			// check to see if the best has improved
			best = getBest();
			
			// Implemented in NN class. 
			outputStats();
			
			//Increment number of completed generations			
		}

		//save the trained network to disk
		saveNeuralNetwork();
	}


	/**
	 * Sets the fitness of the individuals passed as parameters (whole population)
	 * 
	 */
	private void evaluateIndividuals(ArrayList<Individual> individuals) {
		for (Individual individual : individuals) {
			individual.fitness = Fitness.evaluate(individual, this);
		}
	}


	/**
	 * Returns a copy of the best individual in the population
	 * 
	 */
	private Individual getBest() {
		best = null;;
		for (Individual individual : population) {
			if (best == null) {
				best = individual.copy();
			} else if (individual.fitness < best.fitness) {
				best = individual.copy();
			}
		}
		return best;
	}

	/**
	 * Generates a randomly initialised population
	 * 
	 */
	private ArrayList<Individual> initialise() {
		population = new ArrayList<>();
		for (int i = 0; i < Parameters.popSize; ++i) {
			//chromosome weights are initialised randomly in the constructor
			Individual individual = new Individual();
			population.add(individual);
		}
		evaluateIndividuals(population);
		return population;
	}

	private Individual getBestInTournament(ArrayList<Individual> tournament) {
		best = null;
		for  (Individual individual : tournament) {
			if (best == null) {
				best = individual.copy();
			} else{
				if (best.fitness < individual.fitness) {
				best = individual.copy();}
			}
		}
		return best.copy();
	}

	/**
	 * Selection --
	 * 
	 * NEEDS REPLACED with proper selection this just returns a copy of a random
	 * adding tournement selection method
	 * member of the population
	 */
	private Individual select() {
		Individual best = null; // setting the best individual
		int arena_size = 3; // tournament arena
		ArrayList<Individual>tournament = new ArrayList<>();

		for (int i = 0; i < arena_size; ++i) {
			Individual parent = population.get(Parameters.random.nextInt(Parameters.popSize));
			tournament.add(parent.copy()); // adding choice to arena
		}

		return getBestInTournament(tournament);
	}

	/**
	 * Crossover / Reproduction
	 * 
	 * Uses blending crossover
	 */
	private ArrayList<Individual> reproduce(Individual parent1, Individual parent2) {
		ArrayList<Individual> children = new ArrayList<>();
		double alpha = 0.675; // Blending factor
		Individual child1 = parent1.copy();
		Individual child2 = parent2.copy();

		for (int i = 0; i < parent1.chromosome.length; i++) {

			double minGene = Math.min(parent1.chromosome[i], parent2.chromosome[i]);
			double maxGene = Math.max(parent1.chromosome[i], parent2.chromosome[i]);
			double range = maxGene - minGene;

			child1.chromosome[i] = minGene + Parameters.random.nextDouble() * (range + (alpha * range)); // blend min
			child2.chromosome[i] = maxGene - Parameters.random.nextDouble() * (range + (alpha * range)); // blend max value
		}

		children.add(child1);
		children.add(child2);
		return children;
	} 
	
	/**
	 * Mutation
	 * 
	 * 
	 */
	private void mutate(ArrayList<Individual> individuals) {		
		for(Individual individual : individuals) {
			for (int i = 0; i < individual.chromosome.length; i++) {
				double reverseEvaluationFactor = 1 - (evaluations/Parameters.maxEvaluations); // REF, this will become lower over ticks
				double mutationRate = Parameters.mutateRate * reverseEvaluationFactor; // this will become lower over ticks due to REF becoming lower
				if (Parameters.random.nextDouble() < mutationRate) {
					if (Parameters.random.nextBoolean()) {
						individual.chromosome[i] += (Parameters.mutateChange);
					} else {
						individual.chromosome[i] -= (Parameters.mutateChange);
					}
				}
			}
		}		
	}

	/**
	 * 
	 * Replaces the worst member of the population compared with the best of the latest iterated generation of children
	 * (regardless of fitness)
	 * 
	 */
	private void replace(ArrayList<Individual> individuals) {
		for(Individual individual : individuals) {
			int idx = getWorstIndex();
			Individual worst = population.get(idx);
			// for each iteration replaces only if the worst actually have a worse fitness then the best of the children generation
			if (worst.fitness > best.fitness ) {
				population.set(idx, individual);
			}
		}
	}

	

	/**
	 * Returns the index of the worst member of the population
	 * @return
	 */
	private int getWorstIndex() {
		Individual worst = null;
		int idx = -1;
		for (int i = 0; i < population.size(); i++) {
			Individual individual = population.get(i);
			if (worst == null) {
				worst = individual;
				idx = i;
			} else if (individual.fitness > worst.fitness) {
				worst = individual;
				idx = i; 
			}
		}
		return idx;
	}	

	@Override
	public double activationFunction(double x) {
		if (x < -20.0) {
			return -1.0;
		} else if (x > 20.0) {
			return 1.0;
		}
		return Math.tanh(x);
	}
}
