package coursework;
import java.util.ArrayList;
import model.Fitness;
import model.Individual;
import model.NeuralNetwork;

public class GeneticSwarmMutationOptimiser extends NeuralNetwork {
    private ArrayList<GeneticSwarmMutationOptimiser.AnnexParticle> swarm;
    private double inertia = .76;      // Inertia weight
    private static double cognitive= 1.25;     // Personal best influence
    private static double social= 1.25;      // Global best influence
    private static int count = 0;
    private double fitnessPrev=10.0;
    private double fitness=10.0;
    private double operatingMode = 1.0;
    // inner class
    private static class AnnexParticle {
        Individual position;   // Particle position (neural network weights)
        double[] velocity;     // Velocity vector
        Individual pBest;      // Best position found by the particle

        AnnexParticle() {
            position = new Individual(); // Randomly initialized weights
            velocity = new double[position.chromosome.length];
            pBest = position.copy();
        }

        AnnexParticle(Individual individual) {
            position = individual.copy(); // Randomly initialized weights
            velocity = new double[position.chromosome.length];
            pBest = position.copy();
        }
    }

    @Override
    public void run() {
        swarm = initializeSwarm();
        best = getGlobalBest();
//        double evalProgress = ( 0.5 * ((double) evaluations / Parameters.maxEvaluations));
//        inertia = 0.85 - evalProgress;



        System.out.println("Initial Global Best: " + best.fitness);

        while (evaluations < Parameters.maxEvaluations) {
            // for every 500 evaluations
            if (count%50 == 0 && count != 0){
                checkToMutate(); // this will set the operation mode
            }

            if (operatingMode == 1.0) {
                best = getGlobalBest();
                double evalProgress = (0.5 * ((double) evaluations / Parameters.maxEvaluations));
                inertia = 0.85 - evalProgress;


                for (GeneticSwarmMutationOptimiser.AnnexParticle particle : swarm) {
                    updateVelocity(particle);
                    updatePosition(particle);
                    evaluateFitness(particle);
                }
                best = getGlobalBest();  // Update global best solution
            }
            else {

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
            }


            outputStats();
            count++;
        }

        saveNeuralNetwork(); // Save best network
    }

    private void checkToMutate(){

        // if difference is too little or is stagnating mutate the algorithm
        if ( !((fitness - best.fitness) < 0.005)){
            System.out.println("Mutated");
            operatingMode = operatingMode == 1.0? 2.0: 1.0;
            if (operatingMode == 1.0){
                swarm = mutateToPSO(population);
            }
            else{
                population = mutateToGNA(swarm);
            }
        }
        fitness = best.fitness;

    }

    private ArrayList<Individual> mutateToGNA(ArrayList<GeneticSwarmMutationOptimiser.AnnexParticle> particles) {
        ArrayList<Individual> pop = new ArrayList<>();
        for (GeneticSwarmMutationOptimiser.AnnexParticle particle : particles) {
            Individual individual = particle.position.copy();
            pop.add(individual);
        }
        return pop;
    }

    private ArrayList<GeneticSwarmMutationOptimiser.AnnexParticle> mutateToPSO(ArrayList<Individual>population) {
        ArrayList<GeneticSwarmMutationOptimiser.AnnexParticle> pop = new ArrayList<>();
        for (Individual individual : population) {
            GeneticSwarmMutationOptimiser.AnnexParticle annexParticle = new GeneticSwarmMutationOptimiser.AnnexParticle(individual);

            /// resetting chromosome
            annexParticle.position.chromosome = individual.chromosome;
        }

        return pop;
    }

    private ArrayList<GeneticSwarmMutationOptimiser.AnnexParticle> initializeSwarm() {
        ArrayList<GeneticSwarmMutationOptimiser.AnnexParticle> swarm = new ArrayList<>();
        for (int i = 0; i < Parameters.popSize + 100; i++) {
            GeneticSwarmMutationOptimiser.AnnexParticle particle = new GeneticSwarmMutationOptimiser.AnnexParticle();
            evaluateFitness(particle);
            swarm.add(particle);
        }
        return swarm;
    }

    private void evaluateFitness(GeneticSwarmMutationOptimiser.AnnexParticle particle) {
        particle.position.fitness = Fitness.evaluate(particle.position, this);

        // Update personal best
        if (particle.position.fitness < particle.pBest.fitness) {
            particle.pBest = particle.position.copy();
        }
    }

    /**
     * for PSO
     * @return
     */
    private Individual getGlobalBest() {
        for (GeneticSwarmMutationOptimiser.AnnexParticle particle : swarm) {
            if (best == null || particle.position.fitness < best.fitness) {
                best = particle.position.copy();
            }
        }
        return best;
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

    private void updateVelocity(GeneticSwarmMutationOptimiser.AnnexParticle particle) {
        for (int i = 0; i < particle.velocity.length; i++) {
            double r1 = Parameters.random.nextDouble();
            double r2 = Parameters.random.nextDouble();

            // inertia component, cognitive and social components
            double inertiaComponent = inertia * particle.velocity[i];
            double cognitiveComponent = cognitive * r1 *
                    (particle.pBest.chromosome[i] - particle.position.chromosome[i]); // we kinda want the particles to be more reactive
            double socialComponent = social * r2 *
                    (best.chromosome[i] - particle.position.chromosome[i]);

            // recalculating particle velocity
            particle.velocity[i] = inertiaComponent + cognitiveComponent + socialComponent;

            //  velocity_maximum = vMax is set to not let particles explode
            double evalProgress = (double) evaluations / Parameters.maxEvaluations;
            double vMax = (Parameters.maxGene - Parameters.minGene) *  (Parameters.mutateRate);
            if (particle.velocity[i] > vMax) particle.velocity[i] = vMax;
            if (particle.velocity[i] < -vMax) particle.velocity[i] = -vMax;
        }
    }

    private void updatePosition(GeneticSwarmMutationOptimiser.AnnexParticle particle) {
        for (int i = 0; i < particle.position.chromosome.length; i++) {
            particle.position.chromosome[i] += particle.velocity[i];
        }
    }

    /**
     * Crossover / Reproduction
     * Uses blending crossover
     */
    private ArrayList<Individual> reproduce(Individual parent1, Individual parent2) {
        ArrayList<Individual> children = new ArrayList<>();
        double alpha = 0.75; // Decrease blending factor to ensure smoother offspring

        Individual child1 = parent1.copy();
        Individual child2 = parent2.copy();

        for (int i = 0; i < parent1.chromosome.length; i++) {
            double minGene = Math.min(parent1.chromosome[i], parent2.chromosome[i]);
            double maxGene = Math.max(parent1.chromosome[i], parent2.chromosome[i]);
            double range = maxGene - minGene;

            // Blending with a reduced factor for finer adjustments
            child1.chromosome[i] = minGene + Parameters.random.nextDouble() * (range + (alpha * range));
            child2.chromosome[i] = maxGene - Parameters.random.nextDouble() * (range + (alpha * range));
        }

        children.add(child1);
        children.add(child2);
        return children;
    }

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

    /**
     * Sets the fitness of the individuals passed as parameters (whole population)
     *
     */
    private void evaluateIndividuals(ArrayList<Individual> individuals) {
        for (Individual individual : individuals) {
            individual.fitness = Fitness.evaluate(individual, this);
        }
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
        int arena_size =3; // tournament arena
        ArrayList<Individual>tournament = new ArrayList<>();

        for (int i = 0; i < arena_size; ++i) {
            Individual parent = population.get(Parameters.random.nextInt(Parameters.popSize));
            tournament.add(parent.copy()); // adding choice to arena
        }

        return getBestInTournament(tournament);
    }


    /**
     * Mutation
     *
     *
     */
    private void mutate(ArrayList<Individual> individuals) {
        for(Individual individual : individuals) {
            for (int i = 0; i < individual.chromosome.length; i++) {
                double reverseEvaluationFactor =  ((double) evaluations /Parameters.maxEvaluations); // REF, this will become lower over ticks
                double mutationRate = Parameters.mutateChange * 0.5;// this will become lower over ticks due to REF becoming lower

                if (Parameters.random.nextDouble() < Parameters.mutateRate) {
                    individual.chromosome[i] += Parameters.random.nextGaussian()* (mutationRate) * reverseEvaluationFactor;
                } else {
                    individual.chromosome[i] -=  Parameters.random.nextGaussian()* mutationRate * reverseEvaluationFactor;
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
            if (worst.fitness > individual.fitness ) {
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
        return Math.tanh(x);
    }
}
