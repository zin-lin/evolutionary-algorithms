/**
 * Author: Zin Lin Htun
 */

package coursework;
import java.util.ArrayList;
import model.Fitness;
import model.Individual;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;

/*
 *
 */
public class ParticleSwarmOptimiser extends NeuralNetwork {
    private ArrayList<Particle> swarm;
    private double inertia = .76;      // Inertia weight
    private double cognitive = 1.25;    // Personal best influence
    private double social = 1.25;       // Global best influence

    // inner class de Particle
    private class Particle {
        Individual position;   // Particle position, will be assigned as individuals
        double[] velocity;     // velocity vector
        Individual pBest;      // personal best position

        // @constructor __init__
        Particle() {
            position = new Individual(); // Randomly initialized weights
            velocity = new double[position.chromosome.length];
            pBest = position.copy();
        }
    }

    // run method -> Kevin's neural network
    @Override
    public void run() {
        swarm = initializeSwarm();
        // create global best solution
        best = getGlobalBest();

        // introduce particle inertia and social influence factor and self/cognitive influence factor
        inertia = 0.80 - (0.5 * ((double) evaluations / Parameters.maxEvaluations));
        social = 1.25 -  (0.2 * ((double) evaluations / Parameters.maxEvaluations));
        cognitive = 1.25 + (0.2 * ((double) evaluations / Parameters.maxEvaluations));
        System.out.println("Initial Global Best: " + best.fitness);

        // run as per evaluations
        // rules set eval == 20000
        while (evaluations < Parameters.maxEvaluations) {
            for (Particle particle : swarm) {
                updateVelocity(particle);
                updatePosition(particle);
                evaluateFitness(particle);

            }

            best = getGlobalBest();  // Update global best solution
            outputStats(); // output fitness

        }

        // save to weight file
        saveNeuralNetwork(); // Save best network
    }

    // initialise swarm at the start of the run: adding population
    private ArrayList<Particle> initializeSwarm() {
        ArrayList<Particle> swarm = new ArrayList<>();
        // add population
        for (int i = 0; i < Parameters.popSize; i++) {
            Particle particle = new Particle();
            evaluateFitness(particle);
            swarm.add(particle);
        }
        return swarm;
    }

    // evaluating fitness
    private void evaluateFitness(Particle particle) {
        particle.position.fitness = Fitness.evaluate(particle.position, this);

        // choose better fitness, somewhat elitism
        if (particle.position.fitness < particle.pBest.fitness) {
            particle.pBest = particle.position.copy();
        }
    }

    // global best solution
    private Individual getGlobalBest() {
        // check for the best
        for (Particle particle : swarm) {
            if (best == null || particle.position.fitness < best.fitness) {
                best = particle.position.copy();
            }
        }

        return best;
    }

    // update particle velocity
    private void updateVelocity(Particle particle) {
        for (int i = 0; i < particle.velocity.length; i++) {
            // randomizers
            double r1 = Parameters.random.nextDouble();
            double r2 = Parameters.random.nextDouble();
            // affect the inertial, and cognitive as well as social

            double inertiaComponent = inertia * particle.velocity[i];
            double cognitiveComponent = cognitive * r1 *
                    (particle.pBest.chromosome[i] - particle.position.chromosome[i]);
            double socialComponent = social * r2 *
                    (best.chromosome[i] - particle.position.chromosome[i]);

            // construct velocity of the particle, velocity of the rockets in this sense
            particle.velocity[i] = inertiaComponent + cognitiveComponent + socialComponent;

            //  Creating a maximum velocity with influence from the mutation rate from the parameters
            // This will prevent particles from exploring too much when exploitation is required
            // prevent smth called particle explosion
            double progress = (double) evaluations / Parameters.maxEvaluations;
            double vMax = (Parameters.maxGene - Parameters.minGene) * Parameters.mutateRate * (1 - progress);
            if (particle.velocity[i] > vMax) particle.velocity[i] = vMax;
            if (particle.velocity[i] < -vMax) particle.velocity[i] = -vMax;
        }
    }

    // update each chromosome, using + operator so that velocities can be updated REALTIME
    private void updatePosition(Particle particle) {
        for (int i = 0; i < particle.position.chromosome.length; i++) {
            particle.position.chromosome[i] += particle.velocity[i];
        }
    }

    // tanh is kept as an activation function
    @Override
    public double activationFunction(double x) {

        return Math.tanh(x);
    }
}
