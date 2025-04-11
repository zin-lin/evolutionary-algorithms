/**
 * Author: Zin Lin Htun
 */

package coursework;
import java.util.ArrayList;
import model.Fitness;
import model.Individual;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;

import static java.lang.Math.abs;

/*
 *
 */
public class ParticleSwarmOptimiser extends NeuralNetwork {
    private ArrayList<Particle> swarm;
    private double inertia = .76;      // Inertia weight
    private double cognitive = 1.25;    // Personal best influence
    private double social = 1.25;       // Global best influence
    private double previous = 100;

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

        this.previous = best.fitness;

        // introduce particle inertia and social influence factor and self/cognitive influence factor
        inertia = 0.75 - (0.5 * ((double) evaluations / Parameters.maxEvaluations));
        social = 1.25 -  (0.2 * ((double) evaluations / Parameters.maxEvaluations));
        cognitive = 1.25 + (0.2 * ((double) evaluations / Parameters.maxEvaluations));
        System.out.println("Initial Global Best: " + best.fitness);
        System.out.println("Initial Social: " + social);

        // run as per evaluations
        // rules set eval == 20000
        while (evaluations < Parameters.maxEvaluations) {

            reset();

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

    /**
     * reset
     */
    private void reset() {
        // System.out.println("Resetting Global Called");
        if ( this.evaluations >= 2000 && this.evaluations % 200 == 0) {
            System.out.println("Resetting called");
            if (this.best.fitness >= 0.06) {
                System.out.println("Reset situation met");
                reinitializeSwarm();
                System.out.println("Reset called bad seed");
                System.out.println("Reset Global Best: " + best.fitness);
                previous = best.fitness;
            }
            else {
                ArrayList<Particle> newSwarm = new ArrayList<>();
                int count = 0;
                for (Particle particle: swarm) {

                    if (Parameters.mutateRate <= Parameters.random.nextDouble()) {
                        Particle newParticle = new Particle();
                        evaluateFitness(newParticle);
                        newSwarm.add(newParticle);
                        count++;
                    }
                    else {
                        newSwarm.add(particle);
                    }
                }
                this.swarm = newSwarm;
            }
        }
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
    // re-initialising
    private void reinitializeSwarm() {
        ArrayList<Particle> newSwarm = new ArrayList<>();
        int count = 0;
        for (Particle particle: swarm) {

            if (Parameters.mutateRate > Parameters.random.nextDouble()) {
                Particle newParticle = new Particle();
                evaluateFitness(newParticle);
                newSwarm.add(newParticle);
                count++;
            }
            else {
                newSwarm.add(particle);
            }
        }
        this.swarm = newSwarm;
        System.out.println("Reinitialized particle number: " + count);
    }

    // escape the infamous local optima
    private void escapeLocalOptima() {
        double now = best.fitness;
        if (((previous - now ) < 0.005) && evaluations > 10000){
            ArrayList<Particle> newSwarm = new ArrayList<>();
            // add population
            for (Particle particle : swarm) {

                if (Parameters.mutateRate < Parameters.random.nextGaussian()) {
                    if (particle.position.fitness == best.fitness) {
                        newSwarm.add(particle);
                        continue;
                    }
                    Particle newParticle = new Particle();
                    evaluateFitness(newParticle);
                    newSwarm.add(newParticle);
                } else {
                    newSwarm.add(particle);
                }

            }
            swarm =  newSwarm;
        }
        previous = now;
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

    // global best solution
    private Particle getGlobalBestAsParticle() {
        // check for the best
        Particle pbest = new Particle();
        for (Particle particle : swarm) {
            if (best == null || particle.position.fitness < best.fitness) {
                pbest = particle;
            }
        }

        return pbest;
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
