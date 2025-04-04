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
    }

    @Override
    public void run() {
        swarm = initializeSwarm();
        best = getGlobalBest();
        double evalProgress = ( 0.5 * ((double) evaluations / Parameters.maxEvaluations));
        inertia = 0.85 - evalProgress;


        System.out.println("Initial Global Best: " + best.fitness);

        while (evaluations < Parameters.maxEvaluations) {
            for (GeneticSwarmMutationOptimiser.AnnexParticle particle : swarm) {
                updateVelocity(particle);
                updatePosition(particle);
                evaluateFitness(particle);

            }

            best = getGlobalBest();  // Update global best solution
            outputStats();

        }

        saveNeuralNetwork(); // Save best network
    }

    private ArrayList<GeneticSwarmMutationOptimiser.AnnexParticle> initializeSwarm() {
        ArrayList<GeneticSwarmMutationOptimiser.AnnexParticle> swarm = new ArrayList<>();
        for (int i = 0; i < Parameters.popSize; i++) {
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

    private Individual getGlobalBest() {
        for (GeneticSwarmMutationOptimiser.AnnexParticle particle : swarm) {
            if (best == null || particle.position.fitness < best.fitness) {
                best = particle.position.copy();
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

    @Override
    public double activationFunction(double x) {
        return Math.tanh(x);
    }
}
