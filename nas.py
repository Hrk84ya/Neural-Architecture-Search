import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
import copy

@dataclass
class SearchConfig:
    """Configuration for NAS"""
    population_size: int = 8
    max_generations: int = 6
    mutation_rate: float = 0.3
    epochs_per_eval: int = 20
    batch_size: int = 128
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3


class ArchitectureSpace:
    """Defines a structured and valid search space"""

    def __init__(self):
        # CNN block types with predefined valid configurations
        self.conv_blocks = [
            {'type': 'conv', 'filters': [32, 32], 'kernel': 3},
            {'type': 'conv', 'filters': [64, 64], 'kernel': 3},
            {'type': 'conv', 'filters': [32, 64], 'kernel': 3},
            {'type': 'conv', 'filters': [64, 128], 'kernel': 3},
            {'type': 'separable', 'filters': [32, 32], 'kernel': 3},
            {'type': 'separable', 'filters': [64, 64], 'kernel': 3},
            {'type': 'separable', 'filters': [32, 64], 'kernel': 5},
            {'type': 'separable', 'filters': [64, 128], 'kernel': 5},
            {'type': 'residual', 'filters': [32, 32], 'kernel': 3},
            {'type': 'residual', 'filters': [64, 64], 'kernel': 3},
            {'type': 'residual', 'filters': [128, 128], 'kernel': 3},
            {'type': 'inception', 'filters': [32, 64], 'kernel': 3},
            {'type': 'inception', 'filters': [64, 128], 'kernel': 3},
        ]

        self.pooling_ops = ['maxpool', 'avgpool', 'none']
        self.activations = ['relu', 'elu', 'swish']
        self.dense_units = [128, 256, 512]

    def get_random_block(self):
        """Get a random valid CNN block"""
        block = copy.deepcopy(np.random.choice(self.conv_blocks))
        block['activation'] = str(np.random.choice(self.activations))
        block['pooling'] = str(np.random.choice(self.pooling_ops))
        return block


class ArchitectureBuilder:
    """Builds valid Keras models from architecture specifications"""

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def conv_block(self, x, filters, kernel_size, activation='relu'):
        """Standard convolutional block with BatchNorm"""
        x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x

    def separable_block(self, x, filters, kernel_size, activation='relu'):
        """Depthwise separable convolution - more efficient"""
        x = layers.SeparableConv2D(filters, kernel_size, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x

    def residual_block(self, x, filters, kernel_size, activation='relu'):
        """Residual block with skip connection"""
        shortcut = x

        # First conv
        x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)

        # Second conv
        x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        # Match dimensions for skip connection
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # Add skip connection
        x = layers.Add()([x, shortcut])
        x = layers.Activation(activation)(x)
        return x

    def inception_block(self, x, filters, kernel_size, activation='relu'):
        """Simplified Inception-like block with multiple paths"""
        # 1x1 conv path
        path1 = layers.Conv2D(filters // 4, 1, padding='same', use_bias=False)(x)
        path1 = layers.BatchNormalization()(path1)
        path1 = layers.Activation(activation)(path1)

        # 3x3 conv path
        path2 = layers.Conv2D(filters // 4, 1, padding='same', use_bias=False)(x)
        path2 = layers.BatchNormalization()(path2)
        path2 = layers.Activation(activation)(path2)
        path2 = layers.Conv2D(filters // 2, 3, padding='same', use_bias=False)(path2)
        path2 = layers.BatchNormalization()(path2)
        path2 = layers.Activation(activation)(path2)

        # 5x5 conv path (using two 3x3)
        path3 = layers.Conv2D(filters // 4, 1, padding='same', use_bias=False)(x)
        path3 = layers.BatchNormalization()(path3)
        path3 = layers.Activation(activation)(path3)
        path3 = layers.Conv2D(filters // 4, 3, padding='same', use_bias=False)(path3)
        path3 = layers.BatchNormalization()(path3)
        path3 = layers.Activation(activation)(path3)

        # Concatenate all paths
        x = layers.Concatenate()([path1, path2, path3])
        return x

    def build_model(self, architecture: Dict) -> keras.Model:
        """Build a valid Keras model from architecture specification"""
        inputs = keras.Input(shape=self.input_shape)
        x = inputs

        # Initial stem convolution for better feature extraction
        x = layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Build CNN blocks (backbone)
        for i, block_config in enumerate(architecture['blocks']):
            block_type = block_config['type']
            filters_list = block_config['filters']
            kernel = block_config['kernel']
            activation = block_config.get('activation', 'relu')

            # Apply block based on type
            for filters in filters_list:
                if block_type == 'conv':
                    x = self.conv_block(x, filters, kernel, activation)
                elif block_type == 'separable':
                    x = self.separable_block(x, filters, kernel, activation)
                elif block_type == 'residual':
                    x = self.residual_block(x, filters, kernel, activation)
                elif block_type == 'inception':
                    x = self.inception_block(x, filters, kernel, activation)

            # Apply pooling after block if specified
            pooling = block_config.get('pooling', 'none')
            if pooling == 'maxpool' and x.shape[1] > 2:  # Ensure we don't pool too much
                x = layers.MaxPooling2D(2, padding='same')(x)
            elif pooling == 'avgpool' and x.shape[1] > 2:
                x = layers.AveragePooling2D(2, padding='same')(x)

        # Global pooling to convert to 1D
        x = layers.GlobalAveragePooling2D()(x)

        # Dense classification head
        for units in architecture.get('dense_layers', []):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)

        # Final output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='nas_model')
        return model


class EvolutionaryNAS:
    """Evolutionary Neural Architecture Search with all improvements"""

    def __init__(self, input_shape, num_classes, config: SearchConfig = None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.config = config or SearchConfig()

        self.space = ArchitectureSpace()
        self.builder = ArchitectureBuilder(input_shape, num_classes)

        self.population = []
        self.history = []
        self.best_architecture = None
        self.best_accuracy = 0.0
        self.generation_best = []

    def create_random_architecture(self) -> Dict:
        """Create a random but valid architecture"""
        num_blocks = int(np.random.randint(2, 5))  # 2-4 blocks
        blocks = []

        for _ in range(num_blocks):
            block = self.space.get_random_block()
            blocks.append(block)

        # Add dense layers (0-2 layers)
        num_dense = int(np.random.randint(0, 3))
        dense_layers = [int(np.random.choice(self.space.dense_units))
                       for _ in range(num_dense)]

        return {
            'blocks': blocks,
            'dense_layers': dense_layers
        }

    def mutate_architecture(self, architecture: Dict) -> Dict:
        """Mutate an architecture with various strategies"""
        mutated = json.loads(json.dumps(architecture))  # Deep copy

        mutation_choices = []

        # Can add block
        if len(mutated['blocks']) < 5:
            mutation_choices.append('add_block')

        # Can remove block
        if len(mutated['blocks']) > 1:
            mutation_choices.append('remove_block')

        # Can always modify
        mutation_choices.extend(['modify_block', 'modify_pooling',
                                'modify_activation', 'modify_dense'])

        mutation_type = np.random.choice(mutation_choices)

        if mutation_type == 'add_block':
            new_block = self.space.get_random_block()
            insert_pos = int(np.random.randint(0, len(mutated['blocks']) + 1))
            mutated['blocks'].insert(insert_pos, new_block)

        elif mutation_type == 'remove_block':
            remove_idx = int(np.random.randint(0, len(mutated['blocks'])))
            mutated['blocks'].pop(remove_idx)

        elif mutation_type == 'modify_block':
            idx = int(np.random.randint(0, len(mutated['blocks'])))
            new_block = self.space.get_random_block()
            mutated['blocks'][idx] = new_block

        elif mutation_type == 'modify_pooling':
            idx = int(np.random.randint(0, len(mutated['blocks'])))
            mutated['blocks'][idx]['pooling'] = str(np.random.choice(self.space.pooling_ops))

        elif mutation_type == 'modify_activation':
            idx = int(np.random.randint(0, len(mutated['blocks'])))
            mutated['blocks'][idx]['activation'] = str(np.random.choice(self.space.activations))

        elif mutation_type == 'modify_dense':
            num_dense = int(np.random.randint(0, 3))
            mutated['dense_layers'] = [int(np.random.choice(self.space.dense_units))
                                       for _ in range(num_dense)]

        return mutated

    def crossover(self, arch1: Dict, arch2: Dict) -> Dict:
        """Combine two architectures via crossover"""
        child = {'blocks': [], 'dense_layers': []}

        # Crossover blocks - take random sections from each parent
        len1, len2 = len(arch1['blocks']), len(arch2['blocks'])

        if len1 > 0 and len2 > 0:
            # Take first part from parent1
            split1 = int(np.random.randint(1, len1 + 1))
            child['blocks'].extend(arch1['blocks'][:split1])

            # Take second part from parent2
            split2 = int(np.random.randint(0, len2))
            child['blocks'].extend(arch2['blocks'][split2:])

            # Ensure we have at least 1 block and max 5
            if len(child['blocks']) == 0:
                child['blocks'] = [self.space.get_random_block()]
            elif len(child['blocks']) > 5:
                child['blocks'] = child['blocks'][:5]
        else:
            child['blocks'] = arch1['blocks'] if len1 > 0 else arch2['blocks']

        # Crossover dense layers
        child['dense_layers'] = (arch1['dense_layers'] if np.random.random() < 0.5
                                 else arch2['dense_layers'])

        return child

    def evaluate_architecture(self, architecture: Dict,
                            x_train, y_train,
                            x_val, y_val) -> Tuple[float, float, int]:
        """Train and evaluate a single architecture with data augmentation"""
        try:
            model = self.builder.build_model(architecture)

            # Data augmentation
            datagen = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.15,
                height_shift_range=0.15,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            datagen.fit(x_train)

            # Compile model
            initial_lr = 0.001
            optimizer = keras.optimizers.Adam(learning_rate=initial_lr)

            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Callbacks for better training
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=0
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_accuracy',
                    factor=0.5,
                    patience=self.config.reduce_lr_patience,
                    min_lr=1e-7,
                    verbose=0
                )
            ]

            # Train with augmentation
            history = model.fit(
                datagen.flow(x_train, y_train, batch_size=self.config.batch_size),
                validation_data=(x_val, y_val),
                epochs=self.config.epochs_per_eval,
                steps_per_epoch=len(x_train) // self.config.batch_size,
                callbacks=callbacks,
                verbose=0
            )

            # Get best validation accuracy
            val_accuracy = float(max(history.history['val_accuracy']))
            num_params = int(model.count_params())

            # Efficiency score: balance accuracy and model size
            # Penalize models with too many parameters
            param_penalty = (num_params / 10_000_000) * 0.02
            efficiency_score = val_accuracy - param_penalty

            # Clean up
            del model
            keras.backend.clear_session()

            return val_accuracy, efficiency_score, num_params

        except Exception as e:
            print(f"  ⚠ Error evaluating architecture: {e}")
            return 0.0, 0.0, 0

    def pareto_selection(self, population):
        """Select architectures based on Pareto optimality (accuracy vs efficiency)"""
        pareto_front = []
        
        for candidate in population:
            is_dominated = False
            
            # Check if candidate is dominated by any other individual
            for other in population:
                if (other['accuracy'] >= candidate['accuracy'] and 
                    other['efficiency'] >= candidate['efficiency'] and
                    (other['accuracy'] > candidate['accuracy'] or 
                     other['efficiency'] > candidate['efficiency'])):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(candidate)
        
        return pareto_front
    
    def crowding_distance_selection(self, pareto_front, target_size):
        """Select diverse solutions from Pareto front using crowding distance"""
        if len(pareto_front) <= target_size:
            return pareto_front
        
        # Calculate crowding distance for each solution
        for individual in pareto_front:
            individual['crowding_distance'] = 0.0
        
        # Sort by each objective and assign crowding distance
        objectives = ['accuracy', 'efficiency']
        
        for obj in objectives:
            # Sort by current objective
            pareto_front.sort(key=lambda x: x[obj])
            
            # Set boundary solutions to infinite distance
            pareto_front[0]['crowding_distance'] = float('inf')
            pareto_front[-1]['crowding_distance'] = float('inf')
            
            # Calculate crowding distance for intermediate solutions
            obj_range = pareto_front[-1][obj] - pareto_front[0][obj]
            if obj_range > 0:
                for i in range(1, len(pareto_front) - 1):
                    distance = (pareto_front[i+1][obj] - pareto_front[i-1][obj]) / obj_range
                    pareto_front[i]['crowding_distance'] += distance
        
        # Select solutions with highest crowding distance
        pareto_front.sort(key=lambda x: x['crowding_distance'], reverse=True)
        return pareto_front[:target_size]
    
    def multi_objective_selection(self, population, target_size):
        """Multi-objective selection combining Pareto optimality and crowding distance"""
        # Find Pareto front
        pareto_front = self.pareto_selection(population)
        
        print(f"   🎯 Pareto Front Size: {len(pareto_front)}")
        
        # If Pareto front is larger than target, use crowding distance
        if len(pareto_front) > target_size:
            selected = self.crowding_distance_selection(pareto_front, target_size)
            print(f"   📏 Applied crowding distance selection")
        else:
            selected = pareto_front
            
            # Fill remaining slots with best efficiency scores from non-dominated
            remaining_slots = target_size - len(selected)
            if remaining_slots > 0:
                non_pareto = [ind for ind in population if ind not in pareto_front]
                non_pareto.sort(key=lambda x: x['efficiency'], reverse=True)
                selected.extend(non_pareto[:remaining_slots])
                print(f"   ➕ Added {remaining_slots} high-efficiency solutions")
        
        return selected

    def search(self, x_train, y_train, x_val, y_val):
        """Run evolutionary architecture search with multi-objective optimization"""
        print("="*80)
        print("MULTI-OBJECTIVE EVOLUTIONARY NEURAL ARCHITECTURE SEARCH")
        print("="*80)
        print(f"Configuration:")
        print(f"  Population Size: {self.config.population_size}")
        print(f"  Generations: {self.config.max_generations}")
        print(f"  Epochs per Evaluation: {self.config.epochs_per_eval}")
        print(f"  Mutation Rate: {self.config.mutation_rate}")
        print(f"  Training Samples: {len(x_train)}")
        print(f"  Validation Samples: {len(x_val)}")
        print(f"  Objectives: Accuracy ⚖️ Model Efficiency")
        print("="*80 + "\n")

        # Initialize population
        print("🔬 Initializing Population...")
        print("-"*80)

        for i in range(self.config.population_size):
            arch = self.create_random_architecture()
            acc, eff, params = self.evaluate_architecture(
                arch, x_train, y_train, x_val, y_val
            )

            self.population.append({
                'architecture': arch,
                'accuracy': acc,
                'efficiency': eff,
                'params': params,
                'generation': 0
            })

            print(f"  Individual {i+1}/{self.config.population_size}: "
                  f"Acc={acc:.4f} | Params={params:,}")

            if acc > self.best_accuracy:
                self.best_accuracy = acc
                self.best_architecture = arch

        gen_best = max(self.population, key=lambda x: x['accuracy'])
        self.generation_best.append(gen_best)
        print(f"\n✨ Generation 0 Best: {gen_best['accuracy']:.4f}")
        print("="*80 + "\n")

        # Evolution loop
        for gen in range(1, self.config.max_generations + 1):
            print(f"🧬 GENERATION {gen}/{self.config.max_generations}")
            print("-"*80)

            # Sort by efficiency score for display purposes
            self.population.sort(key=lambda x: x['efficiency'], reverse=True)
            
            # Analyze Pareto front
            pareto_front = self.pareto_selection(self.population)
            pareto_accuracies = [p['accuracy'] for p in pareto_front]
            pareto_params = [p['params'] for p in pareto_front]
            
            print(f"📊 Pareto Front Analysis:")
            print(f"   Size: {len(pareto_front)} solutions")
            print(f"   Accuracy Range: {min(pareto_accuracies):.4f} - {max(pareto_accuracies):.4f}")
            print(f"   Parameter Range: {min(pareto_params):,} - {max(pareto_params):,}")

            # Track history
            gen_stats = {
                'generation': gen,
                'best_accuracy': self.population[0]['accuracy'],
                'avg_accuracy': float(np.mean([p['accuracy'] for p in self.population])),
                'best_efficiency': self.population[0]['efficiency'],
                'best_params': self.population[0]['params']
            }
            self.history.append(gen_stats)

            # Update global best
            if self.population[0]['accuracy'] > self.best_accuracy:
                self.best_accuracy = self.population[0]['accuracy']
                self.best_architecture = self.population[0]['architecture']
                print(f"🎯 NEW BEST ARCHITECTURE FOUND!")
                print(f"   Accuracy: {self.best_accuracy:.4f}")
                print(f"   Parameters: {self.population[0]['params']:,}")

            print(f"📊 Generation Stats:")
            print(f"   Best Accuracy: {gen_stats['best_accuracy']:.4f}")
            print(f"   Avg Accuracy:  {gen_stats['avg_accuracy']:.4f}")
            print(f"   Best Params:   {gen_stats['best_params']:,}")

            # Multi-objective selection using Pareto optimality
            num_survivors = self.config.population_size // 2
            survivors = self.multi_objective_selection(self.population, num_survivors)
            print(f"\n🏆 Selected {len(survivors)} Pareto-optimal survivors for breeding")

            # Create new population
            new_population = [s.copy() for s in survivors]
            offspring_count = 0

            # Generate offspring
            while len(new_population) < self.config.population_size:
                # Select parents (tournament selection)
                parent1, parent2 = np.random.choice(survivors, 2, replace=False)

                # Crossover
                child_arch = self.crossover(
                    parent1['architecture'],
                    parent2['architecture']
                )

                # Mutation
                if np.random.random() < self.config.mutation_rate:
                    child_arch = self.mutate_architecture(child_arch)
                    mutation_marker = "🧬"
                else:
                    mutation_marker = "  "

                # Evaluate offspring
                acc, eff, params = self.evaluate_architecture(
                    child_arch, x_train, y_train, x_val, y_val
                )

                offspring_count += 1
                print(f"  {mutation_marker} Offspring {offspring_count}: "
                      f"Acc={acc:.4f} | Params={params:,}")

                new_population.append({
                    'architecture': child_arch,
                    'accuracy': acc,
                    'efficiency': eff,
                    'params': params,
                    'generation': gen
                })

            self.population = new_population

            # Track best of generation
            gen_best = max(self.population, key=lambda x: x['accuracy'])
            self.generation_best.append(gen_best)

            print("="*80 + "\n")

        # Final summary
        print("\n" + "="*80)
        print("🏁 SEARCH COMPLETED!")
        print("="*80)
        print(f"Best Validation Accuracy: {self.best_accuracy:.4f}")
        print(f"\n📋 Best Architecture Found:")
        print(json.dumps(self.best_architecture, indent=2))
        print("="*80)

        return self.best_architecture

    def plot_results(self):
        """Visualize search results with comprehensive plots including Pareto analysis"""
        if not self.history:
            print("No history to plot")
            return

        generations = [h['generation'] for h in self.history]
        best_acc = [h['best_accuracy'] for h in self.history]
        avg_acc = [h['avg_accuracy'] for h in self.history]
        params = [h['best_params'] / 1e6 for h in self.history]

        # Add generation 0
        gen0_best = self.generation_best[0]
        generations = [0] + generations
        best_acc = [gen0_best['accuracy']] + best_acc
        avg_acc = [float(np.mean([p['accuracy'] for p in self.population[:self.config.population_size]]))] + avg_acc
        params = [gen0_best['params'] / 1e6] + params

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

        # 1. Accuracy Evolution
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(generations, best_acc, 'b-o', linewidth=2.5, markersize=9,
                label='Best Accuracy', zorder=3)
        ax1.plot(generations, avg_acc, 'g--s', linewidth=2, markersize=7,
                label='Average Accuracy', alpha=0.7)
        ax1.axhline(y=self.best_accuracy, color='r', linestyle=':', linewidth=2,
                   label=f'Global Best: {self.best_accuracy:.4f}', alpha=0.7)
        ax1.fill_between(generations, best_acc, avg_acc, alpha=0.2, color='blue')
        ax1.set_xlabel('Generation', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Validation Accuracy', fontsize=13, fontweight='bold')
        ax1.set_title('🧬 Evolutionary Progress', fontsize=15, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=11, loc='lower right')
        ax1.set_ylim([0, 1])

        # 2. Model Complexity
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(generations, params, 'r-^', linewidth=2.5, markersize=8)
        ax2.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')
        ax2.set_title('📊 Best Model Complexity', fontsize=14, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, linestyle='--')

        # 3. Pareto Front Visualization
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Plot all population points
        all_acc = []
        all_params = []
        all_gens = []
        
        for gen_best in self.generation_best:
            all_acc.append(gen_best['accuracy'])
            all_params.append(gen_best['params'] / 1e6)
            all_gens.append(gen_best['generation'])
        
        # Plot evolution path
        ax3.plot(all_params, all_acc, 'b-', alpha=0.3, linewidth=1, label='Evolution Path')
        
        # Plot current Pareto front
        final_pareto = self.pareto_selection(self.population)
        pareto_acc = [p['accuracy'] for p in final_pareto]
        pareto_params = [p['params'] / 1e6 for p in final_pareto]
        
        ax3.scatter(all_params, all_acc, c=all_gens, cmap='viridis',
                   s=100, alpha=0.6, edgecolors='gray', linewidth=1, label='Generation Best')
        ax3.scatter(pareto_params, pareto_acc, c='red', s=200, alpha=0.8, 
                   edgecolors='darkred', linewidth=2, marker='*', label='Pareto Front')
        
        ax3.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax3.set_title('🎯 Pareto Front Analysis', fontsize=14, fontweight='bold', pad=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(fontsize=10)

        # 4. Multi-Objective Progress
        ax4 = fig.add_subplot(gs[2, 0])
        
        # Calculate hypervolume approximation (area under Pareto front)
        hypervolumes = []
        for gen in range(len(self.generation_best)):
            # Get population at this generation
            gen_pop = [p for p in self.population if p.get('generation', 0) <= gen]
            if gen_pop:
                pareto = self.pareto_selection(gen_pop)
                # Simple hypervolume approximation
                if pareto:
                    max_acc = max(p['accuracy'] for p in pareto)
                    min_params = min(p['params'] for p in pareto) / 1e6
                    hv = max_acc * (10 - min_params) if min_params < 10 else max_acc
                    hypervolumes.append(hv)
                else:
                    hypervolumes.append(0)
            else:
                hypervolumes.append(0)
        
        ax4.plot(range(len(hypervolumes)), hypervolumes, 'g-o', linewidth=2.5, markersize=8)
        ax4.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Hypervolume (Approx)', fontsize=12, fontweight='bold')
        ax4.set_title('📊 Multi-Objective Progress', fontsize=14, fontweight='bold', pad=10)
        ax4.grid(True, alpha=0.3, linestyle='--')

        # 5. Pareto Front Details
        ax5 = fig.add_subplot(gs[2, 1])
        
        # Show final Pareto front solutions
        final_pareto = self.pareto_selection(self.population)
        pareto_acc = [p['accuracy'] for p in final_pareto]
        pareto_params = [p['params'] / 1e6 for p in final_pareto]
        pareto_eff = [p['efficiency'] for p in final_pareto]
        
        # Create a table-like visualization
        ax5.axis('off')
        
        # Sort Pareto solutions by accuracy
        pareto_sorted = sorted(zip(pareto_acc, pareto_params, pareto_eff), reverse=True)
        
        table_text = "🏆 FINAL PARETO FRONT\n" + "="*35 + "\n"
        table_text += f"{'Rank':<4} {'Accuracy':<9} {'Params(M)':<9} {'Efficiency':<10}\n"
        table_text += "-"*35 + "\n"
        
        for i, (acc, param, eff) in enumerate(pareto_sorted[:8]):  # Show top 8
            table_text += f"{i+1:<4} {acc:<9.4f} {param:<9.2f} {eff:<10.4f}\n"
        
        if len(pareto_sorted) > 8:
            table_text += f"... and {len(pareto_sorted)-8} more solutions\n"
        
        ax5.text(0.05, 0.95, table_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='left',
                fontfamily='monospace', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 6. Summary Statistics
        ax6 = fig.add_subplot(gs[3, 0])
        ax6.axis('off')

        summary_text = f"""
        ╔══════════════════════════════════════╗
        ║        SEARCH SUMMARY               ║
        ╠══════════════════════════════════════╣
        ║                                      ║
        ║  🎯 Best Accuracy: {self.best_accuracy:.4f}        ║
        ║  📊 Final Avg: {avg_acc[-1]:.4f}            ║
        ║  📈 Total Improvement: {(best_acc[-1]-best_acc[0]):.4f}  ║
        ║  🔢 Best Model Params: {params[-1]:.2f}M      ║
        ║  🧬 Generations: {len(generations)-1}              ║
        ║  👥 Population Size: {self.config.population_size}           ║
        ║  ⏱️  Epochs/Eval: {self.config.epochs_per_eval}             ║
        ║  🎯 Pareto Solutions: {len(final_pareto)}           ║
        ║                                      ║
        ╚══════════════════════════════════════╝
        """

        ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # 7. Efficiency Distribution
        ax7 = fig.add_subplot(gs[3, 1])
        
        all_efficiencies = [p['efficiency'] for p in self.population]
        pareto_efficiencies = [p['efficiency'] for p in final_pareto]
        
        ax7.hist(all_efficiencies, bins=15, alpha=0.6, color='lightblue', 
                label='All Population', edgecolor='black')
        ax7.hist(pareto_efficiencies, bins=10, alpha=0.8, color='red', 
                label='Pareto Front', edgecolor='darkred')
        
        ax7.set_xlabel('Efficiency Score', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax7.set_title('⚖️ Efficiency Distribution', fontsize=14, fontweight='bold', pad=10)
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=0.3, linestyle='--', axis='y')

        plt.suptitle('🔬 Multi-Objective Neural Architecture Search Results',
                    fontsize=18, fontweight='bold', y=0.995)

        plt.savefig('nas_complete_results.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("\n📊 Results saved to 'nas_complete_results.png'")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Use more data for better results (can use full 50k for best results)
    train_size = 30000  # Increased from 10k/20k
    x_train = x_train[:train_size]
    y_train = y_train[:train_size]
    x_val = x_test[:2000]
    y_val = y_test[:2000]

    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Input shape: {x_train.shape[1:]}\n")

    # Configure search
    config = SearchConfig(
        population_size=6,      # 6-10 for good results
        max_generations=5,      # 5-10 generations
        mutation_rate=0.3,      # 30% mutation rate
        epochs_per_eval=10,     # 20 epochs per architecture
        batch_size=128,
        early_stopping_patience=5,
        reduce_lr_patience=3
    )

    # Initialize NAS
    nas = EvolutionaryNAS(
        input_shape=(32, 32, 3),
        num_classes=10,
        config=config
    )

    # Run search
    best_arch = nas.search(x_train, y_train, x_val, y_val)

    # Visualize results
    nas.plot_results()

    # Build and display final model
    print("\n" + "="*80)
    print("🏗️  BUILDING FINAL MODEL")
    print("="*80)

    best_model = nas.builder.build_model(best_arch)

    print("\n📐 Model Architecture Summary:")
    print("-"*80)
    best_model.summary()

    print("\n" + "="*80)
    print("💾 SAVE AND DEPLOY")
    print("="*80)

    # Save architecture
    with open('best_architecture.json', 'w') as f:
        json.dump(best_arch, f, indent=2)
    print("✅ Architecture saved to 'best_architecture.json'")

    # Save model weights
    best_model.save('nas_best_model.h5')
    print("✅ Model saved to 'nas_best_model.h5'")

    print("\n" + "="*80)
    print("📚 NEXT STEPS")
    print("="*80)
    print("""
    1. Train the best model for more epochs on full dataset:

       best_model.fit(
           x_train_full, y_train_full,
           validation_data=(x_test, y_test),
           epochs=100,
           callbacks=[early_stopping, reduce_lr, checkpoint]
       )

    2. Evaluate on test set:

       test_loss, test_acc = best_model.evaluate(x_test, y_test)
       print(f'Test Accuracy: {test_acc:.4f}')

    3. Use the architecture for transfer learning or other datasets

    4. Export to TensorFlow Lite for mobile deployment:

       converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
       tflite_model = converter.convert()
    """)

    print("="*80)
    print("✨ NEURAL ARCHITECTURE SEARCH COMPLETE!")
    print("="*80)

    # Optional: Quick evaluation on validation set
    print("\n📊 Quick Evaluation on Validation Set:")
    best_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    val_loss, val_acc = best_model.evaluate(x_val, y_val, verbose=0)
    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Validation Accuracy: {val_acc:.4f}")

    print("\n🎉 All done! Check 'nas_complete_results.png' for visualizations.")