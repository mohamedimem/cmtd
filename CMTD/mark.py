import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class MarkovChain:
    def __init__(self, M, pi0):
        # Vérifier que la matrice est carrée
        if M.shape[0] != M.shape[1]:
            raise ValueError("La matrice M doit être carrée.")

        n = M.shape[0]  # Nombre de lignes (et de colonnes) de la matrice M
        # Vérifier que le nombre de lignes du vecteur pi0 est égal au rang de la matrice
        if pi0.shape[0] != n:
            raise ValueError("Le nombre de lignes du vecteur pi0 doit être égal au rang de la matrice M.")

        self.M = M
        self.pi0 = pi0

    def markov(self, m):
        # Vérifier que l'entier m est supérieur à 1
        if m <= 1:
            raise ValueError("L'entier m doit être supérieur à 1.")

        # Calculer le produit pi0 * M^m
        result = np.dot(self.pi0, np.linalg.matrix_power(self.M, m))
        return result

    def draw_graph_from_matrix(self):
        # Create a directed graph from the matrix
        G = nx.DiGraph()

        # Add nodes to the graph
        for i in range(len(self.M)):
            G.add_node(i)

        # Add weighted edges to the graph
        for i in range(len(self.M)):
            for j in range(len(self.M[i])):
                if self.M[i, j] != 0:
                    G.add_edge(i, j, weight=self.M[i, j])

        # Set the positions using spring layout
        pos = nx.spring_layout(G)

        # Uncomment the next line if you want to set positions manually

        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
        plt.axvline(0.1, alpha=0.1, color='green')
        plt.axhline(0.3, alpha=0.1, color='green')

        # Make the graph - add the pos and connectionstyle arguments
        nx.draw(G, with_labels=True, pos=pos,
                node_size=1500, alpha=0.3, font_weight="bold", arrows=True,
                connectionstyle='arc3, rad = 0.1')

        plt.axis('on')
        plt.show()

    def is_irreducible(self):
        num_states = len(self.M)

        # Vérifiez toutes les paires d'états possibles
        for i in range(num_states):
            for j in range(num_states):
                # Si i == j, pas besoin de vérifier le chemin
                if i != j:
                    # Vérifiez s'il existe un chemin de i à j
                    if not self.has_path(i, j):
                        return False

        # Si aucun contre-exemple n'est trouvé, la matrice est irréductible
        return True

    def has_path(self, start, target):
        num_states = len(self.M)
        visited = [False] * num_states
        stack = [start]

        while stack:
            current_state = stack.pop()
            visited[current_state] = True

            # Si le chemin est trouvé
            if current_state == target:
                return True

            # Ajouter les voisins non visités à la pile
            for neighbor, probability in enumerate(self.M[current_state]):
                if probability > 0 and not visited[neighbor]:
                    stack.append(neighbor)

        # Aucun chemin trouvé
        return False

    def compute_period(self, state):
        num_steps = 1
        current_prob = self.M[state, state]

        while current_prob == 0:
            num_steps += 1
            current_prob = np.linalg.matrix_power(self.M, num_steps)[state, state]

        return num_steps

    def calculate_pgcd(self, state, max_steps):
        pgcd_values = []

        for i in range(1, max_steps + 1):
            prob_ij = np.linalg.matrix_power(self.M, i)[state, state]
            if prob_ij > 0:
                pgcd_values.append(i)

        pgcd_values_np = np.array(pgcd_values)

        if pgcd_values_np.size > 0:
            return np.gcd.reduce(pgcd_values_np)
        else:
            return 0  # GCD default is 0 for an empty sequence

    def check_global_periodicity(self):
        flag = True
        max_steps = 10  # You can adjust this based on the size of your chain

        # Initialize a boolean variable to track whether any state is aperiodic
        is_aperiodic = False

        for state in range(self.M.shape[0]):
            period = self.compute_period(state)
            pgcd = self.calculate_pgcd(state, max_steps)

            if pgcd == 1:
                # print(f"L'état {state} est apériodique.")
                is_aperiodic = True  # Set the flag to True if any state is aperiodic
                flag = False
            # else:
            #     print(f"L'état {state} est périodique avec une période de {period}.")

        if is_aperiodic:
            print("La matrice est apériodique.")
        else:
            print("La matrice est périodique.")
        return flag

    def check_transitivity(self):
        for state in range(self.M.shape[0]):
            # Check if state is recurrent or transient
            is_recurrent = np.linalg.matrix_power(self.M, int(1e6))[state, state] == 1
            is_transient = not is_recurrent

            if is_recurrent:
                print(f"L'état {state} est récurrent.")
            else:
                print(f"L'état {state} est transient.")

    def check_recurrant(self):
        all_recurrent = True  # Assume all states are recurrent until proven otherwise

        for state in range(self.M.shape[0]):
            # Check if state is recurrent or transient
            is_recurrent = np.linalg.matrix_power(self.M, int(1e6))[state, state] == 1

            if not is_recurrent:
                all_recurrent = False

        if all_recurrent == True:
            print('Donc tous les états sont récurrents.')
        else:
            print('Donc tous les états ne sont pas récurrents.')
        return all_recurrent

    def is_stationary(self):
        num_states = len(self.M)

        # Check irreducibility
        irreducible_flag = self.is_irreducible()

        # Check aperiodicity
        max_steps = 10  # You can adjust this based on the size of your chain
        aperiodic_flag = True  # Assume aperiodic until proven periodic

        for state in range(num_states):
            period = self.compute_period(state)
            pgcd = self.calculate_pgcd(state, max_steps)

            if pgcd != 1:
                aperiodic_flag = False
                break  # No need to check further, the chain is proven to be periodic

        # If the chain is irreducible and aperiodic, it has a stationary distribution
        if irreducible_flag and aperiodic_flag:
            print("La chaîne de Markov admet un régime stationnaire.")
        else:
            print("La chaîne de Markov n'admet pas de régime stationnaire.")

    def is_ergodic(self):
        num_states = len(self.M)

        # Check aperiodicity and recurrence for all states
        aperiodic_flag = not self.check_global_periodicity()
        print('_________')
        recurrent_flag = self.check_recurrant()
        print('_________')

        # If the chain is aperiodic and all states are recurrent, it is ergodic
        if aperiodic_flag and recurrent_flag:
            print("La chaîne de Markov est ergodique.")
        else:
            print("La chaîne de Markov n'est pas ergodique.")
