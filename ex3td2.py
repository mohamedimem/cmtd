import CMTD.mark as mark
import numpy as np



def print_with_dashes(argument):
    line = '-' * 15
    line2 = ' ' * 15  # You can adjust the number of dashes as needed
    print(f"{line2}{line2}")
    print(f"{line2}{line2}")
    print(f"{line}{argument}{line}")





n = 3 # Taille de la matrice carrée
# Example transition matrix and initial probability vector
M_matrix = np.array([[0, 0.5, 0.5],
              [0.5, 0, 0.5],
              [1, 0, 0]])
pi0_vector = np.array([0, 0, 1])# Vecteur pi0






                                                             ##question1
print_with_dashes('question 1')

# Create an instance of the MarkovChain class
markov_instance = mark.MarkovChain(M_matrix, pi0_vector)

# Use the markov method
result_markov = markov_instance.markov(2)
print(result_markov)


# Check if the matrix is irreducible
result_irreducible = markov_instance.is_irreducible()
if result_irreducible:
    print("La matrice est irréductible.")
else:
    print("La matrice n'est pas irréductible.")
                                                      ##question2

print_with_dashes('question 2')
markov_instance.check_global_periodicity()

print_with_dashes('question 3')
# question3
markov_instance.check_transitivity()
markov_instance.check_recurrant()
# question4
print_with_dashes('question 4')
markov_instance.is_stationary()



# Use the draw_graph_from_matrix method
markov_instance.draw_graph_from_matrix()





